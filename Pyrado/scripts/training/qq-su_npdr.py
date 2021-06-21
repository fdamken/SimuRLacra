# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Script to identify the domain parameters of the Pendulum environment using Neural Posterior Domain Randomization
"""
import os.path as osp

import sbi.utils as sbiutils
import torch as to
import torch.nn as nn
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.dummy import DummyPolicy
from pyrado.policies.feed_forward.time import TimePolicy
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding, DeltaStepsEmbedding, RNNEmbedding
from pyrado.utils.argparser import get_argparser
from pyrado.utils.sbi import create_embedding


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    seed_str, num_segs_str, len_seg_str = "", "", ""
    if args.seed is not None:
        seed_str = f"_seed-{args.seed}"
    if args.num_segments is not None:
        num_segs_str = f"numsegs-{args.num_segments}"
    elif args.len_segments is not None:
        len_seg_str = f"lensegs-{args.len_segments}"
    else:
        raise pyrado.ValueErr(msg="Either num_segments or len_segments must not be None, but not both or none!")

    use_rec_act = True
    ectl = True

    # Experiment (set seed before creating the modules)
    if ectl:
        ex_dir = setup_experiment(
            QQubeSwingUpSim.name,
            f"{NPDR.name}_{QQubeSwingUpAndBalanceCtrl.name}",
            num_segs_str + len_seg_str + seed_str,
        )
        t_end = 5  # s
    else:
        ex_dir = setup_experiment(
            QQubeSwingUpSim.name,
            f"{NPDR.name}_{TimePolicy.name}",
            num_segs_str + len_seg_str + seed_str,
        )
        t_end = 10  # s

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(dt=1 / 250.0, max_steps=int(t_end * 250))
    env_sim = QQubeSwingUpSim(**env_sim_hparams)
    # env_sim = ActDelayWrapper(env_sim)

    # Create the ground truth target domain and the behavioral policy
    if ectl:
        env_real = osp.join(pyrado.EVAL_DIR, "qq-su_ectrl_250Hz")  # 5s long
        policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec)  # replaced by the recorded actions if use_rec_act=True
    else:
        env_real = osp.join(pyrado.EVAL_DIR, "qq_sin_2Hz_1V_250Hz")
        assert use_rec_act
        policy = DummyPolicy(env_sim.spec)  # replaced by recorded real actions

    # Define a mapping: index - domain parameter
    # dp_mapping = {0: "V_thold_neg", 1: "V_thold_pos"}
    # dp_mapping = {0: "V_thold_neg", 1: "V_thold_pos", 2: "act_delay"}
    # dp_mapping = {0: "damping_rot_pole", 1: "Dp"}
    # dp_mapping = {0: "damping_rot_pole", 1: "Dp", 2: "Rm", 3: "km"}
    # dp_mapping = {0: "Rm", 1: "km", 2: "mass_rot_pole", 3: "mass_pend_pole"}
    # dp_mapping = {0: "damping_rot_pole", 1: "Dp", 2: "Rm", 3: "km", 4: "mass_rot_pole", 5: "mass_pend_pole", 6: "length_rot_pole", 7: "Lp", 8: "g"}
    dp_mapping = {
        0: "damping_rot_pole",
        1: "Dp",
        2: "Rm",
        3: "km",
        4: "mass_rot_pole",
        5: "mass_pend_pole",
        6: "length_rot_pole",
        7: "Lp",
        8: "g",
        9: "act_delay",
    }
    # dp_mapping = {
    #     0: "damping_rot_pole",
    #     1: "Dp",
    #     2: "Rm",
    #     3: "km",
    #     4: "mass_rot_pole",
    #     5: "mass_pend_pole",
    #     6: "length_rot_pole",
    #     7: "Lp",
    #     8: "g",
    #     # 9: "V_thold_neg",
    #     # 10: "V_thold_pos",
    #     11: "act_delay",
    # }

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        # low=to.tensor([-0.1, 0.0, 0]),
        # high=to.tensor([0.0, 0.1, 10])
        # low=to.tensor([0.0, 0.0]),
        # high=to.tensor([2 * 0.0015, 2 * 0.0005]),
        # low=to.tensor([dp_nom["damping_rot_pole"] * 0, dp_nom["Dp"] * 0, dp_nom["Rm"] * 0.5, dp_nom["km"] * 0.5]),
        # high=to.tensor([dp_nom["damping_rot_pole"] * 10, dp_nom["Dp"] * 10, dp_nom["Rm"] * 2.0, dp_nom["km"] * 2.0]),
        # low=to.tensor([dp_nom["Rm"] * 0.5, dp_nom["km"] * 0.5, dp_nom["mass_rot_pole"] * 0.5, dp_nom["mass_pend_pole"] * 0.5]),
        # high=to.tensor([dp_nom["Rm"] * 1.5, dp_nom["km"] * 1.5, dp_nom["mass_rot_pole"] * 1.5, dp_nom["mass_pend_pole"] * 1.5]),
        low=to.tensor(
            [
                dp_nom["damping_rot_pole"] * 0,
                dp_nom["Dp"] * 0,
                dp_nom["Rm"] * 0.1,
                dp_nom["km"] * 0.2,
                dp_nom["mass_rot_pole"] * 0.3,
                dp_nom["mass_pend_pole"] * 0.3,
                dp_nom["length_rot_pole"] * 0.5,
                dp_nom["Lp"] * 0.5,
                dp_nom["g"] * 0.85,
                # -0.1,
                # 0.0,
                0,
            ]
        ),
        high=to.tensor(
            [
                dp_nom["damping_rot_pole"] * 5,
                dp_nom["Dp"] * 20,
                dp_nom["Rm"] * 1.9,
                dp_nom["km"] * 1.8,
                dp_nom["mass_rot_pole"] * 1.7,
                dp_nom["mass_pend_pole"] * 1.7,
                dp_nom["length_rot_pole"] * 1.5,
                dp_nom["Lp"] * 1.5,
                dp_nom["g"] * 1.15,
                # 0,
                # 0.1,
                5,
            ]
        ),
    )
    prior = sbiutils.BoxUniform(**prior_hparam)

    # Time series embedding
    lstm = pyrado.load("policy.pt", osp.join(pyrado.EXP_DIR, "qq-tspred", "lstm", "2021-05-31_19-48-32"))
    # lstm = pyrado.load("policy.pt", osp.join(pyrado.EXP_DIR, "qq-tspred", "lstm", "2021-06-02_17-01-07"))
    embedding_hparam = dict(
        downsampling_factor=1,
        # state_mask_labels=(0, 1, 4),
        len_rollouts=env_sim.max_steps,
        recurrent_network_type=nn.LSTM,
        hidden_size=lstm.rnn_layers.hidden_size,
        num_recurrent_layers=lstm.num_recurrent_layers,
        output_size=lstm.output_layer.out_features,
    )
    embedding = create_embedding(RNNEmbedding.name, env_sim.spec, **embedding_hparam)
    embedding.init_param(init_values=lstm.param_values)  # only for RNNEmbedding

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=50, num_transforms=5)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_real_rollouts=3,
        num_sim_per_round=500,
        num_sbi_rounds=4,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=2,
        num_segments=args.num_segments,
        len_segments=args.len_segments,
        stop_on_done=False,
        use_rec_act=use_rec_act,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=3e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=True,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
            # max_num_epochs=5,  # only use for debugging
        ),
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=True),
        num_workers=20,
    )
    algo = NPDR(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        embedding,
        subrtn_sbi_class=SNPE_C,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_sim_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)
