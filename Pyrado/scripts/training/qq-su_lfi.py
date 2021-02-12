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
Sim-to-sim experiment on the Quanser Qube environment using likelihood-free inference
"""
import numpy as np
import os.path as osp
import torch as to
import torch.nn as nn
from sbi.inference import SNPE
from sbi import utils

import pyrado
from pyrado.algorithms.inference.lfi import LFI
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.special.time import TimePolicy
from pyrado.utils.argparser import get_argparser


def fcn_of_time(t: float):
    act = 1.0 * np.sin(2 * np.pi * t * 2.0)  # 2 Hz
    return act.repeat(env_sim.act_space.flat_dim)


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{LFI.name}_{QQubeSwingUpAndBalanceCtrl.name}")
    # ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{LFI.name}_{TimePolicy.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 250.0, max_steps=1500)
    env_sim = QQubeSwingUpSim(**env_hparams)

    # Create a fake ground truth target domain
    num_real_obs = 1
    env_real = osp.join(pyrado.EVAL_DIR, "qq-su_ectrl_250Hz")
    # env_real = osp.join(pyrado.EVAL_DIR, "qq_sin_250Hz_1V")

    # dp_mapping = {0: "Dr", 1: "Dp", 2: "Rm", 3: "km"}
    dp_mapping = {0: "Dr", 1: "Dp", 2: "Rm", 3: "km", 4: "Mr", 5: "Mp", 6: "Lr", 7: "Lp"}

    # Policy
    behavior_policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec)
    # behavior_policy = TimePolicy(env_sim.spec, fcn_of_time, env_sim.dt)

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        # low=to.tensor([dp_nom["Dr"] * 0, dp_nom["Dp"] * 0, dp_nom["Rm"] * 0.5, dp_nom["km"] * 0.5]),
        # high=to.tensor([dp_nom["Dr"] * 10, dp_nom["Dp"] * 10, dp_nom["Rm"] * 2.0, dp_nom["km"] * 2.0]),
        low=to.tensor(
            [
                dp_nom["Dr"] * 0,
                dp_nom["Dp"] * 0,
                dp_nom["Rm"] * 0.5,
                dp_nom["km"] * 0.5,
                dp_nom["Mr"] * 0.5,
                dp_nom["Mp"] * 0.5,
                dp_nom["Lr"] * 0.5,
                dp_nom["Lp"] * 0.5,
            ]
        ),
        high=to.tensor(
            [
                dp_nom["Dr"] * 100,
                dp_nom["Dp"] * 100,
                dp_nom["Rm"] * 2.0,
                dp_nom["km"] * 2.0,
                dp_nom["Mr"] * 2.0,
                dp_nom["Mp"] * 2.0,
                dp_nom["Lr"] * 2.0,
                dp_nom["Lp"] * 2.0,
            ]
        ),
    )
    prior = utils.BoxUniform(**prior_hparam)
    posterior_nn_hparam = dict(model="maf", embedding_net=nn.Identity(), hidden_features=50, num_transforms=5)

    # Algorithm
    algo_hparam = dict(
        summary_statistic="bayessim",  # bayessim or dtw_distance
        max_iter=10,
        num_real_rollouts=num_real_obs,
        num_sim_per_real_rollout=4000,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=200,
        num_workers=10,
        sbi_sampling_hparam=dict(sample_with_mcmc=True),
    )
    algo = LFI(
        ex_dir,
        env_sim,
        env_real,
        behavior_policy,
        dp_mapping,
        prior,
        posterior_nn_hparam,
        SNPE,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(posterior_nn=posterior_nn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)