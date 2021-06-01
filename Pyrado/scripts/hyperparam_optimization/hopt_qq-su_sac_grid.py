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
Optimize the hyper-parameters of Soft Actor Critic for the Quanser Qube swing-up task.
"""
import functools
import os.path as osp

import optuna
import torch as to
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.step_based.sac import SAC
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.feed_back.two_headed_fnn import TwoHeadedFNNPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.spaces import ValueFunctionSpace
from pyrado.spaces.box import BoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt


def train_and_eval(trial: optuna.Trial, study_dir: str, seed: int):
    """
    Objective function for the Optuna `Study` to maximize.

    .. note::
        Optuna expects only the `trial` argument, thus we use `functools.partial` to sneak in custom arguments.

    :param trial: Optuna Trial object for hyper-parameter optimization
    :param study_dir: the parent directory for all trials in this study
    :param seed: seed value for the random number generators, pass `None` for no seeding
    :return: objective function value
    """

    # Environment
    env_hparams = dict(dt=1 / 100.0, max_steps=600)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    # Learning rate scheduler
    lrs_gamma = trial.suggest_float("exp_lr_scheduler_gamma", low=0, high=0.999)
    if lrs_gamma == 0:
        lrs_gamma = None
    if lrs_gamma is not None:
        lr_sched = lr_scheduler.ExponentialLR
        lr_sched_hparam = dict(gamma=lrs_gamma)
    else:
        lr_sched, lr_sched_hparam = None, dict()

    fnn_size = trial.suggest_categorical("fnn_size", choices=[16, 32, 64])
    # Policy
    policy_hparam = dict(shared_hidden_sizes=[fnn_size, fnn_size], shared_hidden_nonlin=to.relu)
    policy = TwoHeadedFNNPolicy(spec=env.spec, **policy_hparam)

    qfnc_param = dict(hidden_sizes=[fnn_size, fnn_size], hidden_nonlin=to.relu)
    combined_space = BoxSpace.cat([env.obs_space, env.act_space])
    q1 = FNNPolicy(spec=EnvSpec(combined_space, ValueFunctionSpace), **qfnc_param)
    q2 = FNNPolicy(spec=EnvSpec(combined_space, ValueFunctionSpace), **qfnc_param)

    algo_hparam = dict(
        max_iter=500,
        memory_size=100_000,
        gamma=0.995,
        num_updates_per_step=1_000,
        tau=0.995,
        ent_coeff_init=trial.suggest_categorical("ent_coeff", [0.1, 0.2, 0.3, 0.4]),
        learn_ent_coeff=False,
        target_update_intvl=1,
        num_init_memory_steps=None,
        standardize_rew=False,
        min_steps=trial.suggest_categorical("min_steps", [10, 20, 30]) * env.max_steps,
        batch_size=256,
        lr=trial.suggest_categorical("lr", [3e-3, 3e-4, 3e-5, 3e-6]),
        max_grad_norm=10,
        num_workers=1,
        lr_scheduler=lr_sched,
        lr_scheduler_hparam=lr_sched_hparam,
    )

    csv_logger = create_csv_step_logger(osp.join(study_dir, f"trial_{trial.number}"))
    algo = SAC(osp.join(study_dir, f"trial_{trial.number}"), env, policy, q1, q2, **algo_hparam, logger=csv_logger)

    # Train without saving the results
    algo.train(snapshot_mode="latest", seed=seed)

    # Evaluate
    min_rollouts = 1000
    sampler = ParallelRolloutSampler(env, policy, num_workers=args.num_workers // 4, min_rollouts=min_rollouts)
    ros = sampler.sample()
    mean_ret = sum([r.undiscounted_return() for r in ros]) / min_rollouts

    return mean_ret


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument("--repetitions", default=4, type=int)
    args = parser.parse_args()

    if args.dir is None:
        ex_dir = setup_experiment("hyperparams", QQubeSwingUpSim.name, f"{SAC.name}_{FNNPolicy.name}_100Hz_actnorm")
        study_dir = osp.join(pyrado.TEMP_DIR, ex_dir)
        print_cbt(f"Starting a new Optuna study.", "c", bright=True)
    else:
        study_dir = args.dir
        if not osp.isdir(study_dir):
            raise pyrado.PathErr(given=study_dir)
        print_cbt(f"Continuing an existing Optuna study.", "c", bright=True)

    search_space = {
        "exp_lr_scheduler_gamma": [0, 0.995],
        "fnn_size": [16, 32],
        "ent_coeff": [0.1, 0.2, 0.3, 0.4],
        "min_steps": [10, 20, 30],
        "lr": [3e-3, 3e-4, 3e-5, 3e-6],
    }
    sampler = optuna.samplers.GridSampler(search_space=search_space)

    name = f"{QQubeSwingUpSim.name}_{SAC.name}_{FNNPolicy.name}_100Hz_actnorm_grid"
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:////{osp.join(study_dir, f'{name}.db')}",
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    # Start optimizing
    for rep in range(args.repetitions):
        study.optimize(functools.partial(train_and_eval, study_dir=study_dir, seed=args.seed), n_jobs=12)
        save_dicts_to_yaml(
            study.best_params,
            dict(seed=args.seed),
            save_dir=study_dir,
            file_name=f"best_hyperparams_rep{rep}",
        )
