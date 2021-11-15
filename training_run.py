import json
from argparse import Namespace
from pathlib import Path

import gym
import numpy as np
import yaml
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES
import crafter
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from sample_factory.envs.env_wrappers import PixelFormatChwWrapper

from sample_factory.utils.utils import log

import wandb
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm

import sys
from utils.config_validation import Experiment
from wrappers import CrafterStats


def make_crafter(full_env_name, cfg=None, env_config=None):
    env = gym.make('CrafterReward-v1')
    env = PixelFormatChwWrapper(env)
    env = CrafterStats(env, )
    env = MultiAgentWrapper(env)

    return env


def crafter_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    score = np.mean(policy_avg_stats["Score"])
    log.debug(f'Score: {round(float(score))}')
    summary_writer.add_scalar('Score', score, env_steps)

    num_achievements = np.mean(policy_avg_stats["Num_achievements"])
    log.debug(f'Num_achievements: {round(float(num_achievements), 3)}')
    summary_writer.add_scalar('Num_achievements', num_achievements, env_steps)


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='CrafterReward-v1',
        make_env_func=make_crafter,
    )

    EXTRA_PER_POLICY_SUMMARIES.append(crafter_extra_summaries)


def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            full_config=exp.dict()
                            )
    return exp, flat_config


def main():
    register_custom_components()

    import argparse

    parser = argparse.ArgumentParser(description='Process training config.')

    parser.add_argument('--config_path', type=str, action="store",
                        help='path to yaml file with single run configuration', required=False)

    parser.add_argument('--raw_config', type=str, action='store',
                        help='raw json config', required=False)

    parser.add_argument('--wandb_thread_mode', type=bool, action='store', default=False,
                        help='Run wandb in thread mode. Usefull for some setups.', required=False)

    params = parser.parse_args()

    if params.raw_config:
        config = json.loads(params.raw_config)
    else:
        if params.config_path is None:
            config = Experiment().dict()
        else:
            with open(params.config_path, "r") as f:
                config = yaml.safe_load(f)

    exp, flat_config = validate_config(config)
    log.debug(exp.global_settings.experiments_root)

    if exp.global_settings.use_wandb:
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project=exp.name, config=exp.dict(), save_code=False, sync_tensorboard=True)

    status = run_algorithm(flat_config)
    if exp.global_settings.use_wandb:
        import shutil
        path = Path(exp.global_settings.train_dir) / exp.global_settings.experiments_root
        zip_name = str(path)
        shutil.make_archive(zip_name, 'zip', path)
        wandb.save(zip_name + '.zip')
    return status


if __name__ == '__main__':
    sys.exit(main())
