import json
import os
import subprocess
import time
import argparse

from utils.config_validation import Experiment
from utils.gs2dict import generate_variants
import yaml


def start_training_runs(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = []
    for resolved_vars, spec in generate_variants(config):
        joined_vars = ";".join([f'{key[-1]}={value}' for key, value in resolved_vars.items()])
        if joined_vars:
            spec['global_settings']['experiments_root'] = joined_vars
        # validate config
        Experiment(**spec)

        cmd = f"python3 training_run.py --wandb_thread_mode=True --raw_config='{json.dumps(spec)}'"

        env_vars = os.environ.copy()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=env_vars)
        output, err = process.communicate()

        exit_code = process.wait()

        if exit_code != 0:
            break

        time.sleep(5)

    return results


def main():
    parser = argparse.ArgumentParser(description='Process training config.')
    parser.add_argument('--config_path', type=str, action="store",
                        help='path to yaml file with single run configuration', required=False,
                        default='crafter_baseline.yaml')
    params = parser.parse_args()
    start_training_runs(params.config_path)


if __name__ == '__main__':
    main()
