# SampleFactory APPO baseline for Crafter environment


## Installation
Just install all dependencies using:
```bash
pip install -r docker/requirements.txt
```

## Training APPO
Just run ```train.py``` with config_path:
```bash
python main.py --config_path crafter_baseline.yaml
```

## Results:

APPO agent was trained for ~500M environment steps in under 24 hours on a single GPU:

| Achievement | Rainbow | PPO | DreamerV2 | APPO |
| --- | --- | --- | --- | --- |
| Collect Coal | 0.0% | 0.4% | 14.7% | **96.6%** |
| Collect Diamond | 0.0% | 0.0% | 0.0% | **22.6%** |
| Collect Drink | 24.0% | 30.3% | 80.0% | **94.0%** |
| Collect Iron | 0.0% | 0.0% | 0.0% | **83.1%** |
| Collect Sapling |97.4% | 66.7% | 86.6% | **98.9%** |
| Collect Stone | 0.2% | 3.0% | 42.7% | **99.3%** |
| Collect Wood | 74.9% | 83.0% | 92.7% | **99.9%** |
| Defeat Skeleton | 0.7% | 0.2% | 2.6% | **89.3%** |
| Defeat Zombie | 39.6% | 2.0% | 53.1% | **95.2%** |
| Eat Cow | 26.1% | 12.0% | 17.1% | **93.3%** |
| Eat Plant | 0.0% | 0.0% | 0.1% | **1.0%** |
| Make Iron Pickaxe | 0.0% | 0.0% | 0.0% | **77.9%** |
| Make Iron Sword | 0.0% | 0.0% | 0.0% | **55.4%** |
| Make Stone Pickaxe | 0.0% | 0.0% | 0.2% | **97.6%** |
| Make Stone Sword | 0.0% | 0.0% | 0.3% | **98.5%** |
| Make Wood Pickaxe | 4.8% | 21.1% | 59.6% | **99.7%** |
| Make Wood Sword | 9.8% | 20.1% | 40.2% | **99.0%** |
| Place Furnace | 0.0% | 0.1% | 1.8% | **98.0%** |
| Place Plant | 94.2% | 65.0% | 84.4% | **99.0%** |
| Place Stone | 0.0% | 1.7% | 29.0% | **99.0%** |
| Place Table | 52.3% | 66.1% | 85.7% | **99.0%** |
| Wake Up | 93.3% | 92.5% | 92.8% | **97.0%** |
| Score | 4.3% | 4.6% | 10.0% | **50.0%** |

Please consider [Wandb project](https://wandb.ai/cds/crafter-appo-baseline?workspace=user-tviskaron) for more detailed results.


## Docker 
We use [crafting](https://pypi.org/project/crafting/) to automate our experiments. 
You can find an example of running such a pipeline in ```run.yaml``` file. 
You need to have installed Docker, Nvidia drivers, and crafting package. 

The crafting package is available in PyPI:
```bash
pip install crafting
```


To build the image run the command below in ```docker``` folder:
```bash
sh build.sh
```

To run an experiment specify target command in ```command``` field in ```run.yaml``` file and call crafting:
```bash
crafting run.yaml
```

Example of ```run.yaml``` file ():
```yaml
container:
  image: "crafter-appo-baseline:latest"
  command: 'python main.py --config_path crafter_baseline.yaml'
  tty: True
  environment:
    - "WANDB_API_KEY=<YOUR API KEY>"
    - "OMP_NUM_THREADS=1"
    - "MKL_NUM_THREADS=1"
    - "NVIDIA_VISIBLE_DEVICES=0"
code:
  folder: "."

host_config:
  runtime: nvidia
  shm_size: 4g
  mem_limit: 32g
```

Please specify your <WANDB_API_KEY> if you want to save logs in wandb cloud or turn off wandb in the training config.
