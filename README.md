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
```
coming soon
```

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
