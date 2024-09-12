import pathlib
import click

from src.configs.common_configs import PathConfig

from src.train import run as train_run
from src.test import run as test_run

# save_loc = PathConfig().experiments / 'BA_20vertices'

# @click.group()
# def main():
#     pass

# @main.command()
@click.command()
@click.option('--save_loc', default = PathConfig().checkpoints / 'BA_20vertices', help = 'Location to save the results')
@click.option('--timestep', default = 10000, type = int, help = 'Number of timesteps to run the training for')
@click.option('--test', is_flag = True, help = 'Whether to run the test')
def run(timestep, test, save_loc):
    train_run(timestep, save_loc)
    if test:
        test_run(save_loc)
    
    # test_run(save_loc)
    
if __name__ == '__main__':
    run()