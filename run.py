from pathlib import Path
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
@click.option('--save_loc', default = PathConfig().checkpoints, help = 'Location to save the results')
@click.option('--timestep', default = 10000, type = int, help = 'Number of timesteps to run the training for')
@click.option('--test', is_flag = True, help = 'Whether to run the test')
@click.option('--step_factor', default = 2, type = int, help = 'Factor to determine the number of steps in an episode')
@click.option('--n_vertices', default = 20, type = int, help = 'Number of vertices in the graph')
def run(timestep, n_vertices, step_factor, test, save_loc):
    
    if isinstance(save_loc, str):
        save_loc = Path(save_loc)
    save_loc = save_loc / f'BA_{n_vertices}vertices'
    
    train_run(n_vertices, timestep, step_factor, save_loc)
    if test:
        test_run(n_vertices, step_factor, save_loc)
    
    # test_run(save_loc)
    
if __name__ == '__main__':
    run()