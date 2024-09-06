import pathlib
import click

from src.configs.common_configs import PathConfig

from src.train import run as train_run

# save_loc = PathConfig().experiments / 'BA_20vertices'

@click.command()
@click.option('--save_loc', default = PathConfig().checkpoints / 'BA_20vertices', help = 'Location to save the results')
def run(save_loc):
    train_run(save_loc)
    
if __name__ == '__main__':
    run()