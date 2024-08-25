from pathlib import Path
import logging

import torch

from src.configs.config_base import ConfigBase


class PathConfig(ConfigBase):
    root = Path(__file__).resolve().parents[2]
    src = root / 'src'
    checkpoints = root / 'checkpoints'
    scripts = root / 'scripts'
    tests = root / 'tests'
    logs = checkpoints / 'logs'

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


class LoggerConfig(ConfigBase):
    level = logging.INFO
    logs_dir = PathConfig().logs


class OtherConfig(ConfigBase):
    device = 'default'

    def __post_init__(self) -> None:
        if self.device == 'default':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'  
