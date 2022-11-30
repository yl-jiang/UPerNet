import yaml
from pathlib import Path

cwd = Path("__file__").absolute().parent

class Config:
    def __init__(self) -> None:
        self.config = {}

    def update_config(self, args):
        for k, v in args.__dict__.items():
            self.config.update({k: v}) 

    def get_config(self, cfg, args=None):
        assert Path(cfg).exists(), f"config file: {cfg} is not exists!"
        configs = yaml.load(open(str(cfg)), Loader=yaml.FullLoader)
        for k, v in configs.items():
            self.config.update(v)
        if args:
            self.update_config(args)
        return self.config


if __name__ == "__main__":
    config = Config()