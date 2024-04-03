# main.py
from lightning.pytorch.cli import LightningCLI

from trainer.trainer2 import MegaGANTrainer
from modules.datamodule import TTSDataModule2, test



def cli_main():
    cli = LightningCLI(MegaGANTrainer, TTSDataModule2)

if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
