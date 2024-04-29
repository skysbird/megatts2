# main.py
from lightning.pytorch.cli import LightningCLI

from trainer.trainer2 import MegaADMTrainer
from modules.datamodule import TTSDataModule, test



def cli_main():
    cli = LightningCLI(MegaADMTrainer, TTSDataModule)

if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block