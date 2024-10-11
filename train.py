import lightning as L
import torch

from segment_anything import NeuroSamModel
from utils.cli_utils import LoggerSaveConfigCallback, NeuroSAMCLI
from utils.data_module import NeuroSamDataModule

torch.set_float32_matmul_precision('medium')


def cli_main():
    cli = NeuroSAMCLI(
        NeuroSamModel,
        NeuroSamDataModule,
        seed_everything_default=2024,
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs={
            "save_to_log_dir": True,
            "overwrite": True,
        },
        parser_kwargs={
            "parser_mode": "omegaconf",
            "fit": {
                "default_config_files": [
                    "configs/all.yaml",
                ]
            }
        },
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli_main()

