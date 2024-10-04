from argparse import Namespace
from typing import Any, Callable, Dict, List, Type

import lightning as L
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger


class LoggerSaveConfigCallback(SaveConfigCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_config(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(
                self.config,
                skip_none=False,
            )
            trainer.logger.log_hyperparams({"config": config})


class NeuroSAMCLI(LightningCLI):
    """Custom Lightning CLI for NeuroSAM"""

    def __init__(self,
                 model_class: LightningModule = None,
                 datamodule_class: LightningDataModule = None,
                 save_config_callback: SaveConfigCallback = SaveConfigCallback,
                 save_config_kwargs: Dict[str, Any] | None = None,
                 trainer_class: L.Trainer = Trainer,
                 trainer_defaults: Dict[str, Any] | None = None,
                 seed_everything_default: bool | int = True,
                 parser_kwargs: Dict[str, Any] | Dict[str, Dict[str, Any]] | None = None,
                 subclass_mode_model: bool = False,
                 subclass_mode_data: bool = False,
                 args: List[str] | Dict[str, Any] | Namespace | None = None,
                 run: bool = True,
                 auto_configure_optimizers: bool = True
                 ) -> None:
        super().__init__(
            model_class,
            datamodule_class,
            save_config_callback,
            save_config_kwargs,
            trainer_class,
            trainer_defaults,
            seed_everything_default,
            parser_kwargs,
            subclass_mode_model,
            subclass_mode_data,
            args,
            run,
            auto_configure_optimizers
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        
        parser.link_arguments("data.img_size", "model.img_size")
