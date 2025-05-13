import os

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import ReduceLROnPlateau

from train.data import DeepCleanDataset
from train.model import DeepClean
from utils.logging import configure_logging


class AframeCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--verbose", type=bool, default=False)
        parser.add_argument("--output-dir", type=str, default=None)

        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.OneCycleLR)
        # parser.add_lr_scheduler_args(ReduceLROnPlateau)

        parser.link_arguments(
            "data.num_witnesses",
            "model.arch.init_args.num_witnesses",
            apply_on="instantiate",
        )

        # link data arguments to loss function
        parser.link_arguments(
            "data.sample_rate",
            "model.loss.init_args.sample_rate",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.freq_low", "model.loss.init_args.freq_low", apply_on="parse"
        )
        parser.link_arguments(
            "data.freq_high",
            "model.loss.init_args.freq_high",
            apply_on="parse",
        )

        # link data arguments to metric
        parser.link_arguments(
            "data.inference_sampling_rate",
            "model.metric.init_args.inference_sampling_rate",
            apply_on="parse",
        )
        parser.link_arguments(
            "data.sample_rate",
            "model.metric.init_args.sample_rate",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.clean_kernel_length",
            "model.metric.init_args.clean_kernel_length",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.clean_stride",
            "model.metric.init_args.clean_stride",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.bandpass",
            "model.metric.init_args.bandpass",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.y_scaler",
            "model.metric.init_args.y_scaler",
            apply_on="instantiate",
        )

        # link optimizer and scheduler args
        parser.link_arguments(
            "data.steps_per_epoch",
            "lr_scheduler.steps_per_epoch",
            apply_on="instantiate",
        )
        parser.link_arguments("optimizer.lr", "lr_scheduler.max_lr")
        parser.link_arguments("trainer.max_epochs", "lr_scheduler.epochs")


def main(args=None):
    cli = AframeCLI(
        model_class=DeepClean,
        datamodule_class=DeepCleanDataset,
        seed_everything_default=101588,
        run=False,
        parser_kwargs={"default_env": True},
        save_config_kwargs={"overwrite": True},
        args=args,
    )

    log_dir = cli.trainer.logger.log_dir or cli.trainer.logger.save_dir
    if not log_dir.startswith("s3://"):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "train.log")
        configure_logging(log_file)
    else:
        configure_logging()

    cli.trainer.fit(cli.model, cli.datamodule)
    if cli.datamodule.hparams.test_duration > 0:
        weights = torch.load(cli.trainer.checkpoint_callback.best_model_path)
        cli.model.load_state_dict(weights["state_dict"])
        cli.trainer.test(cli.model, cli.datamodule)

    for i in ["X", "y"]:
        name = f"{i}_scaler"
        module = getattr(cli.datamodule, name)
        torch.save(module.state_dict(), os.path.join(log_dir, f"{name}.pt"))


if __name__ == "__main__":
    main()