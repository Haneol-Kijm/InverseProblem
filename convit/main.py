# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a ResNet50-D on the Imagenette Dataset using components from the torchvision, timm and
# torchmetrics libraries.
# This example demonstrates how the trainer can be extended to incorporate techniques such as mixup and modelEMA
# into a training run.
#
# Note: this example requires installing the torchvision, torchmetrics and timm packages
########################################################################
import os, math
from datetime import datetime

# from trainer import TimmPSNRTrainer
import pytorch_lightning as pl
from lightning.fabric import seed_everything

from trainer import LitModel

# from ema import EMA

from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import Result
from ray.train.torch import TorchCheckpoint, TorchPredictor
from ray.tune import CLIReporter, ResultGrid
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback


def train_tune_checkpoint(
    config, checkpoint_dir=None, num_epochs=100, num_gpus=0, data_dir="~/data"
):
    data_dir = os.path.expanduser(data_dir)
    # data_dir = "C:/Works/vscode/data/imagenet-mini_processed/"
    kwargs = {
        "gradient_clip_val": 1.0,
        "max_epochs": num_epochs,
        # If fractional GPUs passed in, convert to int.
        "accelerator": "gpu" if num_gpus != 0 else "cpu",
        "devices": math.ceil(num_gpus),
        "logger": TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
        "enable_progress_bar": False,
        "callbacks": [
            # *DEFAULT_CALLBACKS,
            # EMA(decay=0.99),
            TuneReportCheckpointCallback(
                metrics={"loss": "ptl/val_loss", "mean_psnr": "ptl/val_psnr"},
                filename="checkpoint",
                on="validation_end",
            ),
        ],
    }

    if checkpoint_dir:
        resume_from_checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    else:
        resume_from_checkpoint = None

    model = LitModel(config=config, data_dir=data_dir)
    trainer = pl.Trainer(**kwargs)

    trainer.fit(model, ckpt_path=resume_from_checkpoint)


def plot_trials_psnr(result_grid, exp_name):
    # best_result: Result = result_grid.get_best_result()
    # best_result.metrics_dataframe.plot("training_iteration", "mean_psnr")
    ax = None
    for result in result_grid:
        label = f"lr={result.config['lr']:.3e}, wd={result.config['wd']:.3e}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "mean_psnr", label=label)
        else:
            result.metrics_dataframe.plot("training_iteration", "mean_psnr", ax=ax, label=label)
    ax.set_title("Mean PSNR vs. Training Iteration for All Trials")
    ax.set_ylabel("Mean Test PSNR")
    ax.figure.savefig("/home/haneol.kijm/ray_results/" + exp_name + "/" + exp_name + ".png")


def tune_pb2(args, num_samples=10, num_epochs=100, gpus_per_trial=0, data_dir="~/data"):
    config = {
        "model_name": args.model,
        "img_size": args.img_size,
        # "in_chans": args.img_channels,
        # "patch_size": tune.choice([8, 16]),
        "patch_size": 16,
        # embed dim, num heads,
        "depth": tune.choice([15, 16, 17, 18]),
        "lr": 1e-3,
        "wd": 0.01,
        # "batch_size": tune.choice([16, 32, 64, 128, 200]),
        "batch_size": 16,
        "loss_fn": tune.choice(["l1", "l2"]),
        "max_epochs": num_epochs,
        # "warmup": tune.choice([0, 3]),
        "warmup": 0,
    }

    scheduler = PB2(
        time_attr="training_iteration",
        perturbation_interval=4,
        hyperparam_bounds={
            "lr": [1e-5, 1e-1],
            "wd": [1e-4, 1e-1],
            # "batch_size": [16, 32],
        },
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "wd", "loss_fn", "depth"],
        metric_columns=["loss", "mean_psnr", "training_iteration"],
    )
    exp_name = f"tune_{args.model}_pb2_" + datetime.now().strftime("%Y%m%d-%H%M")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_tune_checkpoint,
                num_epochs=num_epochs,
                num_gpus=gpus_per_trial,
                data_dir=data_dir,
            ),
            resources={"cpu": 1, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            # metric="loss",
            # mode="min",
            metric="mean_psnr",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name=exp_name,
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    result = tuner.fit()

    print("Best hyperparameters found were: ", result.get_best_result().config)
    plot_trials_psnr(result, exp_name)


def main(args):
    ## Set Seeds
    seed_everything(args.seed)

    ## dataset
    if args.dataset == "imagenet":
        data_dir = "~/Works/data/imagenet-mini_processed/"
    elif args.dataset == "sidd":
        data_dir = "~/Works/data/SIDD_processed/"
    else:
        raise ValueError("Wrong dataset name")

    ## tune
    if args.smoke_test:
        tune_pb2(args, num_samples=1, num_epochs=1, gpus_per_trial=1, data_dir=data_dir)
    elif args.resume:
        trainalbe_with_params_restored = tune.with_parameters(
            train_tune_checkpoint,
            num_epochs=args.num_epochs,
            num_gpus=1,
            data_dir=data_dir,
        )
        tuner = tune.Tuner.restore(
            path=args.resume, overwrite_trainable=trainalbe_with_params_restored
        )
        result = tuner.fit()
        print("Best hyperparameters found were: ", result.get_best_result().config)
    elif args.test:
        print(f"Loading results from {args.test}")
        trainable_with_params_restored = tune.with_parameters(
            train_tune_checkpoint,
            num_epochs=args.num_epochs,
            num_gpus=1,
            data_dir=data_dir,
        )
        restored_tuner = tune.Tuner.restore(
            path=args.test,
            overwrite_trainable=trainable_with_params_restored,
            resume_unfinished=False,
        )
        restored_tuner.fit()
        # result = tuner.fit()
        result_grid = restored_tuner.get_results()

        # Get the result with the maximum test set `mean_accuracy`
        best_result: Result = result_grid.get_best_result()
        print("Best hyperparameters found were: ", best_result.config)
        checkpoint: TorchCheckpoint = best_result.checkpoint
        # model_from_ckpt = checkpoint.get_model(LitModel())
        # print(model_from_ckpt)

        restored_config = best_result.config
        restored_config["model"] = "ConVit"
        # Create a Predictor using the best result's checkpoint
        predictor = TorchPredictor.from_checkpoint(checkpoint, LitModel(restored_config))
        val_dataset = PairDataset(
            data_dir + "clean_val",
            data_dir + "noisy_val",
            split="eval",
            transform=PairCenterCrop(224),
        )
        loader = data.DataLoader(val_dataset, batch_size=1, num_workers=8)
        for batch in loader:
            predictor.predict(batch)
            break

    else:
        # Population based training
        tune_pb2(
            args,
            num_samples=4,
            num_epochs=args.num_epochs,
            gpus_per_trial=1,
            data_dir=data_dir,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    ## program level args
    parser.add_argument("--smoke-test", action="store_true", help="Finish quick try for testing")
    parser.add_argument("--notification_email", type=str, default="haneol.kijm@snu.ac.kr")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Put in checkpoint path if you want to resume from it",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Put in checkpoint path if you want to test it",
    )

    ## test setting args
    parser.add_argument("--dataset", type=str, default="sidd", help="Avaliable: imagenet, sidd")
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1234)

    ## model args
    parser.add_argument("--model", type=str, default="convit")
    parser.add_argument("--img_size", type=int, default=224)
    args, _ = parser.parse_known_args()
    main(args)
