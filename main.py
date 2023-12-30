import os
import argparse
import cv2
import builtins
import lightning as L
from lightning.fabric.loggers import TensorBoardLogger

from model import get_pipeline, get_model
from dataset import get_dataloader
from utils import LoggerHelper, TrainHelper, HyperParams as HP

# type hint
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from webdataset import WebLoader


def train(
    fabric: L.Fabric,
    model: Module,
    dataloader: WebLoader,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    train_helper: TrainHelper,
) -> None:
    cfg = HP.instance()
    logger = LoggerHelper.instance()

    pbar = train_helper.process_bar("train", total=cfg.train.num_iteration)
    pbar.update(train_helper.clock.step - 1)

    model.controlnet.train()
    for idx, batch in enumerate(dataloader):
        metrics = {}
        clip_input = batch["clip_input"].to(fabric.device)
        vae_image_input = batch["vae_image_input"].to(fabric.device)
        vae_video_input = batch["vae_video_input"].to(fabric.device)
        added_time_ids = batch["added_time_ids"].to(fabric.device)
        controlnet_cond = batch["controlnet_cond"].to(fabric.device)

        model.set_timesteps(fabric.device)
        loss = model(
            clip_input,
            vae_image_input,
            vae_video_input,
            added_time_ids,
            controlnet_cond,
        )
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        metrics["loss"] = fabric.all_reduce(loss, reduce_op="mean").item()
        metrics["lr"] = lr_scheduler.get_last_lr()[0]

        train_helper.clock.update()
        pbar.update()
        pbar.set_postfix(metrics)

        if cfg.model.use_ema:
            train_helper.ema.update(model.controlnet, cfg.model.ema_momentum)

        if train_helper.clock.step % cfg.monitor.log_interval == 0:
            logger.info(LoggerHelper.dict2str(metrics))

        if fabric.is_global_zero:
            fabric.log_dict(metrics, step=train_helper.clock.step)

        fabric.barrier()
        if train_helper.clock.step % cfg.monitor.latest_interval == 0:
            checkpoint = os.path.join(cfg.fs.model_dir, "latest")
            train_helper.save_state(checkpoint, model, optimizer, lr_scheduler)

        fabric.barrier()
        if train_helper.clock.step % cfg.monitor.save_interval == 0:
            checkpoint = os.path.join(
                cfg.fs.model_dir, f"step_{train_helper.clock.step}"
            )
            train_helper.save_state(checkpoint, model, optimizer, lr_scheduler)

        fabric.barrier()
        if train_helper.clock.step > cfg.train.num_iteration:
            checkpoint = os.path.join(cfg.fs.model_dir, "final")
            train_helper.save_state(checkpoint, model, optimizer, lr_scheduler)
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="hyper-parameters of training stage, should be a yaml file",
    )
    args = parser.parse_args()
    HP.load_config(args.config)
    cfg = HP.instance()

    fabric = L.Fabric(
        strategy="deepspeed_stage_3",
        precision="16-mixed",
        loggers=TensorBoardLogger(
            root_dir=cfg.fs.exp_dir,
            name=cfg.fs.tensorboard_dir,
        ),
    )
    fabric.launch()
    fabric.seed_everything(cfg.train.seed + fabric.local_rank)

    cv2.setNumThreads(0)

    if fabric.is_global_zero:
        os.makedirs(cfg.fs.exp_dir, exist_ok=True)
    fabric.barrier()

    LoggerHelper.init_logger(log_path=cfg.fs.log_path)
    if not fabric.is_global_zero:
        LoggerHelper.disable_in_other_ranks()
        builtins.print = lambda x: None
    logger = LoggerHelper.instance()

    # init pipeline
    pipeline = get_pipeline(cfg.model.model_id)
    # init model
    model = get_model(pipeline, cfg)
    # init train helper
    train_helper = TrainHelper(fabric, model, cfg)
    # init optimizer and scheduler
    optimizer, lr_scheduler = model.configure_optimizers()
    # init dataloader
    dataloader = get_dataloader(cfg)
    # setup model and optimizer
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.train.resume:
        checkpoint = os.path.join(cfg.fs.model_dir, "latest")
        train_helper.load_state(checkpoint, model, optimizer, lr_scheduler)

    train(fabric, model, dataloader, optimizer, lr_scheduler, train_helper)


if __name__ == "__main__":
    main()
