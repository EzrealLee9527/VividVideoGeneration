import os
import argparse
import torch
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

logger = LoggerHelper.get_logger(__name__)


def train(
    fabric: L.Fabric,
    model: Module,
    dataloader: WebLoader,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    train_helper: TrainHelper,
) -> None:
    cfg = HP.instance()
    metric = {}
    pbar = train_helper.process_bar("train", total=cfg.train.num_iteration)

    model.unet.train()
    for idx, batch in enumerate(dataloader):
        # TODO: batch data
        loss = model(batch, idx)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        metric["loss"] = fabric.all_reduce(loss, reduce_op="mean").item()
        metric["lr"] = lr_scheduler.get_last_lr()[0]

        train_helper.clock.update()
        pbar.update()
        pbar.set_postfix(metric)

        if cfg.model.use_ema:
            train_helper.ema.update(model.unet, cfg.model.ema_momentum)

        if train_helper.clock.step % cfg.monitor.log_interval == 0:
            logger.info(train_helper.dict2str(metric))

        fabric.barrier()
        if train_helper.clock.step % cfg.monitor.latest_interval == 0:
            checkpoint = os.path.join(cfg.fs.model_dir, "latest")
            train_helper.save_checkpoint(
                checkpoint, model.unet, optimizer, lr_scheduler
            )

        fabric.barrier()
        if train_helper.clock.step % cfg.monitor.save_interval == 0:
            checkpoint = os.path.join(
                cfg.fs.model_dir, f"step_{train_helper.clock.step}"
            )
            train_helper.save_checkpoint(
                checkpoint, model.unet, optimizer, lr_scheduler
            )

        fabric.barrier()
        if train_helper.clock.step > cfg.num_iteration:
            checkpoint = os.path.join(cfg.fs.model_dir, "final")
            train_helper.save_checkpoint(
                checkpoint, model.unet, optimizer, lr_scheduler
            )
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
        strategy="deepspeed_stage_3_offload",
        precision="16-true",
        loggers=TensorBoardLogger(
            root_dir=cfg.fs.output_dir,
            name=cfg.fs.tensorboard_dir,
        ),
    )
    fabric.launch()

    logger = LoggerHelper.config_logger(
        logger, log_path=cfg.fs.log_path, local_rank=fabric.local_rank
    )

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
    dataloader = dataloader.batched(cfg.train.batch_size // fabric.world_size)
    # setup model and optimizer
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.train.resume:
        checkpoint = os.path.join(cfg.fs.model_dir, "latest")
        train_helper.loal_checkpoint(checkpoint, model.unet, optimizer, lr_scheduler)

    train(fabric, model, dataloader, optimizer, lr_scheduler, train_helper)


if __name__ == "__main__":
    main()
