import os
import argparse
import cv2
import builtins
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


def load_pretrain(fabric, model):
    cfg = HP.instance()
    if not hasattr(cfg.model, "pretrain_weight"):
        return
    state_dict_file = "state_dict.pt"
    if fabric.is_global_zero:
        logger = LoggerHelper.instance()
        import megfile
        from importlib.util import spec_from_file_location, module_from_spec

        local_dir = os.path.basename(cfg.model.pretrain_weight)
        script = "zero_to_fp32.py"

        if megfile.smart_exists(cfg.model.pretrain_weight):
            logger.info(f"load pretrain weight from {cfg.model.pretrain_weight}")
            megfile.smart_sync(cfg.model.pretrain_weight, local_dir)
            spec = spec_from_file_location("zero", os.path.join(local_dir, script))
            m = module_from_spec(spec)
            spec.loader.exec_module(m)
            func = getattr(m, "convert_zero_checkpoint_to_fp32_state_dict")
            func(local_dir, state_dict_file)
    fabric.barrier()

    if os.path.exists(state_dict_file):
        controlnet_weight = {}
        state_dict = torch.load(state_dict_file, map_location="cpu")
        for name in state_dict:
            if "controlnet" in name:
                controlnet_weight[name.replace("controlnet.", "")] = state_dict[name]
        model.controlnet.load_state_dict(controlnet_weight)

        fabric.barrier()
        if fabric.is_global_zero:
            megfile.smart_remove(local_dir)
            megfile.smart_remove(state_dict_file)


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
    pbar.update(train_helper.clock.step)

    model.controlnet.train()
    for idx, batch in enumerate(dataloader):
        metrics = {}
        clip_input = batch["clip_input"].to(fabric.device)
        vae_image_input = batch["vae_image_input"].to(fabric.device)
        vae_video_input = batch["vae_video_input"].to(fabric.device)
        added_time_ids = batch["added_time_ids"].to(fabric.device)
        controlnet_cond = batch["controlnet_cond"].to(fabric.device)

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
    HP.load(args.config)
    cfg = HP.instance()

    fabric = L.Fabric(
        strategy="deepspeed_stage_3",
        precision="16-mixed",
        loggers=TensorBoardLogger(
            root_dir=cfg.fs.exp_dir, name=cfg.fs.tensorboard_dir, version="train_event"
        ),
    )
    fabric.seed_everything(cfg.train.seed + fabric.global_rank)

    cv2.setNumThreads(0)

    if fabric.is_global_zero:
        os.makedirs(cfg.fs.exp_dir, exist_ok=True)
        HP.save(os.path.join(cfg.fs.exp_dir, "hyperparams.yaml"))
        fabric.logger.log_hyperparams(cfg)

    fabric.barrier()
    LoggerHelper.init_logger(log_path=cfg.fs.log_path)
    if not fabric.is_global_zero:
        LoggerHelper.disable_in_other_ranks()
        builtins.print = lambda *args: None

    # init pipeline
    pipeline = get_pipeline(cfg.model.model_id)
    # init model
    model = get_model(pipeline, cfg)
    # load pretrain weight
    load_pretrain(fabric, model)
    # init train helper
    train_helper = TrainHelper(fabric, model, cfg)
    # init optimizer and scheduler
    optimizer, lr_scheduler = model.configure_optimizers()
    # init dataloader
    dataloader = get_dataloader(cfg)
    # setup model and optimizer
    model, optimizer = fabric.setup(model, optimizer)
    model.set_timesteps(fabric.device, fabric._precision._desired_dtype)

    if cfg.train.resume:
        checkpoint = os.path.join(cfg.fs.model_dir, "latest")
        train_helper.load_state(checkpoint, model, optimizer, lr_scheduler)

    train(fabric, model, dataloader, optimizer, lr_scheduler, train_helper)


if __name__ == "__main__":
    main()
