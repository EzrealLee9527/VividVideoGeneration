import os
import torch
import logging
import megfile
from copy import deepcopy
from omegaconf import OmegaConf
from deepspeed.utils import safe_get_full_fp32_param, safe_set_full_fp32_param
from tqdm.auto import tqdm
from time import time

# type hint
from typing import Dict
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig
from lightning import Fabric


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def instance(cls):
        if cls not in cls._instances:
            raise RuntimeError(f"Please initialize '{cls.__name__}' first")
        return cls._instances[cls]


class HyperParams(metaclass=Singleton):
    @classmethod
    def load_config(cls, config: str) -> None:
        cfg = OmegaConf.load(config)
        cfg.fs.log_path = os.path.join(cfg.fs.output_dir, cfg.fs.log_path)
        cfg.fs.model_dir = os.path.join(cfg.fs.output_dir, cfg.fs.model_dir)
        cls._instances[cls] = cfg


class LoggerHelper:
    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        return logger

    @staticmethod
    def config_logger(logger, log_path, local_rank):
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        stream_handler = logging.StreamHandler()
        if local_rank == 0:
            stream_handler.setLevel(logging.INFO)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(filename=log_path, mode="a")
        if local_rank == 0:
            file_handler.setLevel(logging.INFO)
        else:
            file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def dict2str(dict_obj: Dict[str, float]) -> str:
        kv_list = []
        for k, v in dict_obj.items():
            kv_list.append(f"{k}: {v:.7g}")
        return " ".join(kv_list)


class EMAModule:
    def __init__(self, module: Module) -> None:
        self._state_dict: Dict[str, torch.Tensor] = {}

        ema_module = deepcopy(module)
        ema_module.to("cpu")
        for name, param in ema_module.named_parameters():
            self._state_dict[name] = param

    @torch.no_grad()
    def update(self, module: Module, momentum: float = 0.999) -> None:
        for name, param in module.named_parameters():
            param = safe_get_full_fp32_param(param)
            self._state_dict[name].mul_(momentum).add_(param * (1 - momentum))
            safe_set_full_fp32_param(param, self._state_dict[name])

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        for name in self.state_dict:
            self.state_dict[name] = state_dict[name]


class Clock:
    def __init__(self) -> None:
        self._step = 1
        self._tic = 0

    def update(self, n_step=1) -> None:
        self._step += n_step

    def reset(self, step=1) -> None:
        self._step = step

    def tick(self):
        self._tic = time()

    def tock(self):
        used_time = time() - self._tic
        self._tic = time()
        return used_time

    @property
    def step(self) -> int:
        return self._step

    def state_dict(self) -> Dict[str, int]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: Dict[str, int]):
        self._step = state_dict["step"]


class TrainHelper:
    def __init__(self, fabric: Fabric, module: Module, config: DictConfig) -> None:
        self.fabric = fabric
        self.use_ema = config.model.use_ema
        if self.use_ema:
            self.ema = EMAModule(module)
        self.clock = Clock()

    def save_checkpoint(
        self,
        checkpoint: str,
        module: Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
    ) -> None:
        logging.info(f"save checkpoint to {checkpoint}")
        local_filename = os.path.basename(checkpoint)

        state = {"module": module, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
        self.fabric.save(local_filename, state)

        self.fabric.barrier()
        if self.fabric.is_global_zero:
            torch.save(self.clock.state_dict(), os.path.join(local_filename, "clock"))
            if self.use_ema:
                torch.save(self.ema.state_dict(), os.path.join(local_filename, "ema"))
            megfile.smart_sync(local_filename, checkpoint)
            megfile.smart_remove(local_filename)

    def loal_checkpoint(
        self,
        checkpoint: str,
        module: Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
    ) -> None:
        logging.info(f"load checkpoint from {checkpoint}")
        local_filename = os.path.basename(checkpoint)

        if not megfile.smart_exists(checkpoint):
            logging.info(f"checkpoint {checkpoint} not exist")
            return

        if self.fabric.is_global_zero:
            megfile.smart_sync(checkpoint, local_filename)

        self.fabric.barrier()
        state = {"module": module, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
        remainder = self.fabric.load(local_filename, state)

        self.clock.load_state_dict(torch.load(os.path.join(local_filename, "clock")))
        if self.use_ema:
            self.ema.load_state_dict(torch.load(os.path.join(local_filename, "ema")))

        self.fabric.barrier()
        if self.fabric.is_global_zero:
            megfile.smart_remove(local_filename)

    def process_bar(self, desc: str, total: int = None) -> tqdm:
        return tqdm(desc=desc, total=total, disable=not self.fabric.is_global_zero)


class profiling:
    def __init__(self, flag_msg, n_digits=7) -> None:
        self.flag_msg = flag_msg
        self.n_digits = n_digits
        self.tic = 0

    def __enter__(self):
        print(f"'{self.flag_msg}' started...")
        self.tic = time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        used = time() - self.tic
        print(f"'{self.flag_msg}' finished! used {used:.{self.n_digits}f}s")


def profiling_it(flag_msg=None, flag_appendix=None, n_digits=7):
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            msg = ""
            if hasattr(func, "__self__"):
                msg += f"{func.__self__.__class__.__name__}."
            msg += func.__name__
            if flag_msg:
                msg = f"'{flag_msg}'"
            if flag_appendix:
                msg += f"({flag_appendix})"

            with profiling(f"function {msg}", n_digits=n_digits):
                res = func(*args, **kwargs)
            return res

        return inner_wrapper

    return wrapper
