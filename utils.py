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


LOGGING_FORMAT = (
    "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s"
)


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
    def load(cls, config: str) -> None:
        cfg = OmegaConf.load(config)
        cfg.fs.exp_name = os.path.splitext(os.path.basename(config))[0]
        cfg.fs.exp_dir = os.path.join(cfg.fs.output_dir, cfg.fs.exp_name)
        cfg.fs.log_path = os.path.join(cfg.fs.exp_dir, cfg.fs.log_path)
        cfg.fs.model_dir = os.path.join(
            cfg.fs.bucket_name, cfg.fs.user, cfg.fs.proj_name, cfg.fs.exp_name
        )
        cls._instances[cls] = cfg

    @classmethod
    def save(cls, path: str) -> None:
        OmegaConf.save(cls._instances[cls], path)

    @classmethod
    def pretty_format(cls) -> str:
        return OmegaConf.to_yaml(cls._instances[cls])


class LoggerHelper(metaclass=Singleton):
    @classmethod
    def init_logger(cls, log_path):
        logger = logging.getLogger("train_logger")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(LOGGING_FORMAT)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(filename=log_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        cls._instances[cls] = logger

    @classmethod
    def disable_in_other_ranks(cls):
        logger = cls.instance()
        logger.setLevel(logging.ERROR)

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
        self._step = 0
        self._tic = 0

    def update(self, n_step=1) -> None:
        self._step += n_step

    def reset(self, step=0) -> None:
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

    def save_state(
        self,
        checkpoint: str,
        module: Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
    ) -> None:
        logger = LoggerHelper.instance()
        logger.info(f"save checkpoint to {checkpoint}")
        local_filename = os.path.basename(checkpoint)
        state = {
            "model": module,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
        self.fabric.save(local_filename, state)

        if self.fabric.is_global_zero:
            torch.save(self.clock.state_dict(), os.path.join(local_filename, "clock"))
            if self.use_ema:
                torch.save(self.ema.state_dict(), os.path.join(local_filename, "ema"))
            megfile.smart_sync(local_filename, checkpoint)
            megfile.smart_remove(local_filename)

        del state

    def load_state(
        self,
        checkpoint: str,
        module: Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
    ) -> None:
        logger = LoggerHelper.instance()
        if not megfile.smart_exists(checkpoint):
            logger.info(f"{checkpoint} not exist")
            return

        logger.info(f"load checkpoint from {checkpoint}")
        local_filename = os.path.basename(checkpoint)
        if self.fabric.is_global_zero:
            megfile.smart_sync(checkpoint, local_filename)

        self.fabric.barrier()
        state = {
            "model": module,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
        remainder = self.fabric.load(local_filename, state)

        self.clock.load_state_dict(torch.load(os.path.join(local_filename, "clock")))
        if self.use_ema:
            self.ema.load_state_dict(torch.load(os.path.join(local_filename, "ema")))
        self.fabric.barrier()

        if self.fabric.is_global_zero:
            megfile.smart_remove(local_filename)

        del state

    def save_model_only(self, module, checkpoint):
        logger = LoggerHelper.instance()
        logger.info(f"save model to {checkpoint}")

        state_dict = {}
        for name, param in module.named_parameters():
            param = safe_get_full_fp32_param(param)
            state_dict[name] = param

        if self.fabric.is_global_zero:
            with megfile.smart_open(checkpoint, "wb") as f:
                torch.save(state_dict, f)
        del state_dict

    def load_model_only(self, module, checkpoint):
        logger = LoggerHelper.instance()
        logger.info(f"load model from {checkpoint}")

        with megfile.smart_open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        self.fabric.barrier()
        for name, param in module.named_parameters():
            safe_set_full_fp32_param(param, state_dict[name])
        del state_dict

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
