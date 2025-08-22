import torch
import math
from abc import ABC, abstractmethod

from src.utils import Logger


class AbstractWOBSolver(ABC):
    @property
    @abstractmethod
    def rank(self) -> int: ...

    def __init__(
        self,
        wavenumber: float,
        logger: Logger,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._wavenumber = wavenumber
        self.logger = logger
        self.device = device
        self.dtype = dtype

        self.single_layer = None
        self.double_layer = None
        self.hypersingular_layer = None
        self.rhs_layer = None

    @property
    def wavenumber(self) -> float:
        return self._wavenumber

    def InfoLog(self, *args, **kwargs):
        return self.logger.InfoLog(*args, **kwargs)

    def WarnLog(self, *args, **kwargs):
        return self.logger.WarnLog(*args, **kwargs)

    def ErrorLog(self, *args, **kwargs):
        return self.logger.ErrorLog(*args, **kwargs)
