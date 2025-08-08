import torch
import math
from abc import ABC, abstractmethod

from src.utils import Logger, get_gaussion_integration_points_and_weights


class AbstractRhsLayer(ABC):
    @property
    @abstractmethod
    def rank(self) -> int: ...

    def __init__(
        self,
        gaussQR: int,
        BEM_type: int,
        order_type: int,
        wavenumber: float,
        logger: Logger,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._gaussQR = gaussQR
        self._order_type = order_type
        self._BEM_type = BEM_type
        self._wavenumber = wavenumber
        self.logger = logger
        self.device = device
        self.dtype = dtype

        self.rhs_vec = torch.Tensor([[]]).to(self.device).to(self.dtype)

        self.initialized = False

    @abstractmethod
    def compute_rhs(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        face_areas: torch.Tensor,
        panel_relations: torch.Tensor,
    ): ...

    def get_rhs_vec(self) -> torch.Tensor:
        return self.rhs_vec

    @property
    def gaussQR(self) -> int:
        return self._gaussQR

    @property
    def order_type(self) -> int:
        return self._order_type

    @property
    def BEM_type(self) -> int:
        return self._BEM_type

    @property
    def wavenumber(self) -> float:
        return self._wavenumber

    def InfoLog(self, *args, **kwargs):
        return self.logger.InfoLog(*args, **kwargs)

    def WarnLog(self, *args, **kwargs):
        return self.logger.WarnLog(*args, **kwargs)

    def ErrorLog(self, *args, **kwargs):
        return self.logger.ErrorLog(*args, **kwargs)
