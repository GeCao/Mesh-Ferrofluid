import torch
import math
from abc import ABC, abstractmethod

from src.utils import (
    Logger,
    pack_matrix_tight,
    get_gaussion_integration_points_and_weights,
)


class AbstractDoubleLayer(ABC):
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

        self.gaussian_points_1d, self.gaussian_weights_1d = (
            get_gaussion_integration_points_and_weights(
                gaussQR, device=device, dtype=dtype
            )
        )

        # Generate number(r1, r2)
        if self.rank == 2:
            r1 = self.gaussian_points_1d.reshape(gaussQR, 1)

            weights = self.gaussian_weights_1d.reshape((gaussQR, 1))

            self.r1 = r1
            self.gauss_weights = weights
        elif self.rank == 3:
            gaussQR2 = gaussQR * gaussQR

            r1 = self.gaussian_points_1d[None, :]
            r2 = self.gaussian_points_1d[:, None] * r1
            r1 = r1.repeat(gaussQR, 1)
            r1 = r1.reshape((gaussQR2, 1))
            r2 = r2.reshape((gaussQR2, 1))

            weights = self.gaussian_weights_1d * self.gaussian_weights_1d[:, None]
            weights = weights.reshape((gaussQR2, 1))  # [gaussQR2, 1]

            self.r1 = r1
            self.r2 = r2
            self.gauss_weights = weights

        self.unpacked_K_mat = torch.Tensor([[[]]]).to(self.device).to(self.dtype)

        self.initialized = False

    @abstractmethod
    def compute_Kmat(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        face_areas: torch.Tensor,
        panel_relations: torch.Tensor,
    ): ...

    def get_Kmat_unpacked(self) -> torch.Tensor:
        return self.unpacked_K_mat  # [N_faces, N_phiy, N_faces, N_phix]

    def get_Kmat_tight(self, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Get The K_mat in tight format
        Our K_mat is saved with the format of: [N_faces, N_phiy, N_faces, N_phix]
        However, to use it in our BEM solver, it has to be in format of [N_verts, N_verts]

        Args:
            verts (torch.Tensor): [N_verts, dim]
            faces (torch.Tensor): [N_faces, dim]

        Returns:
            torch.Tensor: [N_verts, N_verts]
        """
        return pack_matrix_tight(
            order_type=self.order_type,
            unpacked_matrix=self.unpacked_K_mat,
            verts=verts,
            faces=faces,
        )

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
