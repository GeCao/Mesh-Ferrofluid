import torch


import math
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from typing import List, Tuple, Dict, Callable, Any
from torch.utils.cpp_extension import load


from src.mesh_BEM.double_layer import AbstractDoubleLayer
from src.mesh_BEM.utils import double_integration_on_panels_3d
from src.utils import Logger, LayerType


class DoubleLayer3d(AbstractDoubleLayer):
    rank = 3
    layer_type = int(LayerType.DOUBLE_LAYER)

    def __init__(
        self,
        gaussQR: int,
        order_type: int,
        BEM_type: int,
        wavenumber: float,
        logger: Logger,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(DoubleLayer3d, self).__init__(
            gaussQR=gaussQR,
            order_type=order_type,
            BEM_type=BEM_type,
            wavenumber=wavenumber,
            logger=logger,
            dtype=dtype,
            device=device,
        )

        self.initialized = False

    def compute_Kmat(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        face_areas: torch.Tensor,
        face_normals: torch.Tensor,
        panel_relations: torch.Tensor,
    ):
        rands = torch.cat((self.r1, self.r2), dim=-1)
        self.unpacked_K_mat = double_integration_on_panels_3d(
            BEM_type=self.BEM_type,
            order_type=self.order_type,
            verts=verts,
            faces=faces,
            face_areas=face_areas,
            face_normals=face_normals,
            gauss_weightsx=self.gauss_weights,
            gauss_weightsy=self.gauss_weights,
            rands_x=rands,
            rands_y=rands,
            panel_relations=panel_relations,
            wavenumber=self.wavenumber,
            layer_type=self.layer_type,
        )

        self.initialized = True
