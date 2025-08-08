import torch


import math
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from typing import List, Tuple, Dict, Callable, Any
from torch.utils.cpp_extension import load


from src.mesh_BEM import AbstractBEMSolver
from src.mesh_BEM.rhs_layer import AbstractRhsLayer
from src.utils import (
    Logger,
    BoundaryType,
    OrderType,
    BEMType,
    compute_vert_areas,
    get_gaussion_integration_points_and_weights,
)


class RhsLayer2d(AbstractRhsLayer):
    rank = 2

    def __init__(
        self,
        parent: AbstractBEMSolver,
        gaussQR: int,
        order_type: int,
        BEM_type: int,
        wavenumber: float,
        logger: Logger,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.parent = parent
        super(RhsLayer2d, self).__init__(
            gaussQR=gaussQR,
            order_type=order_type,
            BEM_type=BEM_type,
            wavenumber=wavenumber,
            logger=logger,
            dtype=dtype,
            device=device,
        )

        self.gaussian_points_1d, self.gaussian_weights_1d = (
            get_gaussion_integration_points_and_weights(
                gaussQR, device=device, dtype=dtype
            )
        )

        # Generate number(r1, r2)
        r1 = self.gaussian_points_1d.reshape(gaussQR, 1)

        weights = self.gaussian_weights_1d.reshape((gaussQR, 1))  # [gaussQR, 1]

        self.r1 = r1
        self.gauss_weights = weights

        self.initialized = False

    def compute_rhs(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        vert_bc_Dirichlet: torch.Tensor,
        vert_bc_Neumann: torch.Tensor,
        vert_bc_flags: torch.Tensor,
    ):
        """Compute Right-Hand-Side of the BEM Solver
        Note: For all unknown D/N points, we set them (g/f) as 0

        Args:
            verts (torch.Tensor): [N_verts, dim]
            faces (torch.Tensor): [N_faces, dim]
            vert_bc_Dirichlet (torch.Tensor): [B, N_verts, 1]
            vert_bc_Neumann (torch.Tensor): [B, N_verts, 1]
            vert_bc_flags (torch.Tensor): [B, N_verts, 1]
        """
        N_faces, dim = faces.shape
        batch_size, N_verts, _ = vert_bc_Dirichlet.shape
        device = verts.device
        dtype = verts.dtype
        vert_areas = compute_vert_areas(vertices=verts, faces=faces, keepdim=True)

        order_type = self.order_type

        rhs_vec_list = []
        for batch_idx in range(batch_size):
            vert_is_Dirichlet = vert_bc_flags[batch_idx] == int(BoundaryType.DIRICHLET)
            vert_is_Neumann = vert_bc_flags[batch_idx] == int(BoundaryType.NEUMANN)

            N_GammaD = vert_is_Dirichlet.sum()
            N_GammaN = vert_is_Neumann.sum()

            face_vert_bc_Dirichlet = torch.index_select(
                vert_bc_Dirichlet[batch_idx],
                dim=-2,
                index=faces.flatten().to(torch.int64),
            )
            face_vert_bc_Dirichlet = face_vert_bc_Dirichlet.reshape(N_faces, 1, dim, 1)

            face_vert_bc_Neumann = torch.index_select(
                vert_bc_Neumann[batch_idx],
                dim=-2,
                index=faces.flatten().to(torch.int64),
            )
            face_vert_bc_Neumann = face_vert_bc_Neumann.reshape(N_faces, 1, dim, 1)

            gk = 0
            fi = 0
            # 1. The Dirichlet part.
            if N_GammaD > 0:
                # There exists the Dirichlet B.C.
                gk1 = 0.5 * vert_bc_Dirichlet[batch_idx] * vert_areas

                unpacked_K_mat = self.parent.double_layer.get_Kmat_unpacked()
                if order_type == int(OrderType.PLANAR):
                    # K_mat in [N_facesy, N_facesx, 1, 1]
                    unpacked_K_mat = unpacked_K_mat.repeat(1, 1, dim, dim) / (dim * dim)
                # elif order_type == int(OrderType.LINEAR):
                #     # K_mat in [N_facesy, N_facesx, dimy, dimx]

                gk2_src = (
                    (unpacked_K_mat * face_vert_bc_Dirichlet).sum(dim=0).sum(dim=-2)
                )  # [N_facesx, dimx], sum on sy
                gk2_src = gk2_src.flatten()
                gk2 = torch.zeros((N_verts,), device=device, dtype=gk2_src.dtype)
                gk2.scatter_add_(
                    dim=-1, index=faces.flatten().to(torch.int64), src=gk2_src
                )
                gk2 = gk2.reshape(N_verts, 1)

                gk3 = 0.0
                if N_GammaN > 0:
                    # Mix problem, calculate gk3
                    unpacked_V_mat = self.parent.single_layer.get_Vmat_unpacked()
                    gk3_src = (
                        (-unpacked_V_mat * face_vert_bc_Neumann).sum(dim=0).sum(dim=-2)
                    )  # [N_facesx, dimx], sum on sy
                    gk3_src = gk3_src.flatten()
                    gk3 = torch.zeros((N_verts,), device=device, dtype=gk3_src.dtype)
                    gk3.scatter_add_(
                        dim=-1, index=faces.flatten().to(torch.int64), src=gk3_src
                    )
                    gk3 = gk3.reshape(N_verts, 1)

                gk = gk1 + gk2 + gk3
                if self.BEM_type == int(BEMType.HELMHOLTZ_MOM):
                    gk = gk1 * 2

            # 2. The Neumann part.
            if N_GammaN > 0:
                # There exists the Dirichlet B.C.
                fi1 = 0.5 * vert_bc_Neumann * vert_areas

                fi2_src = (
                    (-unpacked_K_mat * face_vert_bc_Neumann).sum(dim=0).sum(dim=-2)
                )  # [N_facesx, dimx], sum on sy
                fi2_src = fi2_src.flatten()
                fi2 = torch.zeros((N_verts,), device=device, dtype=fi2_src.dtype)
                fi2.scatter_add_(
                    dim=-1, index=faces.flatten().to(torch.int64), src=fi2_src
                )
                fi2 = fi2.reshape(N_verts, 1)

                fi3 = 0.0
                if N_GammaD > 0:
                    # Mix problem, calculate gk3
                    unpacked_D_mat = self.parent.hypersingular_layer.get_Dmat_unpacked()
                    fi3_src = (
                        (-unpacked_D_mat * face_vert_bc_Dirichlet)
                        .sum(dim=0)
                        .sum(dim=-2)
                    )  # [N_facesx, dimx], sum on sy
                    fi3_src = fi3_src.flatten()
                    fi3 = torch.zeros((N_verts,), device=device, dtype=fi3_src.dtype)
                    fi3.scatter_add_(
                        dim=-1, index=faces.flatten().to(torch.int64), src=fi3_src
                    )
                    fi3 = fi3.reshape(N_verts, 1)

                fi = fi1 + fi2 + fi3
                if self.BEM_type == int(BEMType.HELMHOLTZ_MOM):
                    fi = fi1 * 2

            rhs_vec_list.append(gk + fi)

        self.rhs_vec = torch.stack(rhs_vec_list, dim=0)  # [B, N_verts, 1]

        self.initialized = True
