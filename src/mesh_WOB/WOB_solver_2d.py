import torch
from typing import List

from src.mesh_WOB import AbstractWOBSolver
from src.utils import (
    Logger,
    compute_face_areas,
    compute_face_normals,
    compute_vert_areas,
    get_vertices_from_index,
    BoundaryType,
    VertexType,
)


class MeshWOBSolver2d(AbstractWOBSolver):
    rank = 2

    def __init__(
        self,
        wavenumber: float,
        logger: Logger,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(MeshWOBSolver2d, self).__init__(
            wavenumber=wavenumber, logger=logger, dtype=dtype, device=device
        )

    def solve_WOB(self, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Mesh-based WOB solver: To solve Helmholtz problem
        Reference: Ryusuke Sugimoto, Terry Chen, Yiti Jiang, Christopher Batty, Toshiya Hachisuka, A Practical Walk-on-Boundary Method for Boundary Value Problems, 2023.
        https://arxiv.org/abs/2305.04403

        Args:
            verts (torch.Tensor): [..., N_verts, dim]
            faces (torch.Tensor): [..., N_faces, dim]

        Returns:
            torch.Tensor: [B, N_Verts, dim=2], The solved Js
        """
        N_verts, dim = verts.shape[-2:]
        N_faces, dim = faces.shape[-2:]
        batch_size = 1

        device = verts.device
        dtype = verts.dtype

        face_areas = compute_face_areas(vertices=verts, faces=faces, keepdim=True)
        face_normals = compute_face_normals(vertices=verts, faces=faces)
        vert_areas = compute_vert_areas(vertices=verts, faces=faces)
        face_verts = get_vertices_from_index(verts, index=faces.flatten())
        face_verts = face_verts.reshape(N_faces, dim, dim)

        vert_Js = torch.zeros((batch_size, N_verts, dim), dtype=dtype, device=device)

        return vert_Js
