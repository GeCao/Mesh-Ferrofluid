import os
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt

# import drjit as dr
# import mitsuba as mi
from typing import Dict, Any, List, Union
from torch.utils.cpp_extension import load


mesh_MOM_3d = load(
    name="mesh_MOM_3d",
    sources=[
        "../src/mesh_BEM/cuda/mesh_MOM_3d.cpp",
        "../src/mesh_BEM/cuda/mesh_MOM_3d.cu",
    ],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
)


class MeshMOMZmatKernel(torch.autograd.Function):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        self._device = device
        self._dtype = dtype
        super(torch.autograd.Function, self).__init__(*args, **kwargs)

    @staticmethod
    def forward(
        ctx,
        verts: torch.Tensor,
        faces: torch.Tensor,
        face_areas: torch.Tensor,
        face_normals: torch.Tensor,
        gaussian_points_1d_x: torch.Tensor,
        gaussian_weights_1d_x: torch.Tensor,
        gaussian_points_1d_y: torch.Tensor,
        gaussian_weights_1d_y: torch.Tensor,
        wavenumber: float,
    ) -> Any:
        N_faces, dim = faces.shape
        face_edge_Zmat_real = torch.zeros(
            (1, N_faces * dim, N_faces * dim), dtype=verts.dtype, device=verts.device
        )
        face_edge_Zmat_imag = torch.zeros(
            (1, N_faces * dim, N_faces * dim), dtype=verts.dtype, device=verts.device
        )
        mesh_MOM_3d.create_MOM_Zmat_3d_forward(
            verts,
            faces,
            face_areas,
            face_normals,
            gaussian_points_1d_x,
            gaussian_weights_1d_x,
            gaussian_points_1d_y,
            gaussian_weights_1d_y,
            face_edge_Zmat_real,
            face_edge_Zmat_imag,
            wavenumber,
        )

        face_edge_Zmat = face_edge_Zmat_real + 1j * face_edge_Zmat_imag

        return face_edge_Zmat


class MeshMOMRhsKernel(torch.autograd.Function):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        self._device = device
        self._dtype = dtype
        super(torch.autograd.Function, self).__init__(*args, **kwargs)

    @staticmethod
    def forward(
        ctx,
        verts: torch.Tensor,
        faces: torch.Tensor,
        face_areas: torch.Tensor,
        face_normals: torch.Tensor,
        gaussian_points_1d: torch.Tensor,
        gaussian_weights_1d: torch.Tensor,
        wavenumber: float,
    ) -> Any:
        N_faces, dim = faces.shape
        face_edge_rhs_real = torch.zeros(
            (1, N_faces, dim, 1), dtype=verts.dtype, device=verts.device
        )
        face_edge_rhs_imag = torch.zeros(
            (1, N_faces, dim, 1), dtype=verts.dtype, device=verts.device
        )
        mesh_MOM_3d.create_MOM_rhs_3d_forward(
            verts,
            faces,
            face_areas,
            face_normals,
            gaussian_points_1d,
            gaussian_weights_1d,
            face_edge_rhs_real,
            face_edge_rhs_imag,
            wavenumber,
        )

        face_edge_rhs = face_edge_rhs_real + 1j * face_edge_rhs_imag

        return face_edge_rhs


class MeshMOMInterpolateJsKernel(torch.autograd.Function):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        self._device = device
        self._dtype = dtype
        super(torch.autograd.Function, self).__init__(*args, **kwargs)

    @staticmethod
    def forward(
        ctx,
        verts: torch.Tensor,
        faces: torch.Tensor,
        face_edge_indices: torch.Tensor,
        face_areas: torch.Tensor,
        edge_I_coeff: torch.Tensor,
    ) -> Any:
        N_faces, dim = faces.shape
        batch_size = edge_I_coeff.shape[0]
        face_vert_Js_real = torch.zeros(
            (batch_size, N_faces * dim, dim), dtype=verts.dtype, device=verts.device
        )
        face_vert_Js_imag = torch.zeros(
            (batch_size, N_faces * dim, dim), dtype=verts.dtype, device=verts.device
        )

        edge_I_coeff_real = torch.zeros_like(edge_I_coeff).to(verts.dtype)
        edge_I_coeff_imag = torch.zeros_like(edge_I_coeff).to(verts.dtype)
        edge_I_coeff_real = edge_I_coeff_real + edge_I_coeff.real
        edge_I_coeff_imag = edge_I_coeff_imag + edge_I_coeff.imag
        mesh_MOM_3d.interpolate_Icoeff_to_Js_3d_forward(
            verts,
            faces,
            face_edge_indices,
            face_areas,
            edge_I_coeff_real,
            edge_I_coeff_imag,
            face_vert_Js_real,
            face_vert_Js_imag,
        )

        face_vert_Js = face_vert_Js_real + 1j * face_vert_Js_imag

        return face_vert_Js
