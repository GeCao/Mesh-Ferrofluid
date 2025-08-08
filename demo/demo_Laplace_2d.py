"""Copyright (C) 2025, Ge Cao
ACEM research group, https://acem.ece.illinois.edu/
All rights reserved.

This software is free for non-commercial, research and evaluation use
under the terms of the LICENSE.md file.

For inquiries contact  gecao2@illinois.edu
"""

import sys
import os
import pathlib
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List

sys.path.append("../")

from src.mesh_BEM import MeshBEMSolver2d
from src.utils import (
    mkdir,
    Logger,
    BEMType,
    BoundaryType,
    LoadSingleMesh,
    compute_vert_normals,
    render_mesh_to_file,
    remesh_and_transfer_velocity,
)


def f(pos: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """A self-defined function for f(x)
    f(x) = x^2 - y^2

    Args:
        pos (torch.Tensor): [..., dim=2/3]
        keepdim (bool, optional): keepdim (for 2D only). Defaults to False.

    Returns:
        torch.Tensor: [..., (1)]
    """
    fx = pos[..., 0] * pos[..., 0] - pos[..., 1] * pos[..., 1]

    if keepdim:
        fx = fx.unsqueeze(-1)

    return fx


def dfdn(
    pos: torch.Tensor, vert_normals: torch.Tensor, keepdim: bool = False
) -> torch.Tensor:
    """A self-defined function for f(x)
    f(x) = x^2 - y^2

    Args:
        pos (torch.Tensor): [..., dim=2/3]
        vert_normals (torch.Tensor): [..., dim=2/3]
        keepdim (bool, optional): keepdim (for 2D only). Defaults to False.

    Returns:
        torch.Tensor: [..., (1)]
    """
    grad_fx = torch.stack((2 * pos[..., 0], -2 * pos[..., 1]), dim=-1)

    dfdn = (grad_fx * vert_normals).sum(dim=-1, keepdim=keepdim)

    return dfdn


def main(
    geom: str = "Analytical_Sphere",
    res: List[int] = [130, 130, 130],
    gaussQR: int = 8,
    order_type: int = 1,
):
    dim = 2

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # set up the path for saving
    path = pathlib.Path(__file__).parent.absolute()
    save_dir = f"{path}/../save/"
    mkdir(save_dir)
    save_dir = f"{save_dir}/data_Laplace_{dim}d/"
    mkdir(save_dir)

    logger = Logger(root_path="", log_to_disk=False)

    # Create a 2D circle
    N_verts = 128
    verts = torch.linspace(0, N_verts - 1, N_verts, device=device, dtype=dtype)
    verts = (2 * math.pi) * verts / N_verts  # [0, 2PI]
    verts = torch.stack((torch.cos(verts), torch.sin(verts)), dim=-1)
    faces = (
        torch.Tensor([[i, (i + 1) % N_verts] for i in range(N_verts)])
        .to(device)
        .to(torch.int32)
    )
    vert_normals = compute_vert_normals(vertices=verts, faces=faces)

    L = 0.01
    logger.InfoLog(f"The Zone of this liquid: L = {L}")
    verts = verts * (L / 2)

    BEM_solver = MeshBEMSolver2d(
        gaussQR=gaussQR,
        order_type=order_type,
        BEM_type=int(BEMType.LAPLACE),
        wavenumber=1.0,
        logger=logger,
        dtype=dtype,
        device=device,
    )

    vert_f = f(pos=verts.unsqueeze(0), keepdim=True)
    vert_dfdn_GT = dfdn(pos=verts.unsqueeze(0), vert_normals=vert_normals, keepdim=True)
    vert_bc_flags = torch.zeros_like(vert_f).to(torch.uint8)
    vert_bc_flags[...] = int(BoundaryType.DIRICHLET)
    render_mesh_to_file(
        img_dir=save_dir,
        file_label="dfdn(GT)",
        verts=verts / (L / 2),
        faces=faces,
        vert_phi=vert_dfdn_GT,
    )

    vert_dfdn_ours, _ = BEM_solver.solve(
        verts=verts,
        faces=faces,
        vert_bc_Dirichlet=vert_f,
        vert_bc_Neumann=torch.zeros_like(vert_dfdn_GT),
        vert_bc_flags=vert_bc_flags,
    )
    render_mesh_to_file(
        img_dir=save_dir,
        file_label="dfdn(Ours)",
        verts=verts / (L / 2),
        faces=faces,
        vert_phi=vert_dfdn_ours,
    )

    logger.InfoLog(
        f"[Sovled Neumamm]: Ours = {vert_dfdn_ours.mean()}, {vert_dfdn_ours.min()}, {vert_dfdn_ours.max()}"
    )
    logger.InfoLog(
        f"[Sovled Neumamm]: GT   = {vert_dfdn_GT.mean()}, {vert_dfdn_GT.min()}, {vert_dfdn_GT.max()}"
    )


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )

    parser.add_argument(
        "--geom",
        type=str,
        default="Analytical_Plane",
        choices=["Analytical_Sphere", "Analytical_Plane"],
        help="The geometry",
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[32, 32, 32],
        help="Simulation size of the current simulation currently only square",
    )

    parser.add_argument(
        "--gaussQR", type=int, default=4, help="Gaussian integration order"
    )
    parser.add_argument(
        "--order_type", type=int, default=1, choices=[0, 1], help="Planar or Linear"
    )

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
