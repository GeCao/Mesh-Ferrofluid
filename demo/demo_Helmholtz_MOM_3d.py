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
import argparse
import torch
import torch.nn.functional as F
import trimesh
import matplotlib.pyplot as plt
from typing import List

sys.path.append("../")
from src.simulation import EMSimulationParameters, SimulationRunner
from src.utils import (
    mkdir,
    render_mesh_to_file,
    c0,
    Logger,
    BEMType,
    BoundaryType,
    save_heatmap,
    create_3d_meshgrid_tensor,
    get_edge_unique,
    compute_vert_areas,
    LoadSingleMesh,
)

import soft_renderer as sr


def Einc_func(
    freq: float, eps_r: float, mu_r: float, pos: torch.Tensor, return_k: bool = False
) -> torch.Tensor:
    """A self-defined function for calculate incident electrical field E^{inc}
    azimuth angle = 30,
    Pitch angle = 90.

    k = 2 * PI * freq * sqrt(eps_r * mu_r) / c

    Args:
        freq (float):
        eps_r (float):
        mu_r (float):
        pos (torch.Tensor): [..., dim=2/3]

    Returns:
        torch.Tensor: [..., (1)/3]
    """
    azimuth = 30 / 180.0 * math.pi
    wavenumber = 2 * math.pi * freq * math.sqrt(eps_r * mu_r) / c0
    k = wavenumber * torch.Tensor([math.cos(azimuth), math.sin(azimuth), 0.0])
    k = k.to(pos.device).to(pos.dtype)
    kx = (k * pos).sum(dim=-1, keepdim=True)
    Einc = torch.cos(-kx) + 1j * torch.sin(-kx)  # exp(-1j*kx), [..., 1]

    Einc = torch.cat(
        (torch.zeros_like(Einc), torch.zeros_like(Einc), Einc), dim=-1
    )  # [..., 3]

    if return_k:
        return Einc, k

    return Einc


def Hinc_func(
    freq: float, eps_r: float, mu_r: float, pos: torch.Tensor
) -> torch.Tensor:
    """A self-defined function for calculate incident magnetic field H^{inc}

    k = 2 * PI * freq * sqrt(eps_r * mu_r) / c
    H = E / eta

    Args:
        freq (float):
        eps_r (float):
        mu_r (float):
        pos (torch.Tensor): [..., dim=2/3]

    Returns:
        torch.Tensor: [..., (1)/3]
    """
    Einc, k = Einc_func(freq=freq, eps_r=eps_r, mu_r=mu_r, pos=pos, return_k=True)
    k_dir = F.normalize(k, dim=-1)

    eta0 = 1.0  # 376.73
    eta = math.sqrt(mu_r / eps_r) / eta0

    k = k + torch.zeros_like(Einc)
    Hinc = torch.cross(k_dir, Einc, dim=-1) / eta
    return Hinc


def mesh_normalize(
    verts: torch.Tensor, scale: float = 1.0, normalize: bool = True
) -> torch.Tensor:
    if normalize:
        vert_min, _ = verts.min(dim=0)
        vert_max, _ = verts.max(dim=0)
        vert_center = 0.5 * (vert_min + vert_max)
        verts = (verts - vert_center) / (vert_max - vert_min).max() * 2
        verts = verts * 2 - 1  # [-1, 1]

    verts = verts * scale
    verts = verts
    return verts


def G(
    wavenumber: float, p1: torch.Tensor, p2: torch.Tensor, keepdim: bool
) -> torch.Tensor:
    """3D Green function

    G = exp(-1j * k * r) / (4 * PI * r)

    Args:
        wavenumber (float): _description_
        p1 (torch.Tensor): _description_
        p2 (torch.Tensor): _description_
        keepdim (bool): _description_

    Returns:
        torch.Tensor: _description_
    """
    dist = (p1 - p2).norm(dim=-1, keepdim=True)
    singularity_mask = dist < 1e-6
    inv_dist = torch.where(singularity_mask, torch.zeros_like(dist), 1.0 / dist)

    azimuth = 30 / 180.0 * math.pi
    k = wavenumber * torch.Tensor([math.cos(azimuth), math.sin(azimuth), 0.0])
    k = k.to(p1.device).to(p1.dtype)
    kx = k * dist
    G_output = (torch.cos(-kx) + 1j * torch.sin(kx)) / (4 * math.pi) * inv_dist

    if not keepdim:
        G_output = G_output.squeeze(-1)

    return G_output


def gradG_y(wavenumber: float, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """\partial_G \partial_p2

    Args:
        wavenumber (float): _description_
        p1 (torch.Tensor): _description_
        p2 (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    ray_d = F.normalize(p1 - p2, dim=-1)
    dist = (p1 - p2).norm(dim=-1, keepdim=True)
    singularity_mask = dist < 1e-6
    inv_dist = torch.where(singularity_mask, torch.zeros_like(dist), 1.0 / dist)

    azimuth = 30 / 180.0 * math.pi
    k = wavenumber * torch.Tensor([math.cos(azimuth), math.sin(azimuth), 0.0])
    k = k.to(p1.device).to(p1.dtype)
    kx = k * dist

    G_output = G(wavenumber=wavenumber, p1=p1, p2=p2, keepdim=True)
    gradG_y_output = G_output * inv_dist * (1 + 1j * kx) * ray_d

    return gradG_y_output


def main(
    geom: str,
    res: List[int],
    freq: float,
    Z0: float,
    gaussQR: int,
    BEM_type: int,
    order_type: int,
):
    dim = 3
    wavenumber = 2 * math.pi * freq / c0

    # dimension of the
    batch_size = 1
    simulation_size = (batch_size, 1, *res)

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # create renderer with SoftRas
    rasterizer = sr.SoftRasterizer(image_size=512, near=0.1, background_color=[1, 1, 1])

    # set up the path for saving
    path = pathlib.Path(__file__).parent.absolute()
    save_path = f"{path}/../save"
    mkdir(save_path)
    save_path = f"{save_path}/data_Helmholtz_{dim}d/"
    mkdir(save_path)
    asset_path = f"{path}/../assets/"

    # Load a 3D asset
    verts, faces = LoadSingleMesh(
        obj_path=f"{asset_path}/{geom}", device=device, dtype=dtype
    )
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape
    vert_areas = compute_vert_areas(vertices=verts, faces=faces, keepdim=True)
    # Get your edges for carrying Js.
    edges, face_edge_indices, edge_adjacent_faces = get_edge_unique(
        verts=verts, faces=faces
    )
    # edges_unique = (3648, 2), edges = (7296, 2), faces = torch.Size([2432, 3])
    print(f"edges_unique = {edges.shape}, faces = {faces.shape}")
    assert edges.shape[0] * 2 == faces.shape[0] * 3
    render_mesh_to_file(
        img_dir=save_path,
        file_label="normals",
        verts=verts * 10,
        faces=faces,
        rasterizer=rasterizer,
    )

    # set up the simulation parameters
    BEM_params = {"gaussQR": gaussQR, "BEM_type": BEM_type, "order_type": order_type}
    EMsimulationParameters = EMSimulationParameters(
        dim=dim,
        dt=None,
        dx=None,
        simulation_size=simulation_size,
        freq=freq,
        Z0=Z0,
        save_path=save_path,
        BEM_params=BEM_params,
        dtype=dtype,
        device=device,
    )

    # create a simulation runner
    logger = Logger(root_path="", log_to_disk=False)
    simulationRunner = SimulationRunner(
        EM_parameters=EMsimulationParameters, logger=logger
    )
    BEM_solver = simulationRunner.create_mesh_BEM_solver()

    logger.InfoLog(
        f"The simulation of {dim}d MOM: wavenumber={wavenumber}, lambda = {2 * math.pi / wavenumber}"
    )

    # create MOM solver
    results = {}

    EM_Js = BEM_solver.solve_MOM(
        verts=verts, faces=faces, edges=edges, face_edge_indices=face_edge_indices
    )  # [B, N_verts]
    EM_Js = EM_Js.reshape(N_verts, dim)

    print(
        f"EM_Js = {EM_Js.shape}, min = {EM_Js.real.min()}, {EM_Js.imag.min()}, max = {EM_Js.real.max()}, {EM_Js.imag.max()}"
    )

    render_mesh_to_file(
        img_dir=save_path,
        file_label=f"Js",
        verts=verts * 10,
        faces=faces,
        vert_phi=EM_Js[None].norm(dim=-1, keepdim=True),
        rasterizer=rasterizer,
    )

    exit(0)

    meshgrid = create_3d_meshgrid_tensor(simulation_size, device=device, dtype=dtype)
    meshgrid = meshgrid.reshape(dim, -1).permute((1, 0))
    meshgrid = mesh_normalize(meshgrid, scale=5)

    # Radiate from Js to E, H
    Nm = meshgrid.numel() // dim
    pm = meshgrid.reshape(Nm, 1, dim)
    H_scattered = -(
        vert_areas
        * torch.cross(
            EM_Js[None].repeat(Nm, 1, 1),
            -gradG_y(wavenumber=wavenumber, p1=pm, p2=verts),  # gradG_x
            dim=-1,
        )
    ).sum(
        dim=-2
    )  # [Nm, dim]

    Hinc = Hinc_func(freq=freq, eps_r=1.0, mu_r=1.0, pos=pm[:, 0, :])

    logger.InfoLog(
        f"Js = {EM_Js.shape}, Hinc = {Hinc.shape},  H_scattered = {H_scattered.shape}"
    )

    results["Js"] = EM_Js
    results["H_incident"] = Hinc
    results["H_scattered"] = H_scattered

    save_heatmap(
        Hinc.abs().reshape(*res, dim)[res[0] // 2].norm(dim=-1),
        filename=f"{save_path}/incident",
        title=f"incident",
        vmax=1,
    )
    save_heatmap(
        H_scattered.abs().reshape(*res, dim)[res[0] // 2].norm(dim=-1),
        filename=f"{save_path}/scattered",
        title=f"scattered",
        vmax=1,
    )
    save_heatmap(
        (Hinc + H_scattered).abs().reshape(*res, dim)[res[0] // 2].norm(dim=-1),
        filename=f"{save_path}/total",
        title=f"total",
        vmax=2,
    )


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--geom",
        type=str,
        default="nasaAlmond.ply",
        choices=["nasaAlmond.ply"],
        help="The geometry",
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        help="Simulation size of the current simulation currently only square",
    )
    parser.add_argument("--freq", type=float, default=10e9, help="Default frequency")
    parser.add_argument("--Z0", type=float, default=1.0, help="Default Z0")
    parser.add_argument(
        "--gaussQR", type=int, default=4, help="The number of Gauss Points on surface"
    )
    parser.add_argument(
        "--BEM_type",
        type=int,
        default=int(BEMType.HELMHOLTZ_MOM),
        choices=[0, 1],
        help="The type of solving PDEs",
    )
    parser.add_argument(
        "--order_type", type=int, default=1, choices=[0, 1], help="Planar or Linear"
    )

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
