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
import json
import matplotlib.pyplot as plt
from typing import List
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.append("../")
from src.simulation import EMSimulationParameters, SimulationRunner
from src.utils import (
    mkdir,
    render_mesh_to_file,
    c0,
    Logger,
    save_heatmap,
    create_2d_meshgrid_tensor,
    get_edge_unique,
    compute_vert_areas,
    compute_face_normals,
    LoadSingleMesh,
    UniformSampleHemisphereOnSurface,
    GetRayQueryDepthMap,
    G,
    gradG_y,
    initialize_vulkan_ray_querier,
)

import soft_renderer as sr


def Einc_func(wavenumber: float, pos: torch.Tensor) -> torch.Tensor:
    """A self-defined function for calculate incident electrical field E^{inc}
    azimuth angle = 30,
    Pitch angle = 90.

    k = 2 * PI * freq * sqrt(eps_r * mu_r) / c

    Args:
        wavenumber (float):
        pos (torch.Tensor): [..., dim=2/3]

    Returns:
        torch.Tensor: [..., (1)/3]
    """
    # 1. Plane wave
    azimuth = 30 / 180.0 * math.pi
    k = wavenumber * torch.Tensor([math.cos(azimuth), math.sin(azimuth), 0.0])
    k = k.to(pos.device).to(pos.dtype)
    kx = (k * pos).sum(dim=-1, keepdim=True)
    Einc = torch.cos(kx) + 1j * torch.sin(kx)  # exp(+1j*kx), [..., 1]

    Einc = torch.cat(
        (torch.zeros_like(Einc), torch.zeros_like(Einc), Einc), dim=-1
    )  # [..., 3]

    # # 2. Dipole
    # dtype = pos.dtype
    # device = pos.device
    # eps_r = 1.0
    # eps0 = 1.0
    # tx_pos = torch.Tensor([-5.0, 0.0, 0.0]).to(device).to(dtype)
    # tx_p = torch.Tensor([0.0, 0.0, 1.0]).to(device).to(dtype)
    # r = pos - tx_pos
    # r_dist = r.norm(dim=-1, keepdim=True)
    # r_dir = F.normalize(r, dim=-1)
    # kr = wavenumber * r_dist
    # Einc = (wavenumber * wavenumber / r_dist) * torch.cross(
    #     torch.cross(r_dir, tx_p + torch.zeros_like(r_dir), dim=-1), r_dir, dim=-1
    # )
    # Einc = Einc + (1 - 1j * wavenumber * r_dist) / torch.pow(r_dist, 3) * (
    #     (
    #         3 * r_dir[..., None] * r_dir[..., None, :]
    #         - torch.eye(3, 3, dtype=dtype, device=device)
    #     )
    #     @ tx_p.unsqueeze(-1)
    # ).squeeze(-1)
    # Einc = Einc * (torch.cos(kr) + 1j * torch.sin(kr)) / (4 * math.pi * eps_r * eps0)

    return Einc


def Hinc_func(wavenumber: float, pos: torch.Tensor) -> torch.Tensor:
    """A self-defined function for calculate incident magnetic field H^{inc}

    k = 2 * PI * freq * sqrt(eps_r * mu_r) / c
    H = E / eta

    Args:
        wavenumber (float):
        pos (torch.Tensor): [..., dim=2/3]

    Returns:
        torch.Tensor: [..., (1)/3]
    """
    # 1. Plane wave
    Einc = Einc_func(wavenumber=wavenumber, pos=pos)
    azimuth = 30 / 180.0 * math.pi
    k_dir = torch.Tensor([math.cos(azimuth), math.sin(azimuth), 0.0])
    k_dir = k_dir.to(pos.device).to(pos.dtype)

    eta0 = 1.0  # 376.73
    eps_r = 1.0
    mu_r = 1.0
    eta = math.sqrt(mu_r / eps_r) * eta0

    Hinc = torch.cross(k_dir + torch.zeros_like(Einc), Einc, dim=-1) / eta

    # # 2. Dipole
    # dtype = pos.dtype
    # device = pos.device
    # tx_pos = torch.Tensor([-5.0, 0.0, 0.0]).to(device).to(dtype)
    # tx_p = torch.Tensor([0.0, 0.0, 1.0]).to(device).to(dtype)
    # r = pos - tx_pos
    # r_dist = r.norm(dim=-1, keepdim=True)
    # r_dir = F.normalize(r, dim=-1)
    # kr = wavenumber * r_dist
    # Hinc = (wavenumber * wavenumber / r_dist) * torch.cross(
    #     r_dir, tx_p + torch.zeros_like(r_dir), dim=-1
    # )
    # Hinc = Hinc + (-1 + 1j * wavenumber * r_dist) / torch.pow(r_dist, 3) * torch.cross(
    #     r_dir, tx_p + torch.zeros_like(r_dir), dim=-1
    # )
    # Hinc = Hinc * (torch.cos(kr) + 1j * torch.sin(kr)) / (4 * math.pi)

    return Hinc


def mesh_normalize(
    verts: torch.Tensor, scale: float = 1.0, normalize: bool = True
) -> torch.Tensor:
    if normalize:
        vert_min, _ = verts.min(dim=0)
        vert_max, _ = verts.max(dim=0)
        vert_center = 0.5 * (vert_min + vert_max)
        verts = (verts - vert_center) / (vert_max - vert_min).max() * 2
        # verts = verts * 2 - 1  # [-1, 1]

    verts = verts * scale
    verts = verts
    return verts


def main(geom: str, res: List[int], freq: float):
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
    save_path = f"{save_path}/data_BEAST_{dim}d/"
    mkdir(save_path)
    asset_path = f"{path}/../assets/"

    # Load a 3D asset
    verts, faces = LoadSingleMesh(
        obj_path=f"{asset_path}/{geom}", device=device, dtype=dtype
    )
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape
    vert_areas = compute_vert_areas(vertices=verts, faces=faces, keepdim=True)

    # Read json file from BEAST
    with open(os.path.join(save_path, "Esc.json"), "r") as load_f:
        json_data = json.load(load_f)

        E_scattered_real = json_data["Esc_real"]
        E_scattered_imag = json_data["Esc_imag"]
        E_scattered_real = torch.Tensor(E_scattered_real).to(dtype).to(device)
        E_scattered_imag = torch.Tensor(E_scattered_imag).to(dtype).to(device)
        E_scattered = E_scattered_real + 1j * E_scattered_imag
        E_scattered = -E_scattered

        Einc_real = json_data["Ein_real"]
        Einc_imag = json_data["Ein_imag"]
        Einc_real = torch.Tensor(Einc_real).to(dtype).to(device)
        Einc_imag = torch.Tensor(Einc_imag).to(dtype).to(device)
        Einc = Einc_real + 1j * Einc_imag
    load_f.close()
    Nm = E_scattered.numel() // dim

    # create a simulation runner
    logger = Logger(root_path="", log_to_disk=False)

    logger.InfoLog(f"Load E_scattered from BEAST: E_scattered = {E_scattered.shape}")

    # create ray-tracer
    ray_querier = initialize_vulkan_ray_querier(verts=verts, faces=faces)

    meshgrid = create_2d_meshgrid_tensor(
        [batch_size, 1, res[0], res[2]], device=device, dtype=dtype
    )
    meshgrid = meshgrid.reshape(2, -1).permute((1, 0))
    meshgrid = torch.cat(
        (meshgrid[..., 0:1], torch.zeros_like(meshgrid[..., 0:1]), meshgrid[..., 1:2]),
        dim=-1,
    )
    meshgrid = mesh_normalize(meshgrid, scale=2)
    Nm = meshgrid.numel() // dim
    pm = meshgrid.reshape(Nm, 1, dim)

    # find mask
    mask = None
    pm_rayd = (
        torch.Tensor(
            [[[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]]
        )
        .to(dtype)
        .to(device)
        .repeat(Nm, 1, 1)
    )
    pm_depth, _ = GetRayQueryDepthMap(
        vulkan_ray_tracer=ray_querier, ray_o=pm.reshape(Nm, dim), ray_d=pm_rayd
    )
    pm_depth = pm_depth.reshape(Nm, 6, 1)
    mask = (pm_depth >= 0).sum(dim=-2) == pm_depth.shape[-2]  # [Nm, 1]
    mask = mask | (((pm_depth < 0.01) & (pm_depth >= 0)).sum(dim=-2) > 0)
    mask = mask.reshape(res[0], res[2])

    # Einc = Einc_func(wavenumber=wavenumber, pos=pm[:, 0, :]).reshape_as(E_scattered)

    # save_heatmap(
    #     Hinc.abs().reshape(res[0], res[2], dim).norm(dim=-1),
    #     filename=f"{save_path}/incident",
    #     title=f"incident",
    #     mask=mask,
    # )
    save_heatmap(
        E_scattered.abs().reshape(res[0], res[2], dim).norm(dim=-1),
        filename=f"{save_path}/Escattered",
        title=f"E scattered",
        mask=mask,
        vmax=1,
    )
    save_heatmap(
        (Einc + E_scattered).abs().reshape(res[0], res[2], dim).norm(dim=-1),
        filename=f"{save_path}/Etotal",
        title=f"E total",
        mask=mask,
    )


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--geom",
        type=str,
        default="nasaAlmond_v3.ply",
        choices=[
            "fine_sphere.obj",
            "nasaAlmond_v3.ply",
            "nasaAlmond_v2.ply",
            "human.ply",
            "human_v2.ply",
        ],
        help="The geometry",
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[50, 50, 50],
        help="Simulation size of the current simulation currently only square",
    )
    parser.add_argument("--freq", type=float, default=1e9, help="Default frequency")

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
