"""Copyright (C) 2025, Ge Cao
ACEM research group, https://acem.ece.illinois.edu/
All rights reserved.

This software is free for non-commercial, research and evaluation use
under the terms of the LICENSE.md file.

For inquiries contact  gecao2@illinois.edu
"""

import sys
import os
import time
import pathlib
import math
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import trimesh
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
    compute_face_areas,
    compute_face_normals,
    LoadSingleMesh,
    UniformSampleHemisphereOnSurface,
    GetRayQueryDepthMap,
    G,
    gradG_y,
    initialize_vulkan_ray_querier,
    div_Js,
    vis_vert_Js,
    LoadSDFFromMeshPath,
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
    Einc = torch.cos(-kx) + 1j * torch.sin(-kx)  # exp(+1j*kx), [..., 1]

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


def test_ray_tracer(
    save_path: str, vulkan_ray_tracer, verts: torch.Tensor, faces: torch.Tensor
):
    N_verts, dim = verts.shape
    face_normals = compute_face_normals(vertices=verts, faces=faces)

    # test if this ray-tracer works by shooting rays.
    fig = plt.figure(figsize=(5, 4))
    axs = fig.add_subplot(1, 1, 1, projection="3d")

    x0 = torch.index_select(verts, dim=-2, index=faces[0].to(torch.int64)).mean(
        dim=-2, keepdim=True
    )
    n0 = face_normals[0:1]
    np_x0 = x0.cpu().numpy()
    N_sample = 100
    x0_ray_d, _ = UniformSampleHemisphereOnSurface(
        normal=-n0, N_sample_per_surface=N_sample
    )
    x0_ray_d = x0_ray_d.reshape(N_sample, dim)
    x0_depth, x1_face_idx = GetRayQueryDepthMap(
        vulkan_ray_tracer=vulkan_ray_tracer,
        ray_o=x0,
        ray_d=x0_ray_d.reshape(1, N_sample, dim),
        keepdim=True,
    )  # [1, n_rays, 1]
    x0_depth[x0_depth < 0] = 0
    x0_depth = x0_depth.reshape(N_sample, 1)
    x1_face_idx = x1_face_idx.reshape(N_sample, 1)
    x1 = x0 + torch.clamp(x0_depth, 0) * x0_ray_d
    ray_lines = torch.stack((x0 + torch.zeros_like(x1), x1), dim=-2)  # [n_rays, 2, dim]
    np_ray_lines = ray_lines.cpu().numpy()
    ray_lc = Line3DCollection(np_ray_lines, colors="blue", linewidths=0.1)
    axs.add_collection3d(ray_lc)

    n1 = torch.index_select(
        face_normals, dim=-2, index=x1_face_idx.flatten().to(torch.int64)
    )
    x1_ray_d, _ = UniformSampleHemisphereOnSurface(normal=-n1, N_sample_per_surface=1)
    x1_ray_d = x1_ray_d.reshape(N_sample, dim)
    x1_depth, x2_face_idx = GetRayQueryDepthMap(
        vulkan_ray_tracer=vulkan_ray_tracer,
        ray_o=x1.reshape(N_sample, dim),
        ray_d=x1_ray_d.reshape(N_sample, 1, dim),
        keepdim=True,
    )  # [1, n_rays, 1]
    x1_depth[x1_depth < 0] = 0
    x1_depth = x1_depth.reshape(N_sample, 1)
    x2 = x1 + x1_depth * x1_ray_d
    ray2_lines = torch.stack((x1, x2), dim=-2)  # [n_rays, 2, dim]
    np_ray2_lines = ray2_lines.cpu().numpy()
    ray2_lc = Line3DCollection(np_ray2_lines, colors="red", linewidths=0.1)
    axs.add_collection3d(ray2_lc)

    edges, _, _ = get_edge_unique(verts=verts, faces=faces)
    edge_verts = torch.index_select(
        verts, dim=-2, index=edges.flatten().to(torch.int64)
    )
    edge_verts = edge_verts.reshape(edges.shape[-2], 2, dim)
    np_edge_verts = edge_verts.cpu().numpy()
    edge_lc = Line3DCollection(np_edge_verts, colors="black", linewidths=0.01)
    axs.add_collection3d(edge_lc)

    axs.set_title(f"ray tracing inside a mesh")
    axs.set_xlabel(f"X")
    axs.set_ylabel(f"Y")
    axs.set_zlabel(f"Z")
    axs.set_xlim(-0.1, 0.1)
    axs.set_ylim(-0.1, 0.1)
    axs.set_zlim(-0.1, 0.1)
    # axs.set_xticklabels([])
    # axs.set_yticklabels([])
    # axs.set_zticklabels([])
    # axs.axis("equal")
    axs.text(
        x=np_x0[0, 0],
        y=np_x0[0, 1],
        z=np_x0[0, 2],
        s=r"$x_{0}$",
        fontdict={"color": "black", "size": 20},
    )
    # axs.axis("off")
    plt.savefig(
        os.path.join(save_path, "ray_test.png"), pad_inches=0, bbox_inches="tight"
    )
    plt.close()


def main(geom: str, res: List[int], freq: float, Z0: float):
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
    save_path = f"{save_path}/data_WOB_Helmholtz_{dim}d/"
    mkdir(save_path)
    asset_path = f"{path}/../assets/"

    # Load a 3D asset
    verts, faces = LoadSingleMesh(
        obj_path=f"{asset_path}/{geom}", device=device, dtype=dtype
    )
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape
    vert_areas = compute_vert_areas(vertices=verts, faces=faces, keepdim=True)
    face_areas = compute_face_areas(vertices=verts, faces=faces, keepdim=True)
    render_mesh_to_file(
        img_dir=save_path,
        file_label="normals",
        verts=verts,
        faces=faces,
        rasterizer=rasterizer,
    )

    # set up the simulation parameters
    EMsimulationParameters = EMSimulationParameters(
        dim=dim,
        dt=None,
        dx=None,
        simulation_size=simulation_size,
        freq=freq,
        Z0=Z0,
        save_path=save_path,
        dtype=dtype,
        device=device,
    )

    # create a simulation runner
    logger = Logger(root_path="", log_to_disk=False)
    simulationRunner = SimulationRunner(
        EM_parameters=EMsimulationParameters, logger=logger
    )
    WOB_solver = simulationRunner.create_mesh_WOB_solver()

    logger.InfoLog(
        f"The simulation of {dim}d WOB: wavenumber={wavenumber}, lambda = {2 * math.pi / wavenumber}"
    )

    # create ray-tracer
    ray_querier = initialize_vulkan_ray_querier(verts=verts, faces=faces)
    # test_ray_tracer(
    #     save_path=save_path, vulkan_ray_tracer=ray_querier, verts=verts, faces=faces
    # )

    # create WOB solver
    start_time = time.time()
    EM_Js, ray_lines, face_EM_Js = WOB_solver.solve_WOB(
        vulkan_ray_tracer=ray_querier,
        verts=verts,
        faces=faces,
        Einc_func=Einc_func,
        Hinc_func=Hinc_func,
    )  # [B, N_verts]
    end_time = time.time()
    EM_Js = EM_Js.reshape(N_verts, dim)
    face_EM_Js = face_EM_Js.reshape(N_faces, dim)
    # WOB_solver.vis_rays(
    #     save_path=save_path, ray_lines=ray_lines, verts=verts, faces=faces
    # )
    vis_vert_Js(
        save_path=os.path.join(save_path, "WOB_vis_Js.png"), vert_Js=EM_Js, verts=verts
    )

    logger.InfoLog(
        f"EM_Js = {EM_Js.shape}, min = {EM_Js.real.min()}, {EM_Js.imag.min()}, max = {EM_Js.real.max()}, {EM_Js.imag.max()}"
    )
    logger.InfoLog(f"Time consuming: {end_time - start_time} [s]")

    render_mesh_to_file(
        img_dir=save_path,
        file_label=f"Js",
        verts=verts,
        faces=faces,
        vert_phi=EM_Js[None].real,
        rasterizer=rasterizer,
    )

    meshgrid = create_2d_meshgrid_tensor(
        [batch_size, 1, res[0], res[2]], device=device, dtype=dtype
    )
    meshgrid = meshgrid.reshape(2, -1).permute((1, 0))
    meshgrid = torch.cat(
        (meshgrid[..., 0:1], torch.zeros_like(meshgrid[..., 0:1]), meshgrid[..., 1:2]),
        dim=-1,
    )
    meshgrid = mesh_normalize(meshgrid, scale=2)

    face_center = torch.index_select(
        verts, dim=-2, index=faces.flatten().to(torch.int64)
    )
    face_center = face_center.reshape(N_faces, dim, dim).sum(dim=-2)  # [N_faces, dim]

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
    )
    # H_scattered = -(
    #     face_areas
    #     * torch.cross(
    #         face_EM_Js[None].repeat(Nm, 1, 1),
    #         -gradG_y(wavenumber=wavenumber, p1=pm, p2=face_center),  # gradG_x
    #         dim=-1,
    #     )
    # )
    H_scattered = H_scattered.sum(dim=-2)  # [Nm, dim]

    face_div_Js = div_Js(face_Js=face_EM_Js, verts=verts, faces=faces, keepdim=True)
    vert_div_Js = torch.zeros_like(EM_Js[..., 0:1]).reshape(N_verts)
    vert_div_Js.scatter_add_(
        dim=-1,
        index=faces.flatten().to(torch.int64),
        src=face_div_Js.repeat(1, dim).flatten() / dim,
    )
    vert_div_Js = vert_div_Js.reshape(N_verts, 1)
    E_scattered = (-1j * wavenumber * vert_areas) * (
        EM_Js * G(wavenumber=wavenumber, p1=pm, p2=verts, keepdim=True)
        + (1.0 / wavenumber / wavenumber * vert_div_Js)
        * (-gradG_y(wavenumber=wavenumber, p1=pm, p2=verts))
    )  # [Nm, Np, dim]
    E_scattered = E_scattered.sum(dim=-2)  # [Nm, dim]

    # find mask
    np_sdf = LoadSDFFromMeshPath(meshpath=f"{asset_path}/{geom}", size=max(res))
    np_sdf = np_sdf[0, 0, :, np_sdf.shape[-2] // 2, :]  # [W, D]
    np_sdf = np_sdf.transpose((1, 0))
    sdf = torch.from_numpy(np_sdf).to(device).to(dtype)
    mask = sdf <= 0

    Hinc = Hinc_func(wavenumber=wavenumber, pos=pm[:, 0, :])
    Einc = Einc_func(wavenumber=wavenumber, pos=pm[:, 0, :])

    logger.InfoLog(
        f"Js = {EM_Js.shape}, Hinc = {Hinc.shape},  H_scattered = {H_scattered.shape}"
    )
    logger.InfoLog(f"Lambda = {2 * math.pi / wavenumber}")

    save_heatmap(
        Hinc.abs().reshape(res[0], res[2], dim).norm(dim=-1),
        filename=f"{save_path}/Hincident",
        title=f"H incident",
        mask=mask,
    )
    save_heatmap(
        H_scattered.abs().reshape(res[0], res[2], dim).norm(dim=-1),
        filename=f"{save_path}/Hscattered",
        title=f"H scattered",
        mask=mask,
    )
    save_heatmap(
        (Hinc + H_scattered).abs().reshape(res[0], res[2], dim).norm(dim=-1),
        filename=f"{save_path}/Htotal",
        title=f"H total",
        mask=mask,
    )

    save_heatmap(
        Einc.abs().reshape(res[0], res[2], dim).norm(dim=-1),
        filename=f"{save_path}/Eincident",
        title=f"E incident",
        mask=mask,
    )
    save_heatmap(
        E_scattered.abs().reshape(res[0], res[2], dim).norm(dim=-1),
        filename=f"{save_path}/Escattered",
        title=f"E scattered",
        mask=mask,
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
        default="spot.ply",
        choices=[
            "fine_sphere.obj",
            "nasaAlmond_v3.ply",
            "nasaAlmond_v2.ply",
            "human.ply",
            "human_v2.ply",
            "spot.ply",
            "spot_v2.ply",
            "cheburashka.ply",
        ],
        help="The geometry",
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        help="Simulation size of the current simulation currently only square",
    )
    parser.add_argument("--freq", type=float, default=0.5e9, help="Default frequency")
    parser.add_argument("--Z0", type=float, default=1.0, help="Default Z0")

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
