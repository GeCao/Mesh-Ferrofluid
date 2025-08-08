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
    create_2d_meshgrid_tensor,
    compute_vert_normals,
    compute_vert_areas,
    Hankel,
)


def Einc_func(
    wavenumber: float,
    pos: torch.Tensor,
    vert_normals: torch.Tensor,
    keepdim: bool = False,
) -> torch.Tensor:
    """A self-defined function for calculate incident electrical field E^{inc}

    Args:
        wavenumber (float): wave number k = 2 * PI * freq / c
        pos (torch.Tensor): [..., dim=2/3]
        keepdim (bool, optional): keepdim (for 2D only). Defaults to False.

    Returns:
        torch.Tensor: [..., (1)/3]
    """
    Einc = torch.cos(-wavenumber * pos[..., 0]) + 1j * torch.sin(
        -wavenumber * pos[..., 0]
    )

    # When k in x-direction,
    # You only got y-direction as your amplitude (2D).
    if vert_normals is not None:
        Einc = Einc * vert_normals[..., 1]

    if keepdim:
        Einc = Einc.unsqueeze(-1)

    return Einc


def Hinc_func(
    wavenumber: float,
    pos: torch.Tensor,
    vert_normals: torch.Tensor,
    keepdim: bool = False,
) -> torch.Tensor:
    """A self-defined function for calculate incident magnetic field H^{inc}

    Args:
        wavenumber (float): wave number k = 2 * PI * freq / c
        pos (torch.Tensor): [..., dim=2/3]
        keepdim (bool, optional): keepdim (for 2D only). Defaults to False.

    Returns:
        torch.Tensor: [..., (1)/3]
    """
    Hinc = torch.cos(-wavenumber * pos[..., 0]) + 1j * torch.sin(
        -wavenumber * pos[..., 0]
    )

    # When k in x-direction,
    # You only got y-direction as your amplitude (2D).
    # if vert_normals is not None:
    #     Hinc = Hinc * vert_normals[..., 1]

    if keepdim:
        Hinc = Hinc.unsqueeze(-1)

    return Hinc


def mesh_normalize(
    verts: torch.Tensor, scale: float = 1.0, normalize: bool = True
) -> torch.Tensor:
    if normalize:
        vert_min, _ = verts.min(dim=0)
        vert_max, _ = verts.max(dim=0)
        verts = (verts - vert_min) / (vert_max - vert_min).max()
        verts = verts * 2 - 1  # [-1, 1]

    verts = verts * scale
    verts = verts
    return verts


def G(
    wavenumber: float,
    order: int,
    p1: torch.Tensor,
    p2: torch.Tensor,
    keepdim: bool,
    vert_areas: torch.Tensor = None,
) -> torch.Tensor:
    raw_dist = (p1 - p2).norm(dim=-1, keepdim=True)
    singularity_mask = raw_dist < 1e-6
    dist = torch.where(singularity_mask, torch.ones_like(raw_dist), raw_dist)

    Hankel_gamma = 1.7180724
    hankel_input = wavenumber * dist
    # # take kind = 1
    # hankel_output = Hankel(order=order, kind=1, z=hankel_input)
    # singular_output = 1 + (1j * 2 / math.pi) * (
    #     torch.log(wavenumber * raw_dist / 2) + Hankel_gamma
    # )

    # take kind = 2
    hankel_output = Hankel(order=order, kind=2, z=hankel_input)
    if vert_areas is None:
        singular_output = torch.zeros_like(hankel_output)
    else:
        singular_output = 1 - (1j * 2 / math.pi) * (
            torch.log(wavenumber * Hankel_gamma * vert_areas / (4 * math.e))
        )
        singular_output = singular_output + torch.zeros_like(hankel_output)

    G_output = torch.where(singularity_mask, singular_output, hankel_output)

    # # take kind = 1
    # G_output = 0.25 * (-G_output.imag + 1j * G_output.real)

    # take kind = 2
    G_output = 0.25 * (G_output.imag - 1j * G_output.real)

    if not keepdim:
        G_output = G_output.squeeze(-1)

    return G_output


def gradG_y(
    wavenumber: float,
    p1: torch.Tensor,
    p2: torch.Tensor,
    n2: torch.Tensor,
    vert_areas: torch.Tensor,
    singularity_val: float,
    keepdim: bool,
) -> torch.Tensor:
    ray_d = F.normalize(p1 - p2, dim=-1)
    dist = (p1 - p2).norm(dim=-1, keepdim=True)
    singularity_mask = dist < 1e-6
    dist = torch.where(singularity_mask, torch.ones_like(dist), dist)

    G_output = G(wavenumber=wavenumber, order=1, p1=p1, p2=p2, keepdim=True)
    gradG_y_output = wavenumber * G_output * (ray_d * n2).sum(dim=-1, keepdim=True)

    gradG_y_output = torch.where(
        singularity_mask,
        singularity_val + torch.zeros_like(gradG_y_output),
        gradG_y_output * vert_areas,
    )

    if not keepdim:
        gradG_y_output = gradG_y_output.squeeze(-1)

    return gradG_y_output


def main(
    geom: str,
    res: List[int],
    k: float,
    Z0: float,
    N_nodes: int,
    gaussQR: int,
    BEM_type: int,
    order_type: int,
):
    dim = 2
    freq = k * c0 / (2 * math.pi)

    # dimension of the
    batch_size = 1
    simulation_size = (batch_size, 1, *res)

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # set up the path for saving
    path = pathlib.Path(__file__).parent.absolute()
    save_path = f"{path}/../save"
    mkdir(save_path)
    save_path = f"{save_path}/data_Helmholtz_2d/"
    mkdir(save_path)

    # Create a 2D circle
    N_verts = N_nodes
    verts = torch.linspace(0, N_verts - 1, N_verts, device=device, dtype=dtype)
    verts = (2 * math.pi) * verts / N_verts  # [0, 2PI]
    verts = torch.stack((torch.cos(verts), torch.sin(verts)), dim=-1)
    faces = (
        torch.Tensor([[i, (i + 1) % N_verts] for i in range(N_verts)])
        .to(device)
        .to(torch.int32)
    )
    vert_normals = compute_vert_normals(vertices=verts, faces=faces)
    vert_areas = compute_vert_areas(vertices=verts, faces=faces, keepdim=True)
    render_mesh_to_file(
        img_dir=save_path, file_label="normals", verts=verts, faces=faces, vert_phi=None
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

    meshgrid = create_2d_meshgrid_tensor(simulation_size, device=device, dtype=dtype)
    meshgrid = meshgrid.reshape(dim, -1).permute((1, 0))
    meshgrid = mesh_normalize(meshgrid, scale=5)

    # create MOM solver
    results = {}
    equation_types = ["TE"]  # ["TM", "TE"]
    for equation_type in equation_types:
        results[equation_type] = {}

        if equation_type == "TM":
            # Use E^{inc} as your B.C.
            Einc = Einc_func(
                wavenumber=k, pos=verts[None], vert_normals=None, keepdim=True
            )
            vert_bc_Dirichlet = Einc
            vert_bc_Neumann = torch.zeros_like(Einc)
            vert_bc_flags = torch.zeros_like(vert_bc_Dirichlet).to(torch.uint8)
            vert_bc_flags[...] = int(BoundaryType.DIRICHLET)
            Amat_scaler = 0 + 1j * torch.Tensor([k * Z0]).to(dtype).to(device)

            EM_Js, _ = BEM_solver.solve(
                verts=verts,
                faces=faces,
                vert_bc_Dirichlet=vert_bc_Dirichlet,
                vert_bc_Neumann=vert_bc_Neumann,
                vert_bc_flags=vert_bc_flags,
                Amat_scaler=Amat_scaler,
            )  # [B, N_verts]
            EM_Js = EM_Js.reshape(N_verts, 1)
        elif equation_type == "TE":
            # Use E^{inc} as your B.C.
            Hinc = Hinc_func(
                wavenumber=k,
                pos=verts[None],
                vert_normals=vert_normals[None],
                keepdim=True,
            )
            vert_bc_Neumann = Hinc
            vert_bc_Dirichlet = torch.zeros_like(Hinc)
            vert_bc_flags = torch.zeros_like(vert_bc_Neumann).to(torch.uint8)
            vert_bc_flags[...] = int(BoundaryType.NEUMANN)
            Amat_scaler = 1.0

            _, EM_Js = BEM_solver.solve(
                verts=verts,
                faces=faces,
                vert_bc_Dirichlet=vert_bc_Dirichlet,
                vert_bc_Neumann=vert_bc_Neumann,
                vert_bc_flags=vert_bc_flags,
                Amat_scaler=Amat_scaler,
            )  # [B, N_verts]
            EM_Js = EM_Js.reshape(N_verts, 1)

        # Radiate from Js to E, H
        Nm = meshgrid.numel() // dim
        pm = meshgrid.reshape(Nm, 1, dim)
        if equation_type == "TM":
            Zmn = vert_areas.squeeze(-1) * G(
                wavenumber=k,
                order=0,
                p1=pm,
                p2=verts,
                keepdim=False,
                vert_areas=vert_areas,
            )
            Zmn = Amat_scaler * Zmn  # Zmn = (1j * k * Z0) * Zmn
            Ez_scattered = -Zmn @ EM_Js
            Ez_inc = Einc_func(k, pm, vert_normals=None, keepdim=False)

            EM_scattered = Ez_scattered.reshape(Nm)
            EM_inc = Ez_inc.reshape(Nm)
        elif equation_type == "TE":
            Zmn = gradG_y(
                wavenumber=k,
                p1=pm,
                p2=verts,
                n2=vert_normals,
                vert_areas=vert_areas,
                singularity_val=-0.5,
                keepdim=False,
            )
            Hz_scattered = -Zmn @ EM_Js
            Hz_inc = Hinc_func(k, pm, vert_normals=None, keepdim=False)

            EM_scattered = Hz_scattered.reshape(Nm)
            EM_inc = Hz_inc.reshape(Nm)

        EM_inc = EM_inc.reshape(*res)
        EM_scattered = EM_scattered.reshape(*res)
        logger.InfoLog(
            f"{equation_type}, EM_Js = {EM_Js.shape}, EM_inc = {EM_inc.shape},  EM_scattered = {EM_scattered.shape}"
        )

        results[equation_type]["Js"] = EM_Js
        results[equation_type]["incident"] = EM_inc
        results[equation_type]["scattered"] = EM_scattered

        # TODO: Default as disk case
        mask = meshgrid.norm(dim=-1).reshape(*res) < 1

        render_mesh_to_file(
            img_dir=save_path,
            file_label=f"{equation_type}_Js",
            verts=verts,
            faces=faces,
            vert_phi=EM_Js[None].abs(),
        )

        save_heatmap(
            EM_inc.abs(),
            filename=f"{save_path}/{equation_type}_incident",
            title=f"{equation_type} (incident)",
            mask=mask,
            vmax=1,
        )
        save_heatmap(
            EM_scattered.abs(),
            filename=f"{save_path}/{equation_type}_scattered",
            title=f"{equation_type} (scattered)",
            mask=mask,
            vmax=1,
        )
        save_heatmap(
            (EM_inc + EM_scattered).abs(),
            filename=f"{save_path}/{equation_type}_total",
            title=f"{equation_type} (total)",
            mask=mask,
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
        default="Analytical_Plane",
        choices=["Analytical_Sphere", "Analytical_Plane"],
        help="The geometry",
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Simulation size of the current simulation currently only square",
    )
    parser.add_argument(
        "--k", type=float, default=2 * math.pi, help="Default wavenumber"
    )
    parser.add_argument("--Z0", type=float, default=1.0, help="Default Z0")
    parser.add_argument(
        "--N_nodes", type=int, default=256, help="The number of nodes on geometry"
    )
    parser.add_argument(
        "--gaussQR", type=int, default=6, help="The number of Gauss Points on surface"
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
