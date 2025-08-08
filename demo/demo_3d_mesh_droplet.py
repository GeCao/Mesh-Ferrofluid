import sys
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
import imageio
import argparse
import math
from pytorch3d.structures import Meshes
from typing import List

sys.path.append("../")

from src.simulation import FluidSimulationParameters, SimulationRunner
from src.utils import (
    mkdir,
    Logger,
    BoundaryType,
    LoadSingleMesh,
    export_asset,
    render_mesh_to_file,
    remesh_and_transfer_velocity,
)
from tqdm import tqdm

import soft_renderer as sr


def main(
    asset_name: str = "sphere",
    res: List[int] = [130, 130, 130],
    total_steps: int = 350,
    dt: float = 5e-4,
    dx: float = 6e-4,
    gravity_strength: float = 9.81,
    surface_tension: float = 0.02,
    gaussQR: int = 8,
    order_type: int = 1,
):
    dim = 3

    density_gas = 1
    density_fluid = 1.3e3
    density_wall = 1

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # set up the size of the simulation
    batch_size = 1
    simulation_size = (batch_size, 1, *res)

    path = pathlib.Path(__file__).parent.absolute()
    save_dir = f"{path}/../save/"
    mkdir(save_dir)
    save_dir = f"{save_dir}/mesh_{dim}d_droplet/"
    mkdir(save_dir)
    img_dir = f"{save_dir}/img/"
    mkdir(img_dir)
    mesh_dir = f"{save_dir}/mesh/"
    mkdir(mesh_dir)
    plt_dir = f"{save_dir}/plt/"
    mkdir(plt_dir)
    fileList = []

    # create renderer with SoftRas
    rasterizer = sr.SoftRasterizer(image_size=512, near=0.1, background_color=[1, 1, 1])

    logger = Logger(root_path="", log_to_disk=False)

    # load a sphere mesh
    verts, faces = LoadSingleMesh(
        obj_path=f"{path}/../assets/{asset_name}", device=device, dtype=dtype
    )
    N_verts = verts.shape[-2]
    N_faces = faces.shape[-2]
    vert_vel = torch.zeros((batch_size, N_verts, dim)).to(device).to(dtype)

    # [x, y, z] -> [x, z, y] and Normalize:
    verts = torch.stack((verts[..., 0], verts[..., 2], -verts[..., 1]), dim=-1)
    verts = verts - verts.mean(dim=-2, keepdim=True)
    verts = verts / verts.norm(dim=-1).max()  # [Normalize to -1 , 1]
    L = 5 * (2 * math.pi / math.sqrt(density_fluid * 9.81 / surface_tension))
    logger.InfoLog(f"The Zone of this liquid: L = {L}")
    verts = verts * (L / 2)

    target_edge_length = 0.15 * L
    verts, faces, vert_vel = remesh_and_transfer_velocity(
        verts=verts,
        faces=faces,
        vel=vert_vel,
        target_edge_length=target_edge_length,
    )

    render_mesh_to_file(
        img_dir=img_dir,
        file_label="Initial",
        verts=verts / (L / 2),
        faces=faces,
        rasterizer=rasterizer,
    )
    logger.InfoLog(f"[After remeshing]: verts = {verts.shape}, faces = {faces.shape}")

    # set up the simulation parameters
    surface_only_params = {"gaussQR": gaussQR, "order_type": order_type}
    multiphase_params = {
        "density_gas": density_gas,
        "surface_tension": surface_tension,
        "contact_angle": 0.5 * math.pi,
    }
    simulationParameters = FluidSimulationParameters(
        dim=dim,
        dtype=dtype,
        device=device,
        simulation_size=simulation_size,
        dt=dt,
        density_fluid=density_fluid,
        gravity_strength=gravity_strength,
        multiphase_params=multiphase_params,
        surface_only_params=surface_only_params,
        k=0.33,
    )
    # create a simulation runner
    simulationRunner = SimulationRunner(
        fluid_parameters=simulationParameters, logger=logger
    )

    # create solvers
    advection = simulationRunner.create_mesh_advection()
    Helmholtz_decomposition = simulationRunner.create_mesh_Helmholtz_decomposition()
    pressure_solver = simulationRunner.create_mesh_pressure_solver()

    for step in tqdm(range(total_steps)):
        # 1. Advection and remesh: velocity from Harmonics to Non-Harmonics
        vert_vel = advection.solve(verts=verts, dt=dt, vel=vert_vel)
        verts, faces, vert_vel = remesh_and_transfer_velocity(
            verts=verts,
            faces=faces,
            vel=vert_vel,
            target_edge_length=target_edge_length,
        )
        logger.InfoLog(
            f"[After remeshing]: verts = {verts.shape}, faces = {faces.shape}"
        )
        vert_bc_flags = torch.zeros_like(vert_vel[..., 0:1]).to(torch.uint8)
        vert_bc_flags[...] = int(BoundaryType.DIRICHLET)  # Dirichlet: Air-Liquid

        # 2. Helmholtz decomposition
        vert_vel = Helmholtz_decomposition.solve(
            verts=verts, faces=faces, vert_vel=vert_vel
        )

        # 3. pressure solver
        vert_vel = pressure_solver.solve(
            verts=verts,
            faces=faces,
            dt=dt,
            vert_bc_flags=vert_bc_flags,
            vel=vert_vel,
            P_mag=None,
        )

        simulationRunner.step(dt=dt)
        # impl this
        if step % 10 == 0:
            # render mesh to file
            render_filepath_list = render_mesh_to_file(
                img_dir=img_dir,
                file_label="{:03}".format(step),
                verts=verts / (L / 2),
                faces=faces,
                rasterizer=rasterizer,
            )
            fileList = fileList + render_filepath_list

            # # export this mesh to file
            # save_mesh_path = str(mesh_dir) + "/{:03}.obj".format(step)
            # export_asset(save_mesh_path, vertices=verts, faces=faces)

    # render final frame mesh to file
    render_mesh_to_file(
        img_dir=img_dir,
        file_label="final_mesh",
        verts=verts / (L / 2),
        faces=faces,
        rasterizer=rasterizer,
    )

    # export this final mesh to file
    save_mesh_path = str(save_dir) + "/final_mesh.obj"
    export_asset(save_mesh_path, vertices=verts, faces=faces)

    #  VIDEO Loop
    writer = imageio.get_writer(f"{save_dir}/{dim}d_LBM_droplet.mp4", fps=25)
    for im in fileList:
        writer.append_data(imageio.v2.imread(im))
    writer.close()


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )

    parser.add_argument(
        "--asset_name",
        type=str,
        default="fine_cube.obj",
        help="The asset will be loaded",
    )

    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[64 + 2, 64 + 2, 64 + 2],
        help="Simulation size of the current simulation currently only square",
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=500,
        help="For how many step to run the simulation",
    )

    parser.add_argument("--dt", type=float, default=5e-4, help="Delta t of simulation")
    parser.add_argument("--dx", type=float, default=6e-4, help="Delta x of simulation")

    parser.add_argument(
        "--gravity_strength",
        type=float,
        default=9.81 * 0,
        help=("Gravity Strength [m/s2]"),
    )

    # In Surface-Ferrofluid, Fig. 13, it is [0.02, 0.025, 0.03, 0.05]
    # Corresponding to k = (density_fluid * gravity_strength / surface_tension).sqrt()
    # And wavelength = 2 * math.pi / k: [7.9e-3,  8.8e-3,  9.6e-3, 12.4e-3]
    # For "surface_tension == 0.02" case, if you want to see 5 peaks, L=0.0393 [m]
    # Or, we say: Min. E.L./m = 0.6e-3
    parser.add_argument(
        "--surface_tension", type=float, default=0.02, help=("Surface tension [N/m]")
    )

    parser.add_argument(
        "--gaussQR", type=int, default=2, help="Gaussian integration order"
    )
    parser.add_argument(
        "--order_type", type=int, default=1, choices=[0, 1], help="Planar or Linear"
    )

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
