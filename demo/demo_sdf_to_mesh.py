import sys
import pathlib
import torch
import argparse
from typing import List

sys.path.append("../")

from src.utils import (
    mkdir,
    save_img,
    create_droplet,
    export_asset,
    voxelgrids_to_cubic_meshes,
    render_mesh_to_file,
)

import soft_renderer as sr


def main(res: List[int] = [130, 130, 130]):
    dim = 3

    density_gas = 1
    density_fluid = 1e3

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # set up the size of the simulation
    batch_size = 1
    simulation_size = (batch_size, 1, *res)

    # create renderer with SoftRas
    rasterizer = sr.SoftRasterizer(image_size=512, near=0.1, background_color=[1, 1, 1])

    # initialize the domain
    path = pathlib.Path(__file__).parent.absolute()
    save_dir = f"{path}/../save/"
    mkdir(save_dir)
    save_dir = f"{save_dir}/sdf_to_mesh/"
    mkdir(save_dir)

    # initialize the density
    density = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)

    # create a cube droplet
    density[...] = density_gas
    radius = min(res) * 0.225
    droplet_center = (
        torch.Tensor([res[-1] / 2, res[-2] / 2, res[-3] / 2]).to(device).to(dtype)
    )
    density = create_droplet(
        density, center=droplet_center, radius=radius, rho_liquid=density_fluid
    )

    phi = (density - density_gas) / (density_fluid - density_gas) * 2 - 1

    # save the density image to file
    density_filepath = str(save_dir) + "/density.png"
    save_img(density, filename=density_filepath)

    # render final frame mesh to file
    verts, faces = voxelgrids_to_cubic_meshes(phi, normlize=True)
    verts, faces = verts[0], faces[0]
    render_mesh_to_file(
        img_dir=save_dir,
        file_label="mesh",
        rasterizer=rasterizer,
        verts=verts,
        faces=faces,
    )

    # export this final mesh to file
    save_mesh_path = str(save_dir) + "/mesh.obj"
    export_asset(save_mesh_path, vertices=verts, faces=faces)


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[64 + 2, 64 + 2, 64 + 2],
        help="Simulation size of the current simulation currently only square",
    )

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
