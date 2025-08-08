import sys
import pathlib
import torch
import argparse
from typing import List

sys.path.append("../")

from src.utils import (
    mkdir,
    save_img,
    LoadSingleMesh,
    export_asset,
    meshes_to_voxelgrids,
    render_mesh_to_file,
)

import soft_renderer as sr


def main(asset_name: str = "sphere", res: List[int] = [130, 130, 130]):
    dim = 3

    density_gas = 1
    density_fluid = 1e3

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # set up the size of the simulation
    batch_size = 1
    simulation_size = (batch_size, 1, *res)

    # initialize the domain
    path = pathlib.Path(__file__).parent.absolute()
    save_dir = f"{path}/../save/"
    mkdir(save_dir)
    save_dir = f"{save_dir}/mesh_to_sdf/"
    mkdir(save_dir)

    # load a sphere mesh
    verts, faces = LoadSingleMesh(
        obj_path=f"{path}/../assets/{asset_name}.obj", device=device, dtype=dtype
    )

    # create renderer with SoftRas
    rasterizer = sr.SoftRasterizer(image_size=512, near=0.1, background_color=[1, 1, 1])

    # save the density image to file
    density_filepath = str(save_dir) + "/density.png"
    grid_density = meshes_to_voxelgrids(
        simulation_size=simulation_size, verts=verts, faces=faces
    )
    save_img(grid_density, filename=density_filepath)

    # render final frame mesh to file
    render_mesh_to_file(
        img_dir=save_dir,
        file_label="normals",
        rasterizer=rasterizer,
        verts=verts,
        faces=faces,
        show_normals=True,
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
        "--asset_name", type=str, default="sphere", help="The asset will be loaded"
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
