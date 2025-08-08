import sys
import pathlib
import torch
import argparse
from typing import List

sys.path.append("../")

from src.utils import (
    mkdir,
    Logger,
    LoadSingleMesh,
    export_asset,
    compute_vert_normals,
    render_mesh_to_file,
    remesh_and_transfer_velocity,
)

import soft_renderer as sr


def main(asset_name: str = "sphere", res: List[int] = [130, 130, 130]):
    dim = 3

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
    save_dir = f"{save_dir}/remeshing/"
    mkdir(save_dir)

    logger = Logger(root_path="", log_to_disk=False)

    # load a sphere mesh
    verts, faces = LoadSingleMesh(
        obj_path=f"{path}/../assets/{asset_name}", device=device, dtype=dtype
    )
    vert_normals = compute_vert_normals(vertices=verts, faces=faces)

    # [x, y, z] -> [x, z, y] and Normalize:
    verts = torch.stack((verts[..., 0], verts[..., 2], -verts[..., 1]), dim=-1)
    verts = verts - verts.mean(dim=-2, keepdim=True)
    verts = verts / verts.norm(dim=-1).max()

    # create renderer with SoftRas
    rasterizer = sr.SoftRasterizer(image_size=512, near=0.1, background_color=[1, 1, 1])

    # render final frame mesh to file
    render_mesh_to_file(
        img_dir=save_dir,
        file_label="sparse normals",
        rasterizer=rasterizer,
        verts=verts,
        faces=faces,
        vert_phi=vert_normals[None],
    )

    logger.InfoLog(f"[Before remeshing]: verts = {verts.shape}, faces = {faces.shape}")
    N_faces = faces.shape[-2]
    new_verts, new_faces, new_vert_normals = remesh_and_transfer_velocity(
        verts=verts, faces=faces, vel=vert_normals[None], target_edge_length=0.03
    )
    logger.InfoLog(
        f"[After  remeshing]: verts = {new_verts.shape}, faces = {new_faces.shape}"
    )

    # render final frame mesh to file
    render_mesh_to_file(
        img_dir=save_dir,
        file_label="fine normals",
        rasterizer=rasterizer,
        verts=new_verts,
        faces=new_faces,
        vert_phi=new_vert_normals,
    )

    # export this final mesh to file
    save_mesh_path = str(save_dir) + "/mesh.obj"
    export_asset(save_mesh_path, vertices=new_verts, faces=new_faces)


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )

    parser.add_argument(
        "--asset_name", type=str, default="sphere.obj", help="The asset will be loaded"
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
