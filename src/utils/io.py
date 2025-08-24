import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import pywavefront
import trimesh
import cv2
import mesh2sdf
import matplotlib.pyplot as plt
import open3d as o3d
import tecplot as tp
from PIL import Image
from typing import List, Dict
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from src.utils import TaichiTexture


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2numpy_2d_(img):
    # Normalization
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)

    img = img.detach().cpu()
    img = img.permute(0, 2, 3, 1) * 255
    return img[0].numpy()


def tensor2numpy_3d_(img):
    # Normalization
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)

    img = img.mean(dim=2)  # z_proj
    img = img.detach().cpu()
    img = img.permute(0, 2, 3, 1) * 255
    return img[0].numpy()


def save_img(tensor_input, filename):
    if len(tensor_input.shape) == 4:
        np_img = tensor2numpy_2d_(tensor_input)
    elif len(tensor_input.shape) == 5:
        np_img = tensor2numpy_3d_(tensor_input)
    else:
        raise RuntimeError("To save an image, the tensor shape should be 4 or 5")

    cv2.imwrite(filename, cv2.flip(np_img, 0))


def LoadSingleMesh(
    obj_path: str, device: torch.device = torch.device("cpu"), dtype=torch.float32
) -> List[torch.Tensor]:
    """Load mesh data from disk
    We will typically load mesh from json file where,
        'verts'--------[NumOfVertices, 3]
        'faces'--------[NumOfFaces, 3]
        'edges'--------[NumOfEdges, 2, 3]
        'verts_rgba'---[1, NumofVertices, 4]
        'materials'----{NumOfFaces->str}
        'visible'------{NumOfFaces->bool}
    are given.
    """
    dim = 3
    if not os.path.exists(obj_path):
        raise RuntimeError(
            f"Mesh object path :{obj_path} not found, check your DataSet"
        )

    path_str = str(obj_path)

    if ".obj" in path_str:
        scene = pywavefront.Wavefront(obj_path, collect_faces=True)

        new_v = torch.Tensor(scene.vertices).to(device).to(dtype)
        new_f = torch.Tensor(scene.meshes[None].faces).to(device).to(torch.int32)

        new_v = new_v.reshape(-1, dim)
        new_f = new_f.reshape(-1, dim)
    elif ".json" in path_str:
        with open(obj_path, "r") as load_f:
            json_data = json.load(load_f)
            new_v = torch.Tensor(json_data["verts"]).to(device).to(dtype)
            new_f = torch.Tensor(json_data["faces"]).to(device).to(torch.int32)
            new_v = new_v.reshape(-1, dim)
            new_f = new_f.reshape(-1, dim)
            # new_e = torch.Tensor(json_data["edges"]).to(device).to(dtype)
            # new_vcolor = (
            #     torch.Tensor(json_data["verts_rgba"]).to(device).to(dtype)
            # )
        load_f.close()
    elif ".ply" in path_str:
        ply_data = o3d.io.read_triangle_mesh(obj_path)
        new_v = np.asarray(ply_data.vertices)
        new_f = np.asarray(ply_data.triangles)

        new_v = new_v.reshape(-1, dim)
        new_f = new_f.reshape(-1, dim)

        new_v = torch.from_numpy(new_v).to(device).to(dtype)
        new_f = torch.from_numpy(new_f).to(device).to(torch.int32)
    elif ".dae" in path_str:
        dae_data = trimesh.load_mesh(obj_path)
        if type(dae_data) == trimesh.Scene:
            new_v = []
            new_f = []
            for geom_name in dae_data.geometry.keys():
                new_v_tmp = np.asarray(dae_data.geometry[geom_name].vertices)
                new_f_tmp = np.asarray(dae_data.geometry[geom_name].faces)

                new_v_tmp = new_v_tmp.reshape(-1, dim)
                new_f_tmp = new_f_tmp.reshape(-1, dim)

                new_v.append(torch.from_numpy(new_v_tmp).to(device).to(dtype))
                new_f.append(torch.from_numpy(new_f_tmp).to(device).to(torch.int32))
        elif type(dae_data) == trimesh.Trimesh:
            new_v = np.asarray(dae_data.vertices)
            new_f = np.asarray(dae_data.faces)

            new_v = new_v.reshape(-1, dim)
            new_f = new_f.reshape(-1, dim)

            new_v = torch.from_numpy(new_v).to(device).to(dtype)
            new_f = torch.from_numpy(new_f).to(device).to(torch.int32)
        else:
            raise NotImplementedError(
                "trimesh read geom file could return either trimesh.Trimesh or trimesh.Scene"
            )

    return [new_v, new_f]


def LoadMeshes(
    asset_path: str,
    with_primitives: bool,
    obj_folder: str = "objs",
    texture_system: Dict[str, str] = None,
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> List[List[torch.Tensor]]:
    """Load mesh data from disk
    We will typically load mesh from json file where,
        'verts'--------[NumOfVertices, 3]
        'faces'--------[NumOfFaces, 3]
        'edges'--------[NumOfEdges, 2, 3]
        'verts_rgba'---[1, NumofVertices, 4]
        'materials'----{NumOfFaces->str}
        'visible'------{NumOfFaces->bool}
    are given.
    """
    load_texture = False
    obj_dir = os.path.join(asset_path, obj_folder)
    texture_dir = os.path.join(asset_path, "textures")
    if not os.path.exists(obj_dir):
        raise RuntimeError(f"Mesh object path :{obj_dir} not found, check your DataSet")

    if texture_system is not None and os.path.exists(texture_dir):
        load_texture = True

    data_files = os.listdir(obj_dir)
    data_files = sorted(data_files)

    vertices = []
    faces = []
    vert_uvs = []
    face_uvs = []
    textures = []
    primitive_names = []
    for filename in data_files:
        obj_path = os.path.join(obj_dir, filename)
        new_v, new_f = LoadSingleMesh(obj_path=obj_path, device=device, dtype=dtype)

        new_texture = None
        if load_texture:
            texture_filename = texture_system[filename]
            texture_path = os.path.join(texture_dir, texture_filename)
            taichi_texture = TaichiTexture(filename=texture_filename, image=None)
            new_texture = taichi_texture.LoadImage(
                filepath=texture_path, device=device, dtype=dtype
            )

        if type(new_v) == list and type(new_f) == list:
            assert len(new_v) == len(new_f)
            for i in range(len(new_v)):
                vertices.append(new_v[i])
                faces.append(new_f[i])
                textures.append(new_texture)
        else:
            vertices.append(new_v)
            faces.append(new_f)
            textures.append(new_texture)

        primitive_names.append(filename)

    if not with_primitives:
        vert_offset = 0
        for i in range(1, len(vertices)):
            vert_offset = vert_offset + vertices[i - 1].shape[-2]
            faces[i] = faces[i] + vert_offset

        vertices = [torch.cat(vertices, dim=-2)]
        faces = [torch.cat(faces, dim=-2)]
        primitive_names = ["default"]

    if not load_texture:
        vert_uvs = None
        face_uvs = None
        textures = None
    else:
        vert_uvs, face_uvs = TaichiTexture.GenerateUVs(verts=vertices, faces=faces)

    return [vertices, faces, vert_uvs, face_uvs, textures, primitive_names]


def LoadPointCloudFromMesh(meshes: Meshes, num_pts_samples: int) -> torch.Tensor:
    point_clouds, normals = sample_points_from_meshes(
        meshes, num_samples=num_pts_samples, return_normals=True
    )  # [F, NumOfSamples, 3]
    return point_clouds, normals


def DumpCFGFile(
    save_path: str,
    save_name: str,
    point_clouds: torch.Tensor,
    with_floor: bool = True,
    light_probe: torch.Tensor = None,
):
    if len(point_clouds.shape) == 2:
        point_clouds = point_clouds.reshape(1, -1, 3)
    elif len(point_clouds.shape) != 3:
        raise ValueError(f"point clouds should in shape [(B), N, dim=3]")

    batch_size = point_clouds.shape[0]
    mass = 1.0
    floor_height = point_clouds[..., 2].min()
    pts_types = [chr(ord("A") + i) + "V" for i in range(26)]
    total_pts = point_clouds.shape[0] * point_clouds.shape[1]
    if light_probe is not None:
        assert len(light_probe.shape) == 2
        total_pts += light_probe.shape[0]

    with open(os.path.join(save_path, f"{save_name}.cfg"), "w+") as fp:
        fp.write(f"Number of particles = {total_pts}\n")
        fp.write("A = 1 Angstrom (basic length-scale)\n")
        # x-axis
        fp.write(f"H0(1,1) = {1} A\n")
        fp.write(f"H0(1,2) = {0} A\n")
        fp.write(f"H0(1,3) = {0} A\n")
        # y-axis
        fp.write(f"H0(2,1) = {0} A\n")
        fp.write(f"H0(2,2) = {1} A\n")
        fp.write(f"H0(2,3) = {0} A\n")
        # z-axis
        fp.write(f"H0(3,1) = {0} A\n")
        fp.write(f"H0(3,2) = {0} A\n")
        fp.write(f"H0(3,3) = {1} A\n")

        fp.write(".NO_VELOCITY.\n")
        fp.write(f"entry_count = {3}\n")

        for batch_idx in range(batch_size):
            pts_i = point_clouds[batch_idx]
            n_pts = pts_i.shape[0]
            for i in range(n_pts):
                particle_type = pts_types[batch_idx % len(pts_types)]
                if with_floor and pts_i[i, 2] <= floor_height + 0.01:
                    # This is floor particle
                    particle_type = "FL"
                fp.write(
                    f"{mass}\n{particle_type}\n{pts_i[i, 0].item()} {pts_i[i, 1].item()} {pts_i[i, 2].item()}\n"
                )

        if light_probe is not None:
            for i in range(light_probe.shape[0]):
                particle_type = "LP"
                fp.write(
                    f"{mass}\n{particle_type}\n{light_probe[i, 0].item()} {light_probe[i, 1].item()} {light_probe[i, 2].item()}\n"
                )

    fp.close()


def create_2d_meshgrid_tensor(
    size: List[int],
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    [batch, _, height, width] = size
    y_pos, x_pos = torch.meshgrid(
        [
            torch.arange(0, height, device=device, dtype=dtype),
            torch.arange(0, width, device=device, dtype=dtype),
        ]
    )
    mgrid = torch.stack([x_pos, y_pos], dim=0)  # [C, H, W]
    mgrid = mgrid.unsqueeze(0)  # [B, C, H, W]
    mgrid = mgrid.repeat(batch, 1, 1, 1)
    return mgrid


def create_3d_meshgrid_tensor(
    size: List[int],
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    [batch, _, depth, height, width] = size
    z_pos, y_pos, x_pos = torch.meshgrid(
        [
            torch.arange(0, depth, device=device, dtype=dtype),
            torch.arange(0, height, device=device, dtype=dtype),
            torch.arange(0, width, device=device, dtype=dtype),
        ]
    )

    mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)  # [C, D, H, W]
    mgrid = mgrid.unsqueeze(0)  # [B, C, D, H, W]
    mgrid = mgrid.repeat(batch, 1, 1, 1, 1)
    return mgrid


def dump_tecplot_file(filepath: str, grids: Dict[str, torch.Tensor]):
    tp.session.connect()

    # Usually, the density file is forced.
    batch_idx = 0
    rho = grids["rho"]
    if rho.dim() == 4:
        dim = 2
        batch_size, _, H, W = rho.shape
        meshgrid = create_2d_meshgrid_tensor(
            size=rho.shape, device=rho.device, dtype=torch.int32
        )
        variables = ["X", "Y"]
    elif rho.dim() == 5:
        dim = 3
        batch_size, _, D, H, W = rho.shape
        meshgrid = create_3d_meshgrid_tensor(
            size=rho.shape, device=rho.device, dtype=torch.int32
        )
        variables = ["X", "Y", "Z"]

    for key in grids:
        if grids[key].shape[1] == 1:
            variables.append(key)
        elif grids[key].shape[1] == dim:
            variables.append(f"{key}x")
            variables.append(f"{key}y")
            if dim == 3:
                variables.append(f"{key}z")

    with tp.session.suspend():
        # Create the dataset
        ds = tp.active_frame().create_dataset("Data", variables)

        np_x = meshgrid[batch_idx, 0].cpu().numpy()
        ds.X[:] = np_x.ravel(order="F")
        np_y = meshgrid[batch_idx, 1].cpu().numpy()
        ds.Y[:] = np_y.ravel(order="F")
        if dim == 3:
            np_z = meshgrid[batch_idx, 2].cpu().numpy()
            ds.Z[:] = np_z.ravel(order="F")

        for key in grids:
            if grids[key].shape[1] == 1:
                variables.append(key)

                np_grid = grids[key][batch_idx, 0].cpu().numpy()
                ds[key] = np_grid.ravel(order="F")
            elif grids[key].shape[1] == dim:
                np_grid = grids[key][batch_idx, :].cpu().numpy()

                ds[f"{key}x"] = np_grid[0].ravel(order="F")
                ds[f"{key}y"] = np_grid[1].ravel(order="F")
                if dim == 3:
                    ds[f"{key}z"] = np_grid[1].ravel(order="F")

    # Save as binary .plt file
    tp.data.save_tecplot_plt(filepath, dataset=ds)


def LoadSDFFromMeshPath(meshpath: str, size: int) -> np.ndarray:
    mesh_scale = 0.5
    level = 2 / size
    mesh = trimesh.load(meshpath, force="mesh")
    # normalize mesh
    vertices = mesh.vertices
    # fix mesh
    obs_sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True
    )
    obs_sdf = obs_sdf.reshape(1, 1, *obs_sdf.shape)
    return obs_sdf
