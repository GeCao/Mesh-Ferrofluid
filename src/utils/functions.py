import os
import numpy as np
import math
import pymeshlab as ml
import json
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
import matplotlib.pyplot as plt
from typing import Tuple, List, Union


def PostProcessFeatures(
    feat: torch.Tensor,
    scaled_pts: torch.Tensor,
    K_closest_mask: Tuple[List[int]],
    K_closest: int,
    feature_extractor,
    img_resolution=128,
    feature_resolution=16,
):
    """Post Procecss of encode features
    Args:
        feat (torch.Tensor): Encode Features with shape = [B, maps_per_Batch, n_features, feat_height, feat_width]
        scaled_pts (torch.Tensor): points coordinates with shape = [B, n_pts, 3], clamped to (-1, 1)
        K_closest_mask (Tuple of 2 list filled with int): scaled_pts[K_closest_mask] = [B*n_rays*K_closest, 3]
    Returns:
        torch.Tensor: pass
    """
    device = feat.device

    n_batch, maps_per_batch, n_features, feat_heigth, feat_width = feat.shape
    n_feat_maps = maps_per_batch * n_batch
    feat2img_f = img_resolution // feature_resolution
    feat = feat.reshape(-1, n_features, feat_heigth, feat_width)

    batch_size, n_pts, dim = scaled_pts.shape
    K_closest_scaled_pts = scaled_pts[K_closest_mask].reshape(batch_size, -1, 3)

    # TODO: Get coordinates in the feautre maps for each point
    coordinates, coord_x, coord_y, depth = feature_extractor._get_img_coord(
        K_closest_scaled_pts, resolution=img_resolution
    )
    # Adjust for downscaled feature maps
    coord_x = (
        torch.round(coord_x.view(n_feat_maps, -1, K_closest) / feat2img_f)
        .to(torch.long)
        .to(device)
    )
    coord_x = torch.minimum(
        coord_x.to(device), torch.tensor([feat_heigth - 1]).to(device)
    )
    coord_y = torch.round(
        coord_y.view(n_feat_maps, -1, K_closest).to(device) / feat2img_f
    ).to(torch.long)
    coord_y = torch.minimum(coord_y, torch.tensor([feat_width - 1]).to(device))
    feat = feat.permute(0, 2, 3, 1)
    pts_feat = torch.stack(
        [feat[i][tuple([coord_x[i], coord_y[i]])] for i in range(n_feat_maps)]
    )
    pts_feat = pts_feat.reshape(n_batch, maps_per_batch, -1, K_closest, n_features)
    pts_feat = pts_feat.permute(0, 2, 3, 1, 4)

    return pts_feat


def _export_ply(
    save_path: str, verts: np.ndarray, faces: np.ndarray, vert_color: np.ndarray
):
    mesh = ml.MeshSet()
    mesh_c = ml.Mesh(
        vertex_matrix=verts,
        face_matrix=faces,
        v_color_matrix=vert_color,  # key name as "Col"
    )
    mesh.add_mesh(mesh_c)
    mesh.save_current_mesh(save_path)


def _export_obj(save_path: str, vertices: np.ndarray, faces: np.ndarray):
    np_faces = faces.reshape(-1, 3)
    np_vertices = vertices.reshape(-1, vertices.shape[-1])
    if np_faces.min() == 0:
        np_faces = np_faces + 1
    with open(save_path, "w") as f:
        f.write("# OBJ file\n")
        for i in range(np_vertices.shape[0]):
            if np_vertices.shape[-1] >= 6:
                f.write(
                    "v {} {} {} {} {} {}\n".format(
                        np_vertices[i, 0],
                        np_vertices[i, 1],
                        np_vertices[i, 2],
                        np_vertices[i, 3],
                        np_vertices[i, 4],
                        np_vertices[i, 5],
                    )
                )
            else:
                f.write(
                    "v {} {} {}\n".format(
                        np_vertices[i, 0], np_vertices[i, 1], np_vertices[i, 2]
                    )
                )
        for j in range(np_faces.shape[0]):
            f.write(
                "f {} {} {}\n".format(np_faces[j, 0], np_faces[j, 1], np_faces[j, 2])
            )
    f.close()


def export_asset(
    save_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    vert_color: np.ndarray = None,
):
    if ".obj" in str(save_path):
        _export_obj(save_path=save_path, vertices=vertices, faces=faces)
    elif ".ply" in str(save_path):
        _export_ply(
            save_path=save_path, verts=vertices, faces=faces, vert_color=vert_color
        )
    else:
        raise NotImplementedError("Currently no support for your file-format")


def gkern(
    kernlen=21, nsig=7, device: torch.device = torch.device("cpu"), dtype=torch.float32
):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    res = kern2d / kern2d.sum()
    return torch.from_numpy(res).to(device).to(dtype)


def ApplyPositionBasedKernel(
    grid: torch.Tensor, kernel: torch.Tensor, grid_pos: torch.Tensor, gain: torch.Tensor
) -> torch.Tensor:
    assert gain.numel() * 2 == grid_pos.numel()
    gain = gain.flatten()
    assert grid_pos.shape[-1] == 2 and len(grid_pos.shape) == 2
    kernel_size_sqr = kernel.numel()
    kernel_size = int(math.sqrt(kernel_size_sqr))
    assert kernel_size % 2 == 1 and kernel_size * kernel_size == kernel_size_sqr
    r = (kernel_size - 1) // 2

    _, _, H, W = grid.shape
    device = grid.device
    dtype = grid.dtype

    kernel = kernel.reshape(1, 1, kernel_size, kernel_size)
    index = (
        grid_pos[:, 1].cpu().tolist(),
        grid_pos[:, 0].cpu().tolist(),
    )  # (y-idx, x-idx)
    grid = grid.reshape(1, 1, H, W)
    grid_pad = F.pad(grid, pad=(r, r, r, r), mode="replicate")
    grid_pad = grid_pad.reshape(*grid_pad.shape[-2:])
    for i in range(kernel_size):
        for j in range(kernel_size):
            index_ij = ([ele + i for ele in index[0]], [ele + j for ele in index[1]])
            grid_pad[index_ij] += gain * kernel[0, 0, i, j]
    grid_pad = grid_pad.reshape(1, 1, *grid_pad.shape[-2:])
    # output = torch.conv2d(input=grid_pad, weight=kernel)
    output = grid_pad[:, :, r:-r, r:-r]
    return output


def GetAABB(x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if type(x) == list:
        assert len(x) > 0
        device = x[0].device
        dtype = x[0].dtype
        dim = x[0].shape[-1]
        AABB = torch.zeros((2, dim), dtype=dtype, device=device)
        AABB[0] = x[0].view(-1, dim).min(dim=0)[0]
        AABB[1] = x[0].view(-1, dim).max(dim=0)[0]
        for i in range(1, len(x)):
            AABB_min_i = x[i].view(-1, dim).min(dim=0)[0]
            AABB_max_i = x[i].view(-1, dim).max(dim=0)[0]

            AABB[0] = torch.minimum(AABB[0], AABB_min_i)
            AABB[1] = torch.maximum(AABB[1], AABB_max_i)
    else:
        device = x.device
        dtype = x.dtype
        dim = x.shape[-1]
        AABB = torch.zeros((2, dim), dtype=dtype, device=device)
        AABB[0] = x.view(-1, dim).min(dim=0)[0]
        AABB[1] = x.view(-1, dim).max(dim=0)[0]

    return AABB


def ScaleAABB(
    x: Union[List[torch.Tensor], torch.Tensor],
    AABB: torch.Tensor,
    take_neg: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    if AABB is None:
        return x

    dim = 3
    assert AABB.dim() == 2
    assert AABB.shape[-2] == 2
    assert AABB.shape[-1] == dim

    # AABB_len = AABB[..., 1, :] - AABB[..., 0, :]  # [dim,]
    AABB_min = AABB[0, :]
    AABB_max = AABB[1, :]
    AABB_center = (AABB_min + AABB_max) / 2.0
    AABB_len, scale_dim = (AABB_max - AABB_min).max(dim=-1)

    scale = 2.0 / AABB_len if take_neg else 1.0 / AABB_len  # [-1, 1]
    center = AABB_center if take_neg else AABB_min  # [0, 1]

    center = center.to(x.device)
    scale = scale.to(x.device)
    if type(x) == list:
        for i in range(len(x)):
            x[i] = (x[i] - center) * scale
    else:
        x = (x - center) * scale

    return x


def SampleSkydomRays(
    n_rays: int, device: torch.device = torch.device("cpu"), dtype=torch.float32
) -> List[torch.Tensor]:
    """Sampling rays in a spherical rasterizer"""
    n_theta = int(math.sqrt(n_rays / 2))  # 180 degree
    assert 2 * n_theta * n_theta == n_rays
    n_phi = n_theta * 2  # 360 degree

    # thetas
    thetas = torch.linspace(0, math.pi, n_theta + 1, device=device, dtype=dtype)
    thetas = thetas[:-1]
    theta_interval = float((thetas[1] - thetas[0]).cpu().item())
    thetas = thetas + theta_interval / 2.0  # + 0.5
    thetas = thetas[:, None].repeat(1, n_phi)
    thetas = thetas.flatten()

    # phis
    phis = torch.linspace(0, 2.0 * math.pi, n_phi + 1, device=device, dtype=dtype)
    phis = phis[:-1]
    phi_interval = float((phis[1] - phis[0]).cpu().item())
    phis = phis + phi_interval / 2.0
    phis = phis[None, :].repeat(n_theta, 1)
    phis = phis.flatten()

    # Rectangular coordinates
    x = torch.sin(thetas) * torch.cos(phis)
    y = torch.sin(thetas) * torch.sin(phis)
    z = torch.cos(thetas)
    ray_d = torch.stack((x, y, z), dim=-1)  # [n_rays, 3]

    # Normalization
    ray_d = F.normalize(ray_d, dim=-1)

    return [ray_d, thetas, phis, theta_interval, phi_interval]


def SampleIcosahedralRays(
    n_rays: int, device: torch.device = torch.device("cpu"), dtype=torch.float32
) -> List[torch.Tensor]:
    """Sampling rays in an icosahedral spherical rasterizer"""
    if n_rays % 20 != 0:
        raise RuntimeError(
            "For icosahedral rasterizer, n_rays % 20 == 0 should always be ensured"
        )

    nray_per_faces = n_rays // 20
    n_theta = int(math.sqrt(nray_per_faces))
    if nray_per_faces != n_theta * n_theta:
        raise RuntimeError("n_rays should be the form of := 20 * pow(x, 2)")

    thetas = []
    phis = []
    # The 1st layer: 0, 1, 2, 3, 4
    face_verts = []
    for k in range(5):
        theta_expand = math.pi / 3.0
        phi_expand = 2 * math.pi / 5
        theta_offset = 0.0
        phi_offset = phi_expand * (k + 0.5)
        theta_k = []
        phi_k = []
        face_verts_k = []
        for col in range(n_theta):
            for row in range(col * 2 + 1):
                col_frac = (col + 0.5) / n_theta
                theta_k.append(theta_offset + theta_expand * col_frac)
                row_frac = (row + 0.5) / (col * 2 + 1)
                phi_k.append(phi_offset + phi_expand * row_frac)

                if row % 2 == 0:
                    theta1 = theta_offset + theta_expand * (col / n_theta)
                    theta2 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    theta3 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    phi1 = phi_offset
                    if col != 0:
                        phi1 = phi1 + phi_expand * (row / (col * 2))
                    phi2 = phi_offset + phi_expand * (row / (col * 2 + 2))
                    phi3 = phi_offset + phi_expand * ((row + 2) / (col * 2 + 2))
                else:
                    theta1 = theta_offset + theta_expand * (col / n_theta)
                    theta2 = theta_offset + theta_expand * (col / n_theta)
                    theta3 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    phi1 = phi_offset
                    phi2 = phi_offset
                    if col != 0:
                        phi1 = phi1 + phi_expand * ((row - 1) / (col * 2))
                        phi2 = phi2 + phi_expand * ((row + 1) / (col * 2))
                    phi3 = phi_offset + phi_expand * ((row + 1) / (col * 2 + 2))
                x1 = [
                    math.cos(phi1) * math.sin(theta1),
                    math.sin(phi1) * math.sin(theta1),
                    math.cos(theta1),
                ]
                x2 = [
                    math.cos(phi2) * math.sin(theta2),
                    math.sin(phi2) * math.sin(theta2),
                    math.cos(theta2),
                ]
                x3 = [
                    math.cos(phi3) * math.sin(theta3),
                    math.sin(phi3) * math.sin(theta3),
                    math.cos(theta3),
                ]
                face_verts_k.append([x1, x2, x3])
        theta_k = torch.Tensor(theta_k).to(device).to(dtype)
        phi_k = torch.Tensor(phi_k).to(device).to(dtype)
        face_verts_k = torch.Tensor(face_verts_k).to(device).to(dtype)
        thetas.append(theta_k)
        phis.append(phi_k)
        face_verts.append(face_verts_k)

    # The 2nd layer: 5, 6, 7, 8, 9
    for k in range(5):
        theta_expand = -math.pi / 3.0
        theta_offset = 2 * math.pi / 3.0
        theta_k = []
        phi_k = []
        face_verts_k = []
        for col in range(n_theta):
            phi_extent = (2 * math.pi / 5) * (col + 0.5) / n_theta / 2
            phi_offset = (2 * math.pi / 5) * (k + 1)
            phi_extent_lower = 2 * math.pi / 5 * col / n_theta / 2
            phi_extent_upper = 2 * math.pi / 5 * (col + 1) / n_theta / 2
            for row in range(-col, col + 1):
                col_frac = (col + 0.5) / n_theta
                theta_k.append(theta_offset + theta_expand * col_frac)
                row_frac = row / (col + 0.5) if col != 0 else 0
                phi_k.append(phi_offset + phi_extent * row_frac)

                if (row + col) % 2 == 0:
                    theta1 = theta_offset + theta_expand * (col / n_theta)
                    theta2 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    theta3 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    phi1 = phi_offset
                    if col != 0:
                        phi1 = phi1 + phi_extent_lower * (row / (col))
                    phi2 = phi_offset + phi_extent_upper * ((row + 1) / (col + 1))
                    phi3 = phi_offset + phi_extent_upper * ((row - 1) / (col + 1))
                else:
                    theta1 = theta_offset + theta_expand * (col / n_theta)
                    theta2 = theta_offset + theta_expand * (col / n_theta)
                    theta3 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    phi1 = phi_offset
                    phi2 = phi_offset
                    if col != 0:
                        phi1 = phi1 + phi_extent_lower * ((row - 1) / (col))
                        phi2 = phi2 + phi_extent_lower * ((row + 1) / (col))
                    phi3 = phi_offset + phi_extent_upper * ((row) / (col + 1))
                x1 = [
                    math.cos(phi1) * math.sin(theta1),
                    math.sin(phi1) * math.sin(theta1),
                    math.cos(theta1),
                ]
                x2 = [
                    math.cos(phi2) * math.sin(theta2),
                    math.sin(phi2) * math.sin(theta2),
                    math.cos(theta2),
                ]
                x3 = [
                    math.cos(phi3) * math.sin(theta3),
                    math.sin(phi3) * math.sin(theta3),
                    math.cos(theta3),
                ]
                face_verts_k.append([x1, x2, x3])
        theta_k = torch.Tensor(theta_k).to(device).to(dtype)
        phi_k = torch.Tensor(phi_k).to(device).to(dtype)
        face_verts_k = torch.Tensor(face_verts_k).to(device).to(dtype)
        thetas.append(theta_k)
        phis.append(phi_k)
        face_verts.append(face_verts_k)

    # The 2nd layer: 10, 11, 12, 13, 14
    for k in range(5):
        theta_expand = math.pi / 3.0
        theta_offset = math.pi / 3.0
        theta_k = []
        phi_k = []
        face_verts_k = []
        for col in range(n_theta):
            phi_extent = (2 * math.pi / 5) * (col + 0.5) / n_theta / 2
            phi_offset = (2 * math.pi / 5) * (k + 0.5)
            phi_extent_lower = 2 * math.pi / 5 * col / n_theta / 2
            phi_extent_upper = 2 * math.pi / 5 * (col + 1) / n_theta / 2
            for row in range(-col, col + 1):
                col_frac = (col + 0.5) / n_theta
                theta_k.append(theta_offset + theta_expand * col_frac)
                row_frac = row / (col + 0.5) if col != 0 else 0
                phi_k.append(phi_offset + phi_extent * row_frac)

                if (row + col) % 2 == 0:
                    theta1 = theta_offset + theta_expand * (col / n_theta)
                    theta2 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    theta3 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    phi1 = phi_offset
                    if col != 0:
                        phi1 = phi1 + phi_extent_lower * (row / (col))
                    phi2 = phi_offset + phi_extent_upper * ((row + 1) / (col + 1))
                    phi3 = phi_offset + phi_extent_upper * ((row - 1) / (col + 1))
                else:
                    theta1 = theta_offset + theta_expand * (col / n_theta)
                    theta2 = theta_offset + theta_expand * (col / n_theta)
                    theta3 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    phi1 = phi_offset
                    phi2 = phi_offset
                    if col != 0:
                        phi1 = phi1 + phi_extent_lower * ((row - 1) / (col))
                        phi2 = phi2 + phi_extent_lower * ((row + 1) / (col))
                    phi3 = phi_offset + phi_extent_upper * ((row) / (col + 1))
                x1 = [
                    math.cos(phi1) * math.sin(theta1),
                    math.sin(phi1) * math.sin(theta1),
                    math.cos(theta1),
                ]
                x2 = [
                    math.cos(phi2) * math.sin(theta2),
                    math.sin(phi2) * math.sin(theta2),
                    math.cos(theta2),
                ]
                x3 = [
                    math.cos(phi3) * math.sin(theta3),
                    math.sin(phi3) * math.sin(theta3),
                    math.cos(theta3),
                ]
                face_verts_k.append([x1, x2, x3])
        theta_k = torch.Tensor(theta_k).to(device).to(dtype)
        phi_k = torch.Tensor(phi_k).to(device).to(dtype)
        face_verts_k = torch.Tensor(face_verts_k).to(device).to(dtype)
        thetas.append(theta_k)
        phis.append(phi_k)
        face_verts.append(face_verts_k)

    # The 3rd layer: 15, 16, 17, 18, 19
    for k in range(5):
        theta_expand = -(math.pi / 3.0)
        phi_expand = 2 * math.pi / 5
        theta_offset = math.pi
        phi_offset = phi_expand * k
        theta_k = []
        phi_k = []
        face_verts_k = []
        for col in range(n_theta):
            for row in range(col * 2 + 1):
                col_frac = (col + 0.5) / n_theta
                theta_k.append(theta_offset + theta_expand * col_frac)
                row_frac = (row + 0.5) / (col * 2 + 1)
                phi_k.append(phi_offset + phi_expand * row_frac)

                if row % 2 == 0:
                    theta1 = theta_offset + theta_expand * (col / n_theta)
                    theta2 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    theta3 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    phi1 = phi_offset
                    if col != 0:
                        phi1 = phi1 + phi_expand * (row / (col * 2))
                    phi2 = phi_offset + phi_expand * (row / (col * 2 + 2))
                    phi3 = phi_offset + phi_expand * ((row + 2) / (col * 2 + 2))
                else:
                    theta1 = theta_offset + theta_expand * (col / n_theta)
                    theta2 = theta_offset + theta_expand * (col / n_theta)
                    theta3 = theta_offset + theta_expand * ((col + 1) / n_theta)
                    phi1 = phi_offset
                    phi2 = phi_offset
                    if col != 0:
                        phi1 = phi1 + phi_expand * ((row - 1) / (col * 2))
                        phi2 = phi2 + phi_expand * ((row + 1) / (col * 2))
                    phi3 = phi_offset + phi_expand * ((row + 1) / (col * 2 + 2))
                x1 = [
                    math.cos(phi1) * math.sin(theta1),
                    math.sin(phi1) * math.sin(theta1),
                    math.cos(theta1),
                ]
                x2 = [
                    math.cos(phi2) * math.sin(theta2),
                    math.sin(phi2) * math.sin(theta2),
                    math.cos(theta2),
                ]
                x3 = [
                    math.cos(phi3) * math.sin(theta3),
                    math.sin(phi3) * math.sin(theta3),
                    math.cos(theta3),
                ]
                face_verts_k.append([x1, x2, x3])
        theta_k = torch.Tensor(theta_k).to(device).to(dtype)
        phi_k = torch.Tensor(phi_k).to(device).to(dtype)
        face_verts_k = torch.Tensor(face_verts_k).to(device).to(dtype)
        thetas.append(theta_k)
        phis.append(phi_k)
        face_verts.append(face_verts_k)

    thetas = torch.stack(thetas, dim=0).flatten()  # [n_tri * n_theta^2]
    phis = torch.stack(phis, dim=0).flatten()  # [n_tri * n_theta^2]
    phis = torch.where(phis >= 2 * math.pi, phis - 2 * math.pi, phis)
    phis = torch.where(phis < 0, phis + 2 * math.pi, phis)
    face_verts = torch.stack(face_verts, dim=0)  # [n_tri, 3, dim=3]

    # Rectangular coordinates
    x = torch.sin(thetas) * torch.cos(phis)
    y = torch.sin(thetas) * torch.sin(phis)
    z = torch.cos(thetas)
    ray_d = torch.stack((x, y, z), dim=-1)  # [n_rays, 3]

    # Normalization
    ray_d = F.normalize(ray_d, dim=-1)

    # from src.photon_splatting.utils import LoadSingleMesh

    # disk_verts, disk_faces = LoadSingleMesh(
    #     obj_path=f"../assets/cube.obj", device=device, dtype=dtype
    # )

    # photon_verts = []
    # photon_faces = []
    # for batch_idx in range(n_rays):
    #     pos = ray_d[batch_idx][None]

    #     new_verts = disk_verts * 0.02 + pos

    #     new_faces = disk_faces

    #     photon_verts.append(new_verts)
    #     photon_faces.append(new_faces)

    # vert_offset = 0
    # for i in range(1, len(photon_verts)):
    #     vert_offset = vert_offset + photon_verts[i - 1].shape[-2]
    #     photon_faces[i] = photon_faces[i] + vert_offset

    # photon_verts = torch.cat(photon_verts, dim=-2).cpu().numpy()
    # photon_faces = torch.cat(photon_faces, dim=-2).cpu().numpy()
    # export_asset(
    #     save_path=f"../assets/icosahedral_rayd.obj",
    #     vertices=photon_verts,
    #     faces=photon_faces,
    # )
    # exit(0)

    return [ray_d, thetas, phis, face_verts]


def SampleImportanceRays(
    n_rays: int, device: torch.device = torch.device("cpu"), dtype=torch.float32
) -> List[torch.Tensor]:
    sampling_H = int(math.sqrt(n_rays))
    sampling_W = sampling_H
    assert sampling_H * sampling_W == n_rays

    r1 = torch.linspace(0, sampling_W - 1, sampling_W, device=device, dtype=dtype)
    r2 = torch.linspace(0, sampling_W - 1, sampling_W, device=device, dtype=dtype)
    r1 = (r1 + 0.5) / sampling_W
    r2 = (r2 + 0.5) / sampling_H
    r2, r1 = torch.meshgrid([r2, r1])  # [H, W]
    ray_d = squareToUniformSphere(r1=r1, r2=r2)
    ray_d = ray_d.reshape(n_rays, 3)
    thetas, phis = GetThetaAndPhiFromRayDirection(ray_d=ray_d, keepdim=False)

    return [ray_d, thetas, phis, r1, r2]


def squareToUniformCylinder(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    fai = 2.0 * math.pi * r2
    res = [torch.cos(fai), torch.sin(fai), 2 * r1 - 1]
    res = torch.stack(res, dim=-1)  # [..., 3]
    return res


def squareToUniformSphere(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    Cylinder_res = squareToUniformCylinder(r1, r2)
    r = torch.sqrt(1 - Cylinder_res[..., 2] * Cylinder_res[..., 2])
    res = [r * Cylinder_res[..., 0], r * Cylinder_res[..., 1], Cylinder_res[..., 2]]
    res = torch.stack(res, dim=-1)  # [..., 3]
    return res


def squareToUniformHemisphere(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    Cylinder_res = squareToUniformCylinder(r1, r2)
    z = Cylinder_res[..., 2] * 0.5 + 0.5
    r = torch.sqrt(1 - z * z)
    res = [r * Cylinder_res[..., 0], r * Cylinder_res[..., 1], -z]
    res = torch.stack(res, dim=-1)  # [..., 3]
    return res


def CalculateSkyboxViewPerspectiveMatrices(
    skybox_eyes: torch.Tensor, near: float = 0.1, far: float = 2000
) -> List[torch.Tensor]:
    """What skybox do you expect from here?
                         ^
                   5.(z) |    > 3.(y)
                  _______|___/_____
                 /|      |  /     /|
                / |      . /     / |
               /__|_______/_____/  |
    0.(-x)<----|-.|      .      | .|---------> 2.(x)
               |  |_____/|______|__|
               | /     / |      | /
               |/_____/__|______|/
                     /   |
             1.(-y) <    V 4.(-z)
    What does this skybox map looks like?
             _________
             |        |
             | 5.(z)  | ^
             |        | | up=(y)
    _________|________|_________________
    |        |        |        |        |
    | 0.(-x) | 1.(-y) | 2.(x)  | 3.(y)  |  ^
    |        |        |        |        |  | up=(z)
    |________|________|________|________|
             |        |
             | 4.(-z) | ^
             |        | | up=(-y)
             |________|
    To calcluate view matrix:
    Imagine you walk to the position of camera (translate),
    Then you rotate your head to the lookat position (rotation).
    view_matrix = [right_x,   right_y,   right_z,   -right * eye]
                  [up_x,      up_y,      up_z,      -up    * eye]
                  [-lookat_x, -lookat_y, -lookat_z, lookat * eye]
                  [0,         0,         0,         1           ]
    To calculate perspective matrix:
    """
    batch_shape = [1] * len(skybox_eyes.shape[:-1])  # [*1,]
    device = skybox_eyes.device
    dtype = skybox_eyes.dtype

    # -x, -y, x, y, -z, z
    skybox_lookats = [
        [-1, 0, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ]
    skybox_lookats = torch.Tensor(skybox_lookats).to(dtype).to(device)
    skybox_lookats = skybox_lookats.reshape(*batch_shape, 6, 3)  # [*1, 6, 3]

    # z, z, z, z, -y, y
    skybox_ups = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, -1, 0], [0, 1, 0]]
    skybox_ups = torch.Tensor(skybox_ups).to(dtype).to(device)
    skybox_ups = skybox_ups.reshape(*batch_shape, 6, 3)  # [*1, 6, 3]

    # y, -x, -y, x, -x, -x
    skybox_rights = torch.cross(skybox_lookats, skybox_ups, dim=-1)  # [*1, 6, 3]

    # Calculate view matrix:
    # from world coordinates to (right=x, up=y, lookat=-z) coordinates.
    # [*1, 6, 3, 3]
    skybox_view_rots = torch.stack((skybox_rights, skybox_ups, -skybox_lookats), dim=-2)
    skybox_view_translates = torch.stack(
        (
            (-skybox_eyes.unsqueeze(-2) * skybox_rights).sum(dim=-1),
            (-skybox_eyes.unsqueeze(-2) * skybox_ups).sum(dim=-1),
            (-skybox_eyes.unsqueeze(-2) * (-skybox_lookats)).sum(dim=-1),
        ),
        dim=-1,
    )  # [..., 6, 3]
    # [..., 6, 3, 1]
    skybox_view_translates = skybox_view_translates.unsqueeze(-1)
    # [..., 6, 3, 3]
    skybox_view_rots = skybox_view_rots * torch.ones_like(skybox_view_translates)
    skybox_view_mats = torch.cat(
        (skybox_view_rots, skybox_view_translates), dim=-1
    )  # [..., 6, 3, 4]
    skybox_view_mats = torch.cat(
        (
            skybox_view_mats,
            torch.zeros_like(skybox_view_mats[..., 0:1, :]),
        ),
        dim=-2,
    )  # [6, 3, 4] + [6, 1, 4] = [6, 4, 4]
    skybox_view_mats[..., -1, -1] = 1.0

    # tan 45 == 1
    skybox_pers_mats = (
        torch.Tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ],
        )
        .to(dtype)
        .to(device)
    )
    skybox_pers_mats = skybox_pers_mats.reshape(*batch_shape, 1, 4, 4)

    return [
        skybox_lookats,
        skybox_ups,
        skybox_rights,
        skybox_view_mats,
        skybox_pers_mats,
    ]


def TBN(normals: torch.Tensor) -> List[torch.Tensor]:
    nx = normals[..., 0]
    ny = normals[..., 1]
    nz = normals[..., 2]

    u_dirs = torch.where(
        normals[..., 2:3].abs().repeat(*([1] * len(normals.shape[:-1])), 3) >= 0.9,
        torch.stack((torch.ones_like(nx), torch.zeros_like(ny), -nx / nz), dim=-1),
        torch.stack((ny, -nx, torch.zeros_like(nz)), dim=-1),
    )
    u_dirs = F.normalize(u_dirs, dim=-1)

    v_dirs = torch.cross(normals, u_dirs, dim=-1)
    v_dirs = F.normalize(v_dirs, dim=-1)

    u_dirs = torch.cross(v_dirs, normals, dim=-1)
    u_dirs = F.normalize(u_dirs, dim=-1)

    return [u_dirs, v_dirs]


def FibonacciLattice(
    n_rays: int, device: torch.device = torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Generates a Fibonacci lattice for the unit 3D sphere
    [deprecated in our training]
    No random number involved.
    """
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0

    if n_rays % 2 == 1:
        # Odd number
        n_max = n_rays // 2
        n_min = -(n_rays // 2)
    else:
        # Even number
        n_max = n_rays // 2 - 1
        n_min = -(n_rays // 2)

    ns = torch.linspace(n_min, n_max, n_rays, device=device, dtype=dtype)

    # Spherical coordinate
    phis = 2.0 * math.pi * ns / golden_ratio
    thetas = torch.acos(2.0 * ns / n_rays)

    # Rectangular coordinates
    x = torch.sin(thetas) * torch.cos(phis)
    y = torch.sin(thetas) * torch.sin(phis)
    z = torch.cos(thetas)
    ray_d = torch.stack((x, y, z), dim=-1)  # [n_rays, 3]

    return ray_d


def GetThetaAndPhiFromRayDirection(
    ray_d: torch.Tensor, keepdim: bool = True
) -> List[torch.Tensor]:
    theta = torch.acos(ray_d[..., 2:3])  # 0 -> PI
    sintheta = torch.sin(theta)  # zero or positive
    eps = 0  # 1e-10
    inv_sintheta = torch.where(
        sintheta.abs() < eps, torch.zeros_like(sintheta), 1.0 / sintheta
    )
    cosphi = ray_d[..., 0:1] * inv_sintheta
    sinphi = ray_d[..., 1:2] * inv_sintheta
    cosphi = torch.clamp(cosphi, -1, 1)
    sinphi = torch.clamp(sinphi, -1, 1)
    phi = torch.acos(cosphi)  # 0 -> PI
    phi = torch.where(sinphi >= 0, phi, 2 * math.pi - phi)  # 0 -> 2PI

    # print(f"theta and phi = {theta.max()}, {theta.min()}, {phi.max()}, {phi.min()}")
    if not keepdim:
        theta = theta.squeeze(-1)
        phi = phi.squeeze(-1)

    return [theta, phi]


def GetThetaAndPhiFromSinAndCos(
    costheta: torch.Tensor, sinphi: torch.Tensor, cosphi: torch.Tensor
) -> List[torch.Tensor]:
    theta = torch.acos(costheta)  # 0 -> PI

    # phi = torch.acos(cosphi)  # 0 -> PI
    # phi = torch.where(sinphi >= 0, phi, 2 * math.pi - phi)  # 0 -> 2PI

    phi = torch.asin(torch.clamp(sinphi, -1, 1))  # -PI/2 -> PI/2
    phi = torch.where(cosphi >= 0, phi, math.pi - phi)  # -PI/2 -> 3PI/2
    phi = torch.where(phi < 0, phi + 2 * math.pi, phi)  # 0 -> 2PI

    # print(f"theta and phi = {theta.max()}, {theta.min()}, {phi.max()}, {phi.min()}")

    return [theta, phi]


def GetSinAndCosFromRayDirection(ray_d: torch.Tensor) -> List[torch.Tensor]:
    costheta = torch.clamp(ray_d[..., 2:3], -1, 1)
    sintheta = torch.sqrt(1 - costheta * costheta)  # zero or positive
    eps = 0  # 1e-5
    inv_sintheta = torch.where(
        sintheta.abs() > eps, 1.0 / sintheta, torch.zeros_like(sintheta)
    )
    cosphi = ray_d[..., 0:1] * inv_sintheta
    sinphi = ray_d[..., 1:2] * inv_sintheta

    # print(f"theta and phi = {theta.max()}, {theta.min()}, {phi.max()}, {phi.min()}")

    results = [sintheta, costheta, cosphi, sinphi]
    results = [torch.clamp(x, -1, 1) for x in results]
    return results


def ReflectionTransform(
    e_s: torch.Tensor, e_p: torch.Tensor, e_i_s: torch.Tensor, e_i_p: torch.Tensor
) -> torch.Tensor:
    """
    Compute basis change matrix for reflections

    Input
    -----
    e_s : [..., 3], tf.float
        Source unit vector for S polarization

    e_p : [..., 3], tf.float
        Source unit vector for P polarization

    e_i_s : [..., 3], tf.float
        Target unit vector for S polarization

    e_i_p : [..., 3], tf.float
        Target unit vector for P polarization

    Output
    -------
    r : [..., 2, 2], tf.float
        Change of basis matrix for going from (e_s, e_p) to (e_i_s, e_i_p)
    """
    r_11 = (e_i_s * e_s).sum(dim=-1)
    r_12 = (e_i_s * e_p).sum(dim=-1)
    r_21 = (e_i_p * e_s).sum(dim=-1)
    r_22 = (e_i_p * e_p).sum(dim=-1)
    r1 = torch.stack((r_11, r_12), dim=-1)
    r2 = torch.stack((r_21, r_22), dim=-1)
    r = torch.stack((r1, r2), dim=-2)
    return r


def _GenOrthogonalVector(k: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Generate an arbitrary vector that is orthogonal to ``k``.

    Input
    ------
    k : [..., 3], tf.float
        Vector

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    : [..., 3], tf.float
        Vector orthogonal to ``k``
    """
    dtype = k.dtype
    device = k.device
    ex = torch.Tensor([1.0, 0.0, 0.0]).to(device).to(dtype)
    ex = ex + torch.zeros_like(k)

    ey = torch.Tensor([0.0, 1.0, 0.0]).to(device).to(dtype)
    ey = ey + torch.zeros_like(k)

    n1 = torch.cross(k, ex, dim=-1)
    n1_norm = torch.norm(n1, dim=-1, keepdim=True)
    n2 = torch.cross(k, ey, dim=-1)
    return torch.where(n1_norm > epsilon, n1, n2)


def ComputeFieldUnitVectors(
    k_i: torch.Tensor,
    k_r: torch.Tensor,
    n: torch.Tensor,
    epsilon: float,
    return_e_r=True,
):
    """
    Compute unit vector parallel and orthogonal to incident plane

    Input
    ------
    k_i : [..., 3], tf.float
        Direction of arrival

    k_r : [..., 3], tf.float
        Direction of reflection

    n : [..., 3], tf.float
        Surface normal

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    return_e_r : bool
        If `False`, only ``e_i_s`` and ``e_i_p`` are returned.

    Output
    ------
    e_i_s : [..., 3], tf.float
        Incident unit field vector for S polarization

    e_i_p : [..., 3], tf.float
        Incident unit field vector for P polarization

    e_r_s : [..., 3], tf.float
        Reflection unit field vector for S polarization.
        Only returned if ``return_e_r`` is `True`.

    e_r_p : [..., 3], tf.float
        Reflection unit field vector for P polarization
        Only returned if ``return_e_r`` is `True`.
    """
    n = n + torch.zeros_like(k_i)
    e_i_s = torch.cross(k_i, n, dim=-1)
    e_i_s_norm = torch.norm(e_i_s, dim=-1, keepdim=True)
    # In case of normal incidence, the incidence plan is not uniquely
    # define and the Fresnel coefficent is the same for both polarization
    # (up to a sign flip for the parallel component due to the definition of
    # polarization).
    # It is required to detect such scenarios and define an arbitrary valid
    # e_i_s to fix an incidence plane, as the result from previous
    # computation leads to e_i_s = 0.
    e_i_s = torch.where(e_i_s_norm > epsilon, e_i_s, _GenOrthogonalVector(n, epsilon))

    e_i_s = F.normalize(e_i_s, dim=-1)
    e_i_p = F.normalize(torch.cross(e_i_s, k_i, dim=-1), dim=-1)
    if not return_e_r:
        return e_i_s, e_i_p
    else:
        e_r_s = e_i_s
        e_r_p = F.normalize(torch.cross(e_r_s, k_r, dim=-1), dim=-1)
        return e_i_s, e_i_p, e_r_s, e_r_p


def ReflectionCoefficient(eta: float, cos_theta: torch.Tensor) -> List[torch.Tensor]:
    """
    Compute simplified reflection coefficients

    Input
    ------
    eta : Any shape, tf.complex
        Real part of the relative permittivity

    cos_thehta : Same as ``eta``, tf.float
        Cosine of the incident angle

    Output
    -------
    r_te : Same as input, tf.complex
        Fresnel reflection coefficient for S direction

    r_tm : Same as input, tf.complex
        Fresnel reflection coefficient for P direction
    """
    # Fresnel equations
    a = cos_theta
    b = torch.sqrt(eta - 1.0 + cos_theta**2)
    r_te = torch.where(a + b == 0, torch.zeros_like(a), (a - b) / (a + b))

    c = eta * a
    d = b
    r_tm = torch.where(c + d == 0, torch.zeros_like(c), (c - d) / (c + d))
    return r_te, r_tm


def ComputeSHBasisSimple(l: int, normals: torch.Tensor):
    assert l >= 0

    x = normals[..., 0:1]
    y = normals[..., 1:2]
    z = normals[..., 2:3]
    device = normals.device
    dtype = normals.dtype

    SH_basis_hlp = [0.5 * math.sqrt(0.5 / math.pi)]
    raw_basis = torch.ones_like(x)
    if l == 0:
        SH_basis_hlp = torch.Tensor(SH_basis_hlp).to(device).to(dtype)
        return raw_basis * SH_basis_hlp

    # else l >= 1
    SH_basis_hlp = SH_basis_hlp + [
        0.5 * math.sqrt(1.5 / math.pi),
        0.5 * math.sqrt(1.5 / math.pi),
        0.5 * math.sqrt(1.5 / math.pi),
    ]
    raw_basis = torch.cat((raw_basis, x, z, y), dim=-1)
    if l == 1:
        SH_basis_hlp = torch.Tensor(SH_basis_hlp).to(device).to(dtype)
        return raw_basis * SH_basis_hlp

    # else l >= 2
    SH_basis_hlp = SH_basis_hlp + [
        0.25 * math.sqrt(7.5 / math.pi),
        0.5 * math.sqrt(7.5 / math.pi),
        0.25 * math.sqrt(2.5 / math.pi),
        0.5 * math.sqrt(7.5 / math.pi),
        0.25 * math.sqrt(7.5 / math.pi),
    ]
    raw_basis = torch.cat(
        (raw_basis, x * x - y * y, x * z, 3 * z * z - 1, y * z, 2 * x * y), dim=-1
    )
    if l == 2:
        SH_basis_hlp = torch.Tensor(SH_basis_hlp).to(device).to(dtype)
        return raw_basis * SH_basis_hlp

    # else l >= 3
    SH_basis_hlp = SH_basis_hlp + [
        0.25 * math.sqrt(17.5 / math.pi),
        0.5 * math.sqrt(105 / math.pi),
        0.25 * math.sqrt(10.5 / math.pi),
        0.25 * math.sqrt(7 / math.pi),
        0.25 * math.sqrt(10.5 / math.pi),
        0.25 * math.sqrt(105 / math.pi),
        0.25 * math.sqrt(17.5 / math.pi),
    ]
    raw_basis = torch.cat(
        (
            raw_basis,
            y * (3 * x * x - y * y),
            x * y * z,
            y * (5 * z * z - 1),
            5 * z * z * z - 3 * z,
            x * (5 * z * z - 1),
            (x * x - y * y) * z,
            x * (x * x - 3 * y * y),
        ),
        dim=-1,
    )
    if l == 3:
        SH_basis_hlp = torch.Tensor(SH_basis_hlp).to(device).to(dtype)
        return raw_basis * SH_basis_hlp

    # else l >= 4
    SH_basis_hlp = SH_basis_hlp + [
        0.75 * math.sqrt(35 / math.pi),
        0.75 * math.sqrt(17.5 / math.pi),
        0.75 * math.sqrt(5 / math.pi),
        0.75 * math.sqrt(2.5 / math.pi),
        3.0 / 16.0 * math.sqrt(1.0 / math.pi),
        0.75 * math.sqrt(2.5 / math.pi),
        0.75 * math.sqrt(5 / math.pi),
        0.75 * math.sqrt(17.5 / math.pi),
        0.75 * math.sqrt(35 / math.pi),
    ]
    raw_basis = torch.cat(
        (
            raw_basis,
            x * y * (x * x - y * y),
            y * z * (3 * x * x - y * y),
            x * y * (7 * z * z - 1),
            y * z * (7 * z * z - 3),
            35 * z * z * z * z - 30 * z * z + 3,
            x * z * (7 * z * z - 3),
            (x * x - y * y) * (7 * z * z - 1),
            x * z * (x * x - 3 * y * y),
            x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y),
        ),
        dim=-1,
    )
    if l == 4:
        SH_basis_hlp = torch.Tensor(SH_basis_hlp).to(device).to(dtype)
        return raw_basis * SH_basis_hlp

    raise RuntimeError(f"For SH: l = {l}, we currently not implemented")


def ComputeSHBasis(l: int, normals: torch.Tensor):
    n_SH = (l + 1) ** 2
    device = normals.device
    dtype = normals.dtype

    theta, phi = GetThetaAndPhiFromRayDirection(ray_d=normals, keepdim=True)
    costheta = torch.cos(theta)

    SH_index = torch.linspace(0, n_SH - 1, n_SH, dtype=torch.int32, device=device)
    ls = torch.sqrt(SH_index).to(torch.int32)  # [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, ...]
    ms = SH_index - ls * ls  # [0, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, ...]
    ms = ms - ls  # [0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, - 1, 0, 1, ...]
    Nlm = torch.ones((n_SH,), dtype=dtype, device=device)
    m_max = ms.max()
    for m in range(1, m_max + 1):
        Nlm = torch.where(ms.abs() >= m, Nlm / ((ls - m + 1) * (ls + m)), Nlm)

    Nlm = ((2 * ls + 1) / (4.0 * math.pi) * Nlm).sqrt()  # Always same for -m or m
    # exp_m = torch.cos(ms * phi) + 1j * torch.sin(ms * phi)
    ms_critera = ms + torch.zeros_like(phi).to(ms.dtype)
    exp_m = math.sqrt(2) * torch.where(
        ms_critera < 0, torch.cos(ms.abs() * phi), torch.sin(ms * phi)
    )
    exp_m[ms_critera == 0] = 1

    np_costheta = costheta.cpu().numpy()
    Plm_list = []
    for i in range(l + 1):
        for j in range(-i, i + 1):
            m = abs(j)

            # Plm_this = lpmv(m, i, np_costheta)
            # Plm_list.append(Plm_this)

            Pmm = torch.pow(1 - costheta * costheta, m / 2)
            for jj in range(m):
                Pmm = -Pmm * (2 * jj + 1)  # (-1)**m * (2m-1)!!

            Pmm_next = costheta * (2 * m + 1) * Pmm
            if m == i:
                Plm_list.append(Pmm)
                continue
            elif m + 1 == i:
                Plm_list.append(Pmm_next)
                continue

            # else in for-loop
            Plm_this = 0
            Plm_prev = Pmm_next
            Plm_prev_prev = Pmm
            for ii in range(m + 2, i + 1):
                Plm_this = (
                    (2 * ii - 1) * costheta * Plm_prev - (ii + m - 1) * Plm_prev_prev
                ) / (ii - m)
                Plm_prev_prev = Plm_prev + 0
                Plm_prev = Plm_this + 0

            Plm_list.append(Plm_this)
    Plm = torch.cat(Plm_list, dim=-1)  # [n_rays, n_SH]

    # Plm = np.concatenate(Plm_list, axis=-1)  # [n_rays, n_SH]
    # Plm = torch.from_numpy(Plm).to(device).to(dtype)
    Yml = Plm * Nlm * exp_m

    return Yml


def get_splatted_map_from_CIR(
    n_rays: int,
    shifted_a: torch.Tensor,
    theta_r: torch.Tensor,
    phi_r: torch.Tensor,
    mode: str = "scatter_",
) -> torch.Tensor:
    """
    Args:
        shifted_a: torch.Tensor = [B, HW, n_paths, ...]
        theta_r:   torch.Tensor = [B, HW, n_paths, ...]
        phi_r:     torch.Tensor = [B, HW, n_paths, ...]
    Return:
        shifted_a_map: torch.Tensor = [B, HW, n_rays, ...]
    """
    # TODO: This function need to support other kind of rasterizers
    device = shifted_a.device
    n_theta = int(math.sqrt(n_rays / 2))
    n_phi = n_theta * 2
    theta_interval = math.pi / n_theta
    phi_interval = (2 * math.pi) / n_phi

    batch_size, HW, n_path = shifted_a.shape[0:3]
    rest_shape = shifted_a.shape[3:]
    shifted_a_map = torch.zeros(
        (batch_size * HW, n_rays, *rest_shape), dtype=shifted_a.dtype, device=device
    )

    theta_idx = (theta_r / theta_interval).to(torch.int64)
    theta_idx = theta_idx % n_theta
    phi_r = torch.where(phi_r < 0, phi_r + 2 * math.pi, phi_r)
    phi_idx = (phi_r / phi_interval).to(torch.int64)
    phi_idx = phi_idx % n_phi

    theta_idx = theta_idx.reshape(
        batch_size * HW, n_path, -1, *([1] * (len(rest_shape) - 1))
    )
    phi_idx = phi_idx.reshape(
        batch_size * HW, n_path, -1, *([1] * (len(rest_shape) - 1))
    )
    ray_idx = theta_idx * n_phi + phi_idx

    src_shifted_a = shifted_a.reshape(batch_size * HW, n_path, *rest_shape)

    n_patch = ray_idx.dim() - src_shifted_a.dim()
    if n_patch > 0:
        ray_idx = ray_idx.reshape(*ray_idx.shape, *([1] * n_patch))
    ray_idx = ray_idx.expand_as(src_shifted_a)

    if mode == "scatter_":
        shifted_a_map.scatter_(dim=1, index=ray_idx, src=src_shifted_a)
    elif mode == "scatter_add_":
        shifted_a_map.scatter_add_(dim=1, index=ray_idx, src=src_shifted_a)

    shifted_a_map = shifted_a_map.reshape(batch_size, HW, n_rays, *rest_shape)

    return shifted_a_map


def ZeroForcing_Precoder(H: torch.Tensor) -> torch.Tensor:
    """
    Args:
        H (torch.Tensor): [n_tx, n_rx]
    return:
        torch.Tensor = [n_rx, n_tx]
    """
    V = torch.linalg.pinv(H)
    num_rx, _ = H.shape
    D = torch.zeros((num_rx, num_rx)).to(H.device).to(H.dtype)
    V_norm = torch.linalg.norm(V, ord=2, dim=0)
    V_norm_inv = torch.where(V_norm > 0, 1.0 / V_norm, torch.zeros_like(V_norm))
    D[range(num_rx), range(num_rx)] = V_norm_inv + 1j * torch.zeros_like(V_norm_inv)
    G = V @ D  # [n_rx, n_rx]
    return G


def round_to_grid(values: torch.Tensor, grid: torch.Tensor):
    # Ensure the grid is sorted
    grid = torch.sort(grid)[0]

    values_dict = {"real": values.real, "imag": values.imag}
    for key in values_dict:
        # Find the indices of the nearest grid points
        indices = np.searchsorted(grid, values_dict[key], side="left")

        # Adjust indices for values that might be closer to the previous grid point
        indices = np.clip(indices, 1, len(grid) - 1)
        left = grid[indices - 1]
        right = grid[indices]
        indices -= (values_dict[key] - left) < (right - values_dict[key])

        # Return the grid values at these indices
        values_dict[f"{key}_rounded"] = grid[indices]

    # Combine rounded real and imaginary parts into complex numbers
    return values_dict["real_rounded"] + 1j * values_dict["imag_rounded"]


def find_nearest_coefficient(
    z: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    distances = torch.abs(coefficients - z)
    nearest_index = torch.argmin(distances)
    return coefficients[nearest_index]


def round_to_grid2(values: torch.Tensor) -> torch.Tensor:
    R = 2  # Assuming R = 4, adjust as needed

    # Generate all possible binary combinations for Vre, Wre, Vim, Wim
    binary_combinations = list(itertools.product([0, 1], repeat=4 * R))

    # Calculate coefficients for each combination
    coefficients = []
    for combo in binary_combinations:
        Vre = combo[:R]
        Wre = combo[R : 2 * R]
        Vim = combo[2 * R : 3 * R]
        Wim = combo[3 * R :]

        coeff_re = sum(2**r * (Vre[r] - Wre[r]) for r in range(R)) / float(
            (2.0**R - 1.0)
        )
        coeff_im = sum(2**r * (Vim[r] - Wim[r]) for r in range(R)) / float(
            (2.0**R - 1.0)
        )

        coefficients.append(coeff_re + 1j * coeff_im)

    # Convert to numpy array
    coefficients = np.array(coefficients)
    coefficients = torch.from_numpy(coefficients).to(values.device)

    number = find_nearest_coefficient(values, coefficients)

    return number


def calculate_sum_rate(
    save_path: str, H: torch.Tensor, datatype: str, mode: int = 0
) -> None:
    """
    Args:
        save_path (str): The directory of saving files
        H (torch.Tensor): [n_tx, n_rx]
        mode (int): 0 or 1 for Downlink or Uplink sum_rate
    """
    ZF_diagonal_path = os.path.join(save_path, f"ZF_diagonal_output_{datatype}.json")
    perfect_SINR_path = os.path.join(save_path, f"perfect_SINR_{datatype}.png")
    perfect_coeff_path = os.path.join(save_path, f"perfect_coeff_{datatype}.png")
    ZF_diagonal_rounded_path = os.path.join(
        save_path, f"ZF_diagonal_rounded_output_{datatype}.json"
    )

    G = ZeroForcing_Precoder(H)

    # -----------------------------------------------------------------------------
    # Perfect Precoder

    # print(H.shape)

    num_tx, num_rx = H.shape
    Y = H.T @ G.T  # [n_rx, n_rx]

    diagonal_elements = Y.abs().diag()
    with open(ZF_diagonal_path, "w") as f:
        json.dump(diagonal_elements.cpu().tolist(), f)

    # print("Y = " ,Y.shape, " | maxY = ", np.max(np.abs(Y)))
    # print("G = " ,G.shape, " | maxG = ", np.max(np.abs(G)))

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(
        range(num_rx),
        range(num_rx),
        Y.abs().cpu().numpy(),
        shading="auto",
        cmap="binary",
        vmax=5e-3,
    )
    plt.colorbar()
    plt.xlabel("Data Stream")
    plt.ylabel("User")
    plt.title("Signal Strength")

    plt.subplot(1, 2, 2)
    plt.pcolormesh(
        range(num_tx),
        range(num_rx),
        G.abs().cpu().numpy(),
        shading="auto",
        cmap="binary",
    )
    plt.colorbar(label="Average SINR")
    plt.xlabel("TX")
    plt.ylabel("User")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)
    plt.savefig(perfect_SINR_path)

    plt.figure(figsize=(12, 12))
    G_flat = G.flatten()
    plt.scatter(G_flat.real.cpu().numpy(), G_flat.imag.cpu().numpy(), alpha=0.5)
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.title("Scatter plot of G values (Real vs Imaginary)")
    plt.scatter(0, 0, color="red", s=100, zorder=5)
    plt.axis("equal")
    plt.grid(True)
    plt.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    plt.axvline(x=0, color="k", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(perfect_coeff_path)

    # -----------------------------------------------------------------------------
    # Quantization

    # Assuming G is your original precoding matrix
    grid = torch.linspace(-1, 1, 2**3, device=H.device, dtype=torch.float32)
    step = grid[1] - grid[0]
    grid2 = torch.linspace(
        -1 + step / 2, 1 + step / 2, 2**3, device=H.device, dtype=torch.float32
    )
    print("grid and grid2:", grid, grid2)

    G_rounded = torch.zeros_like(G)

    # Round each entry individually
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G_rounded[i, j] = round_to_grid2(G[i, j])

    Y_rounded = H.T @ G_rounded.T

    diagonal_elements = Y_rounded.abs().diag()
    with open(ZF_diagonal_rounded_path, "w") as f:
        json.dump(diagonal_elements.cpu().tolist(), f)

    # plt.figure(figsize=(16, 8))
    # plt.subplot(1, 2, 1)
    # plt.pcolormesh(
    #     range(num_rx), range(num_rx), Y_rounded.abs().cpu().numpy(), shading="auto"
    # )
    # plt.colorbar()
    # plt.xlabel("Data Stream")
    # plt.ylabel("User")
    # plt.title("Signal Strength")

    # plt.subplot(1, 2, 2)
    # plt.pcolormesh(
    #     range(num_tx), range(num_rx), G_rounded.abs().cpu().numpy(), shading="auto"
    # )
    # plt.colorbar(label="Average SINR")
    # plt.xlabel("TX")
    # plt.ylabel("User")
    # plt.savefig("QUNATIZED_SINR.png")

    # plt.figure(figsize=(12, 12))
    # G_flat = G_rounded.flatten()
    # plt.scatter(np.real(G_flat), np.imag(G_flat), alpha=0.5)
    # plt.xlabel("Real part")
    # plt.ylabel("Imaginary part")
    # plt.title("Scatter plot of G values (Real vs Imaginary)")
    # plt.scatter(0, 0, color="red", s=100, zorder=5)
    # plt.axis("equal")
    # plt.grid(True)
    # plt.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    # plt.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
    # plt.tight_layout()
    # plt.savefig("QUANTIZED_COEFF.png")

    # -----------------------------------------------------------------------------
