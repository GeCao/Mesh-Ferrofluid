import numpy as np
import imageio
import torch
import torch.nn.functional as F
import kaolin.ops.conversions as conversions
import open3d as o3d
from scipy.spatial import cKDTree
from typing import List, Union

from src.utils import PanelRelationType


def voxelgrids_to_cubic_meshes(
    phi: torch.Tensor, iso_value: float = 0, normlize: bool = False
) -> List[List[torch.Tensor]]:
    """Get Triangle mesh for a SDF grid
    We have to return the verts & faces with List format,
    as for different batch, their number of vertices and faces might different.

    Args:
        phi (torch.Tensor): [B, 1, D, H, W], include liquid (<0) and gas (>0)
        iso_value (float): 0, indicated the critical value for dividing surface
        normlize (bool): Normalize the mesh to [-1, 1], default as False

    Raises:
        RuntimeError: Only 3D grid is supported to transfer a SDF to triangle mesh

    Returns:
        List[torch.Tensor]: B x [N_verts, dim==3]
        List[torch.Tensor]: B x [N_faces, dim==3]
    """
    if phi.dim() != 5:
        raise RuntimeError(
            "Only 3D grid is supported to transfer a SDF to triangle mesh\n"
            "A 3D grid should be in format: [B, C, (D), H, W]"
        )

    batch_size, n_ch, D, H, W = phi.shape
    assert n_ch == 1  # A scalar field

    sdf = phi[:, 0, 1:-1, 1:-1, 1:-1]  # delete boundary and cancel channel
    sdf = sdf.permute((0, 3, 2, 1))  # From [B, D, H, W] -> [B, W, H, D]
    verts_list, faces_list = conversions.voxelgrids_to_trianglemeshes(
        sdf, iso_value=iso_value
    )  # B x [N_verts, 3], B x [N_faces, 3]

    # Normalize the mesh
    if normlize:
        for i in range(len(verts_list)):
            AABB_min, _ = verts_list[i].min(dim=-2)
            AABB_max, _ = verts_list[i].max(dim=-2)
            AABB_center = (AABB_min + AABB_max) / 2.0
            AABB_len = AABB_max - AABB_min
            AABB_radius = AABB_len.max() / 2.0
            verts_list[i] = (verts_list[i] - AABB_center) / AABB_radius

    return verts_list, faces_list


def meshes_to_voxelgrids(
    simulation_size: List[int], verts: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Get voxel grid from a triangle mesh

    Args:
        simulation_size (List[int]): [B, 1, D, H, W]
        verts (torch.Tensor): [N_verts, dim=3]
        faces (torch.Tensor): [N_faces, dim = 3]

    Returns:
        torch.Tensor: [B, 1, D, H, W], the voxel grid.
    """
    batch_size, _, D, H, W = simulation_size
    min_res = min(simulation_size[2:])
    resolution = int(min_res * 0.45)
    assert verts.dim() == 2

    phi = conversions.trianglemeshes_to_voxelgrids(
        vertices=verts[None], faces=faces, resolution=resolution
    )
    phi = phi.permute((0, 3, 2, 1)).unsqueeze(1)

    padx = ((W - resolution) // 2, W - resolution - (W - resolution) // 2)
    pady = ((H - resolution) // 2, H - resolution - (H - resolution) // 2)
    padz = ((D - resolution) // 2, D - resolution - (D - resolution) // 2)
    pad = (*padx, *pady, *padz)
    phi = F.pad(phi, pad=pad, mode="constant", value=0)

    return phi


def get_vertices_from_index(
    vertices: Union[torch.Tensor, List[torch.Tensor]], index: torch.Tensor
) -> torch.Tensor:
    """Select Vertices by index
    Typically used to select face_vertices by faces index

    Args:
        vertices (torch.Tensor): [N_verts, dim=3]
        index (torch.Tensor): [N], dtype=int64

    Args (Alternative):
        vertices (torch.Tensor): [B, N_verts, dim=3]
        index (torch.Tensor): [B, N], dtype=int64

    Args (Alternative):
        vertices (List[torch.Tensor]): B x [N_verts, dim=3]
        index (List[torch.Tensor]): B x [N,], dtype=int64

    Raises:
        RuntimeError: vertices shape should be in [(B), N_verts, dim]

    Returns:
        torch.Tensor: _description_
    """
    if type(vertices) == torch.Tensor:
        if len(vertices.shape) == 2:  # [N_verts, dim=3]
            assert len(index.shape) == 1
            selected_vertices = torch.index_select(
                vertices, -2, torch.flatten(index).to(torch.int64)
            )  # [...,  N_faces*dim, dim]
        elif len(vertices.shape) == 3:
            batch_size, N_verts, dim = vertices.shape
            assert len(index.shape) == 2 and index.shape[0] == batch_size

            batch_offset = torch.linspace(
                0, batch_size - 1, batch_size, dtype=index.dtype, device=index.device
            )
            batch_offset = batch_offset[:, None] * N_verts
            batch_offset = batch_offset.to(index.device).to(index.dtype)

            selected_vertices = torch.index_select(
                vertices.reshape(-1, dim),
                -2,
                torch.flatten(index + batch_offset).to(torch.int64),
            )  # [...,  N_faces*dim, dim]

            selected_vertices = selected_vertices.reshape(batch_size, -1, dim)
        else:
            raise RuntimeError("Vertices shape should be in [(B), N_verts, dim]")
    elif type(vertices) == list:
        batch_size = len(vertices)
        assert len(index) == batch_size

        selected_vertices = []
        for batch_idx in range(batch_size):
            assert len(vertices[batch_idx].shape) == 2  # [N_verts, dim=3]
            assert len(index[batch_idx].shape) == 1  # [N,]

            new_selected_vertices = torch.index_select(
                vertices, -2, index[batch_idx].to(torch.int64)
            )  # [...,  N_faces*dim, dim]
            selected_vertices.append(new_selected_vertices)

    return selected_vertices


def compute_tensor_face_normals(
    vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Compute face normals for tensor mesh input

    Args:
        vertices (torch.Tensor): [(B), N_verts, dim]
        faces (torch.Tensor): [(B), N_faces, dim]

    Raises:
        RuntimeError: We only accept 2D/3D as dimension

    Returns:
        torch.Tensor: [(B), N_faces, dim], the face normals
    """
    assert vertices.dim() == 2 or vertices.dim() == 3
    N_faces, dim = faces.shape[-2:]
    if dim == 2:
        edges_index = faces.reshape(*(faces.shape[:-2]), N_faces * dim)
        edge_vertices = get_vertices_from_index(vertices, index=edges_index)

        edges = (
            edge_vertices[..., 1::2, :] - edge_vertices[..., 0::2, :]
        )  # [N_faces, dim]
        normals = torch.stack((edges[..., 1], -edges[..., 0]), dim=-1)
        normals = F.normalize(normals, dim=-1)
    elif dim == 3:
        edges_index = torch.cat((faces[..., 0:2], faces[..., 1:3]), dim=-1)
        edges_index = edges_index.reshape(*(faces.shape[:-2]), N_faces * 4)
        edge_vertices = get_vertices_from_index(vertices, index=edges_index)

        edges = (
            edge_vertices[..., 1::2, :] - edge_vertices[..., 0::2, :]
        )  # [N_faces*2, dim]
        normals = torch.cross(edges[..., 0::2, :], edges[..., 1::2, :], dim=-1)
        normals = F.normalize(normals, dim=-1)  # [N_faces, dim]
    else:
        raise RuntimeError("We only accept 2D/3D as dimension")

    return normals


def compute_tensor_face_areas(
    vertices: torch.Tensor, faces: torch.Tensor, keepdim: bool = True
) -> torch.Tensor:
    """Compute face areas for tensor mesh input

    Args:
        vertices (torch.Tensor): [(B), N_verts, dim]
        faces (torch.Tensor): [(B), N_faces, dim]
        keepdim (bool): keep dim or not

    Raises:
        RuntimeError: We only accept 2D/3D as dimension

    Returns:
        torch.Tensor: [N_faces, (1)], the face areas
    """
    assert vertices.dim() == 2 or vertices.dim() == 3
    N_faces, dim = faces.shape[-2:]
    if dim == 2:
        edges_index = faces.reshape(*(faces.shape[:-2]), N_faces * dim)
        edge_vertices = get_vertices_from_index(vertices, index=edges_index)

        edges = (
            edge_vertices[..., 1::2, :] - edge_vertices[..., 0::2, :]
        )  # [...,  N_faces, dim]
        areas = edges.norm(dim=-1, keepdim=keepdim)
    elif dim == 3:
        edges_index = torch.cat((faces[..., 0:2], faces[..., 1:3]), dim=-1)
        edges_index = edges_index.reshape(*(faces.shape[:-2]), N_faces * 4)
        edge_vertices = get_vertices_from_index(vertices, index=edges_index)

        edges = (
            edge_vertices[..., 1::2, :] - edge_vertices[..., 0::2, :]
        )  # [...,  NumOfFaces*2, dim]
        areas = torch.cross(edges[..., 0::2, :], edges[..., 1::2, :], dim=-1)
        areas = 0.5 * areas.norm(dim=-1, keepdim=keepdim)
    else:
        raise RuntimeError("We only accept 2D/3D as dimension")

    return areas


def compute_tensor_vert_normals(
    vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Compute vert normals for tensor mesh input

    Args:
        vertices (torch.Tensor): [(B), N_verts, dim]
        faces (torch.Tensor): [(B), N_faces, dim]

    Returns:
        torch.Tensor: [(B), N_verts, 3], the face normals
    """
    dim = vertices.shape[-1]
    face_normals = compute_tensor_face_normals(vertices=vertices, faces=faces)
    face_areas = compute_tensor_face_areas(vertices=vertices, faces=faces, keepdim=True)
    vert_normals = torch.zeros_like(vertices)  # [(B), N_verts, dim]

    N_faces, dim = faces.shape[-2:]

    hlp_repeat_index = [1] * len(face_normals.shape[:-1])
    vert_normals.scatter_add_(
        -2,
        faces.reshape(*faces.shape[:-2], -1, 1)
        .repeat(*hlp_repeat_index, dim)
        .to(torch.int64),
        (face_areas * face_normals)
        .unsqueeze(-2)
        .repeat(*hlp_repeat_index, dim, 1)
        .reshape(*face_normals.shape[:-2], -1, dim),
    )
    vert_normals = F.normalize(vert_normals, dim=-1)

    return vert_normals


def compute_tensor_vert_areas(
    vertices: torch.Tensor, faces: torch.Tensor, keepdim: bool = True
) -> torch.Tensor:
    """Compute vert areas for tensor mesh input

    Args:
        vertices (torch.Tensor): [(B), N_verts, dim]
        faces (torch.Tensor): [(B), N_faces, dim]
        keepdim (bool): True as default

    Returns:
        torch.Tensor: [(B), N_verts, 3], the face areas
    """
    dim = vertices.shape[-1]
    face_areas = compute_tensor_face_areas(vertices=vertices, faces=faces, keepdim=True)
    vert_areas = torch.zeros_like(vertices[..., 0:1])  # [(B), N_verts, 1]

    N_faces, dim = faces.shape[-2:]

    hlp_repeat_index = [1] * len(face_areas.shape[:-1])
    vert_areas_index = faces.reshape(*faces.shape[:-2], N_faces * dim, 1)
    vert_areas.scatter_add_(
        -2,
        vert_areas_index.to(torch.int64),
        face_areas.unsqueeze(-2)
        .repeat(*hlp_repeat_index, dim, 1)
        .reshape(*face_areas.shape[:-2], N_faces * dim, 1)
        / dim,
    )

    if not keepdim:
        vert_areas = vert_areas.squeeze(-1)

    return vert_areas


def compute_face_normals(
    vertices: Union[torch.Tensor, List[torch.Tensor]],
    faces: Union[torch.Tensor, List[torch.Tensor]],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute face normals for single/multiple mesh input

    Args:
        vertices (torch.Tensor): [N_verts, dim=3]
        index (torch.Tensor): [N], dtype=int64

    Args (Alternative):
        vertices (torch.Tensor): [B, N_verts, dim=3]
        index (torch.Tensor): [B, N], dtype=int64

    Args (Alternative):
        vertices (List[torch.Tensor]): B x [N_verts, dim=3]
        index (List[torch.Tensor]): B x [N,], dtype=int64

    Raises:
        RuntimeError: mesh vertices has to be in format of Tensor or List[Tensor]

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: face normals
    """
    if type(vertices) == list:
        batch_size = len(vertices)
        assert len(faces) == len(vertices)

        results = []
        for i in range(batch_size):
            results.append(
                compute_tensor_face_normals(vertices=vertices[i], faces=faces[i])
            )
    elif type(vertices) == torch.Tensor:
        results = compute_tensor_face_normals(vertices=vertices, faces=faces)
    else:
        raise RuntimeError(
            f"mesh vertices has to be in format of Tensor or List[Tensor]"
        )

    return results


def compute_face_areas(
    vertices: Union[torch.Tensor, List[torch.Tensor]],
    faces: Union[torch.Tensor, List[torch.Tensor]],
    keepdim: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute face areas for single/multiple mesh input

    Args:
        vertices (torch.Tensor): [N_verts, dim=3]
        index (torch.Tensor): [N], dtype=int64
        keepdim (bool): True as default

    Args (Alternative):
        vertices (torch.Tensor): [B, N_verts, dim=3]
        index (torch.Tensor): [B, N], dtype=int64
        keepdim (bool): True as default

    Args (Alternative):
        vertices (List[torch.Tensor]): B x [N_verts, dim=3]
        index (List[torch.Tensor]): B x [N,], dtype=int64
        keepdim (bool): True as default

    Raises:
        RuntimeError: mesh vertices has to be in format of Tensor or List[Tensor]

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: face areas
    """
    if type(vertices) == list:
        batch_size = len(vertices)
        assert len(faces) == len(vertices)

        results = []
        for i in range(batch_size):
            results.append(
                compute_tensor_face_areas(
                    vertices=vertices[i], faces=faces[i], keepdim=keepdim
                )
            )
    elif type(vertices) == torch.Tensor:
        results = compute_tensor_face_areas(
            vertices=vertices, faces=faces, keepdim=keepdim
        )
    else:
        raise RuntimeError(
            f"mesh vertices has to be in format of Tensor or List[Tensor]"
        )

    return results


def compute_vert_normals(
    vertices: Union[torch.Tensor, List[torch.Tensor]],
    faces: Union[torch.Tensor, List[torch.Tensor]],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute vert normals for single/multiple mesh input

    Args:
        vertices (torch.Tensor): [N_verts, dim=3]
        index (torch.Tensor): [N], dtype=int64

    Args (Alternative):
        vertices (torch.Tensor): [B, N_verts, dim=3]
        index (torch.Tensor): [B, N], dtype=int64

    Args (Alternative):
        vertices (List[torch.Tensor]): B x [N_verts, dim=3]
        index (List[torch.Tensor]): B x [N,], dtype=int64

    Raises:
        RuntimeError: mesh vertices has to be in format of Tensor or List[Tensor]

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: vert normals
    """
    if type(vertices) == list:
        batch_size = len(vertices)
        assert len(faces) == len(vertices)

        results = []
        for i in range(batch_size):
            results.append(
                compute_tensor_vert_normals(vertices=vertices[i], faces=faces[i])
            )
    elif type(vertices) == torch.Tensor:
        results = compute_tensor_vert_normals(vertices=vertices, faces=faces)
    else:
        raise RuntimeError(
            f"mesh vertices has to be in format of Tensor or List[Tensor]"
        )

    return results


def compute_vert_areas(
    vertices: Union[torch.Tensor, List[torch.Tensor]],
    faces: Union[torch.Tensor, List[torch.Tensor]],
    keepdim: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Compute vert aeras for single/multiple mesh input

    Args:
        vertices (torch.Tensor): [N_verts, dim=3]
        index (torch.Tensor): [N], dtype=int64
        keepdim (bool): True as default

    Args (Alternative):
        vertices (torch.Tensor): [B, N_verts, dim=3]
        index (torch.Tensor): [B, N], dtype=int64
        keepdim (bool): True as default

    Args (Alternative):
        vertices (List[torch.Tensor]): B x [N_verts, dim=3]
        index (List[torch.Tensor]): B x [N,], dtype=int64
        keepdim (bool): True as default

    Raises:
        RuntimeError: mesh vertices has to be in format of Tensor or List[Tensor]

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: vert areas
    """
    if type(vertices) == list:
        batch_size = len(vertices)
        assert len(faces) == len(vertices)

        results = []
        for i in range(batch_size):
            results.append(
                compute_tensor_vert_areas(
                    vertices=vertices[i], faces=faces[i], keepdim=keepdim
                )
            )
    elif type(vertices) == torch.Tensor:
        results = compute_tensor_vert_areas(
            vertices=vertices, faces=faces, keepdim=keepdim
        )
    else:
        raise RuntimeError(
            f"mesh vertices has to be in format of Tensor or List[Tensor]"
        )

    return results


def compute_mesh_face_tangent_gradient(
    vert_p: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the tangent gradient of u on mesh,

    Args:
        vert_p (torch.Tensor): [B, N_verts, dim=3]
        verts (torch.Tensor): [N_Verts, dim=3]
        faces (torch.Tensor): [N_faces, dim=3]
        face_normals (torch.Tensor): [N_faces, dim=3]

    Returns:
        torch.Tensor: [B, N_faces, dim] The tangent gradient of u.
    """
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape
    batch_size, N_verts, _ = vert_p.shape

    if face_normals is None:
        face_normals = compute_face_normals(vertices=verts, faces=faces)

    face_areas = compute_face_areas(vertices=verts, faces=faces, keepdim=True)

    face_verts = get_vertices_from_index(verts, index=faces.flatten())
    face_verts = face_verts.reshape(N_faces, dim, dim)

    face_vert_p = torch.index_select(
        vert_p, dim=-2, index=faces.flatten().to(torch.int64)
    )
    face_vert_p = face_vert_p.reshape(batch_size, N_faces, dim, 1)
    face_gradp_tangent0 = face_vert_p[..., 0, :] * torch.cross(
        face_normals, face_verts[..., 2, :] - face_verts[..., 1, :], dim=-1
    )  # [B, N_faces, dim]
    face_gradp_tangent1 = face_vert_p[..., 1, :] * torch.cross(
        face_normals, face_verts[..., 0, :] - face_verts[..., 2, :], dim=-1
    )
    face_gradp_tangent2 = face_vert_p[..., 2, :] * torch.cross(
        face_normals, face_verts[..., 1, :] - face_verts[..., 0, :], dim=-1
    )
    face_gradp_tangent = (
        face_gradp_tangent0 + face_gradp_tangent1 + face_gradp_tangent2
    ) / 2.0  # [B, N_faces, dim]

    face_gradp_tangent = face_gradp_tangent / face_areas

    return face_gradp_tangent
    # v0 = verts[faces[:, 0]]  # [N_f, 3]
    # v1 = verts[faces[:, 1]]
    # v2 = verts[faces[:, 2]]

    # e1 = v1 - v0  # [N_f, 3]
    # e2 = v2 - v0

    # J = torch.stack([e1, e2], dim=-1)  # [N_f, 3, 2]

    # # Compute inverse transpose of Jacobian (J^-T)
    # JT_inv = torch.linalg.pinv(J.transpose(1, 2))  # [N_f, 2, 3]

    # # Extract per-face pressure (batched)
    # p0 = vert_p[:, faces[:, 0], 0]  # [B, N_f]
    # p1 = vert_p[:, faces[:, 1], 0]
    # p2 = vert_p[:, faces[:, 2], 0]
    # dp = torch.stack([p1 - p0, p2 - p0], dim=-1)  # [B, N_f, 2]

    # # Compute tangential gradient: JT_inv @ dp^T
    # grad = torch.einsum("fij,bfj->bfi", JT_inv, dp)  # [B, N_f, 3]

    # return grad  # Tangential gradient per triangle


def compute_mesh_vert_tangent_gradient(
    vert_p: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the tangent gradient of u on mesh,

    Args:
        vert_p (torch.Tensor): [B, N_verts, dim=3]
        verts (torch.Tensor): [N_Verts, dim=3]
        faces (torch.Tensor): [N_faces, dim=3]
        face_normals (torch.Tensor): [N_faces, dim=3]

    Returns:
        torch.Tensor: [B, N_verts, dim] The tangent gradient of u.
    """
    device = vert_p.device
    dtype = vert_p.dtype
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape
    batch_size, N_verts, _ = vert_p.shape

    face_gradp_tangent = compute_mesh_face_tangent_gradient(
        vert_p=vert_p, verts=verts, faces=faces, face_normals=face_normals
    )  # [B, N_faces, dim]

    scatter_src = (
        face_gradp_tangent[..., None, :]
        .repeat(1, 1, dim, 1)
        .reshape(batch_size, N_faces * dim, dim)
        / dim
    )
    scatter_index = (
        faces.reshape(1, N_faces * dim, 1).repeat(batch_size, 1, dim).to(torch.int64)
    )
    vert_gradp_tangent = torch.zeros(
        (batch_size, N_verts, dim), device=device, dtype=dtype
    )
    vert_gradp_tangent.scatter_add_(-2, scatter_index, scatter_src)  # [B, N_verts, dim]

    return vert_gradp_tangent


def compute_mesh_face_curl(
    vert_p: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor = None,
    face_gradp_tangent: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the curl operator on mesh

    Args:
        vert_p (torch.Tensor): [B, N_verts, dim]
        verts (torch.Tensor): [N_verts, dim]
        faces (torch.Tensor): [N_faces, dim]
        face_normals (torch.Tensor, optional): [N_faces, dim]. Defaults to None.
        face_gradp_tangent (torch.Tensor, optional): [B, N_faces, dim]. Defaults to None.

    Returns:
        torch.Tensor: [B, N_faces, dim], the curl(p)
    """
    batch_size, N_verts, _ = vert_p.shape

    if face_normals is None:
        face_normals = compute_face_normals(vertices=verts, faces=faces)

    if face_gradp_tangent is None:
        face_gradp_tangent = compute_mesh_face_tangent_gradient(
            vert_p=vert_p, verts=verts, faces=faces, face_normals=face_normals
        )  # [B, N_faces, dim=3]

    batched_face_normals = face_normals[None].repeat(batch_size, 1, 1)
    face_curlp = torch.cross(
        batched_face_normals, face_gradp_tangent, dim=-1
    )  # [B, N_faces, dim=3]

    return face_curlp


def compute_mesh_vert_curl(
    vert_p: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor = None,
    face_gradp_tangent: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the curl operator on mesh

    Args:
        vert_p (torch.Tensor): [B, N_verts, dim]
        verts (torch.Tensor): [N_verts, dim]
        faces (torch.Tensor): [N_faces, dim]
        face_normals (torch.Tensor, optional): [N_faces, dim]. Defaults to None.
        face_gradp_tangent (torch.Tensor, optional): [B, N_faces, dim]. Defaults to None.

    Returns:
        torch.Tensor: [B, N_faces, dim], the curl(p)
    """
    N_faces, dim = faces.shape
    batch_size, N_verts, dim = vert_p.shape

    face_curlp = compute_mesh_face_curl(
        vert_p=vert_p,
        verts=verts,
        faces=faces,
        face_normals=face_normals,
        face_gradp_tangent=face_gradp_tangent,
    )  # [B, N_faces, dim=3]

    scatter_src = (
        face_curlp[..., None, :]
        .repeat(1, 1, dim, 1)
        .reshape(batch_size, N_faces * dim, dim)
        / dim
    )
    scatter_index = (
        faces.reshape(1, N_faces * dim, 1).repeat(batch_size, 1, dim).to(torch.int64)
    )
    vert_curlp = torch.zeros_like(vert_p)
    vert_curlp.scatter_add_(-2, scatter_index, scatter_src)  # [B, N_verts, dim]

    return vert_curlp


def compute_panel_relation(faces: torch.Tensor) -> torch.Tensor:
    """Compute the relationship between each panels

    Args:
        faces (torch.Tensor): [N_faces, dim]

    Returns:
        torch.Tensor: [N_faces, N_faces]
    """
    N_faces, dim = faces.shape
    device = faces.device

    panel_flags = torch.zeros((N_faces, N_faces), device=device, dtype=torch.uint8)
    panel_flags[...] = int(PanelRelationType.SEPARATE)

    same_vertex_number = (
        faces.reshape(N_faces, 1, dim, 1) == faces.reshape(N_faces, 1, dim)
    ).to(torch.int32)
    same_vertex_number = same_vertex_number.reshape(N_faces, N_faces, dim * dim).sum(
        dim=-1
    )  # [N_faces, N_faces]
    panel_flags[same_vertex_number == 1] = int(PanelRelationType.SAME_VERTEX)
    panel_flags[same_vertex_number == 2] = int(PanelRelationType.SAME_EDGE)
    panel_flags[same_vertex_number == 3] = int(PanelRelationType.SAME_FACE)

    return panel_flags


def remesh_and_transfer_velocity(
    verts: torch.Tensor,
    faces: torch.Tensor,
    vel: torch.Tensor,
    target_edge_length: int = 0.1,
) -> List[torch.Tensor]:
    """_summary_

    Args:
        verts (torch.Tensor): [N_verts, dim=3]
        faces (torch.Tensor): [N_faces, dim=3]
        vel (torch.Tensor): [B, N_verts, dim=3]
        target_edge_length (float, optional): Defaults to 0.1.

    Returns:
        new_verts (torch.Tensor): [N_new_verts, dim=3]
        new_faces (torch.Tensor): [N_new_faces, dim=3]
        new_vel (torch.Tensor): [B, N_new_verts, dim=3]
    """
    device = vel.device
    dtype = vel.dtype
    B, N_verts, _ = vel.shape
    assert verts.shape[0] == N_verts and verts.shape[1] == 3

    np_verts = verts.cpu().numpy()
    np_faces = faces.cpu().numpy()

    # --- Step 1: Remesh using Open3D (CPU)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np_verts)
    mesh.triangles = o3d.utility.Vector3iVector(np_faces)
    mesh.compute_vertex_normals()

    # --- Step 2: Subdivide mesh until edge length < target
    def average_edge_length(mesh: o3d.geometry.TriangleMesh) -> float:
        """
        Compute average edge length of a triangle mesh using face data.
        """
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        # Extract all edges from faces (3 edges per triangle)
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        # Sort and deduplicate to get unique edges
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        # Compute edge lengths
        v0 = verts[edges[:, 0]]
        v1 = verts[edges[:, 1]]
        lengths = np.linalg.norm(v1 - v0, axis=1)

        return np.mean(lengths)

    current_avg_len = mesh.get_max_bound() - mesh.get_min_bound()
    current_avg_len = np.linalg.norm(current_avg_len) / (np.sqrt(len(mesh.vertices)))
    iterations = 0

    while True:
        avg_len = average_edge_length(mesh)
        if avg_len <= target_edge_length or iterations >= 3:
            break
        mesh = mesh.subdivide_loop(number_of_iterations=1)
        iterations += 1

    # Optional: simplify slightly to stabilize triangle count
    mesh = mesh.simplify_quadric_decimation(int(len(mesh.triangles) * 0.9))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    np_new_verts = np.asarray(mesh.vertices)  # [N_new, 3]
    np_new_faces = np.asarray(mesh.triangles)  # [F_new, 3]

    # --- Step 2: GPU-based nearest neighbor transfer
    new_verts = torch.from_numpy(np_new_verts).to(device).to(dtype)  # [N_new, 3]
    new_faces = torch.from_numpy(np_new_faces).to(device).to(faces.dtype)  # [F_new, 3]

    # Compute pairwise distances [N_new, N_verts]
    dist = torch.cdist(new_verts[None], verts[None])  # [1, N_new, N_verts]
    nearest_idx = dist.argmin(dim=2).squeeze(0)  # [N_new]

    # Gather velocities [B, N_new, 3]
    new_vel = torch.gather(
        vel, 1, nearest_idx[None, :, None].expand(B, -1, 3)
    )  # [B, N_new, 3]

    return new_verts, new_faces, new_vel


def compute_mean_curvature(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute Discrete Mean Curvatures, the ground truth is refered as (1/R1 + 1/R2)
    For more details, please refer link: http://multires.caltech.edu/pubs/diffGeoOps.pdf
    Where we recommand the eq.7 from chapter 3.3, and eq.8 from chapter 3.5
    C
    / ^
    /  |
    <   |
    A--->B

    Args:
        vertices (torch.Tensor): Mesh vertices: sahhep = [..., NumOfVertices, 3]
        faces (torch.Tensor): Mesh Faces: shape = [..., NumOfFaces, 3]

    Returns:
        torch.Tensor: Discrete Mean Curvatures: shape = [..., NumOfVertices, 1]
    """
    face_vertices = torch.index_select(
        vertices, -2, torch.flatten(faces).to(torch.int64)
    )  # [...,  NumOfFaces*3, 3]
    edges = torch.zeros_like(face_vertices)  # [...,  NumOfFaces*3, 3]
    edges[..., 0::3, :] = (
        face_vertices[..., 1::3, :] - face_vertices[..., 0::3, :]
    )  # AB
    edges[..., 1::3, :] = (
        face_vertices[..., 2::3, :] - face_vertices[..., 1::3, :]
    )  # BC
    edges[..., 2::3, :] = (
        face_vertices[..., 0::3, :] - face_vertices[..., 2::3, :]
    )  # CA

    edges_norm = torch.norm(edges, p=2, dim=-1).unsqueeze(-1)  # [...,  NumOfFaces*3, 1]
    normalized_edges = F.normalize(edges, p=2, dim=-1)  # [...,  NumOfFaces*3, 3]
    cos_alphas = torch.zeros_like(edges_norm)  # [...,  NumOfFaces*3, 1]
    cos_alphas[..., 0::3, :] = (
        -(normalized_edges[..., 1::3, :] * normalized_edges[..., 2::3, :])
        .sum(-1)
        .unsqueeze(-1)
    )  # cosC
    cos_alphas[..., 1::3, :] = (
        -(normalized_edges[..., 2::3, :] * normalized_edges[..., 0::3, :])
        .sum(-1)
        .unsqueeze(-1)
    )  # cosA
    cos_alphas[..., 2::3, :] = (
        -(normalized_edges[..., 0::3, :] * normalized_edges[..., 1::3, :])
        .sum(-1)
        .unsqueeze(-1)
    )  # cosB
    cot_alphas = cos_alphas / torch.sqrt(1.0 - cos_alphas * cos_alphas)

    vert_voronoi_areas = torch.zeros_like(vertices[..., 0:1])  # [...,  NumOfVerts, 1]
    vert_mean_curvature = torch.zeros_like(vertices)  # [...,  NumOfVerts, 3]
    voronoi_areas_src = (
        0.125 * cot_alphas * edges_norm * edges_norm
    )  # [...,  NumOfFaces*3, 1]
    mean_curvature_src = cot_alphas * edges

    hlp_repeat_index = [1] * (len(faces.shape) - 1)

    # 1. Add to node A
    vert_voronoi_areas.scatter_add_(
        -2,
        faces[..., 0:1].to(torch.int64),
        voronoi_areas_src[..., 0::3, :] + voronoi_areas_src[..., 2::3, :],
    )  # C, B
    vert_mean_curvature.scatter_add_(
        -2,
        faces[..., 0:1].repeat(*hlp_repeat_index, 3).to(torch.int64),
        mean_curvature_src[..., 0::3, :] - mean_curvature_src[..., 2::3, :],
    )
    # 2. Add to node B
    vert_voronoi_areas.scatter_add_(
        -2,
        faces[..., 1:2].to(torch.int64),
        voronoi_areas_src[..., 0::3, :] + voronoi_areas_src[..., 1::3, :],
    )  # C, A
    vert_mean_curvature.scatter_add_(
        -2,
        faces[..., 1:2].repeat(*hlp_repeat_index, 3).to(torch.int64),
        mean_curvature_src[..., 1::3, :] - mean_curvature_src[..., 0::3, :],
    )
    # 3. Add to node C
    vert_voronoi_areas.scatter_add_(
        -2,
        faces[..., 2:3].reshape(*faces.shape[:-2], -1, 1).to(torch.int64),
        voronoi_areas_src[..., 1::3, :] + voronoi_areas_src[..., 2::3, :],
    )  # A, B
    vert_mean_curvature.scatter_add_(
        -2,
        faces[..., 2:3].repeat(*hlp_repeat_index, 3).to(torch.int64),
        mean_curvature_src[..., 2::3, :] - mean_curvature_src[..., 1::3, :],
    )
    vert_normals = compute_vert_normals(vertices=vertices, faces=faces)
    vert_mean_curvature = -(vert_mean_curvature * vert_normals).sum(dim=-1).unsqueeze(
        -1
    ) / (2.0 * vert_voronoi_areas)
    vert_mean_curvature = (
        vert_mean_curvature / 2.0
    )  # What we sovled is K.dot(n), which equals 2.0 * kappa, see page 4 from reference
    return vert_mean_curvature


def get_edge_unique(verts: torch.Tensor, faces: torch.Tensor) -> List[torch.Tensor]:
    """Get the unique edge from a mesh geometry

    face_edge_indices can be further utilized for scatter_add_
    [N_edges,].scatter_add_(..., src=[N_faces*dim,], index=face_edge_indices)

    Args:
        verts (torch.Tensor): [N_verts, dim]
        faces (torch.Tensor): [N_faces, dim]

    Returns:
        List[torch.Tensor]:
            edges: torch.Tensor = [N_edges, dim]
            face_edge_indices: torch.Tensor = [N_faces, dim]
            edge_adjacent_faces: torch.Tensor = [N_edges, 2]
    """
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape

    face_edges = torch.cat(
        (faces[..., 1:3], faces[..., 2:3], faces[..., 0:1], faces[..., 0:2]), dim=-1
    )  # [N_faces, 6]
    face_edges = face_edges.reshape(N_faces * dim, 2)  # [N_faces * dim, 2]
    face_edges_face_idx = (
        torch.linspace(0, N_faces - 1, N_faces, dtype=faces.dtype, device=faces.device)[
            :, None
        ]
        .repeat(1, 3)
        .reshape(N_faces * dim)
    )  # [N_faces * dim, 2]
    # Make the edges unique
    face_edges_flags = (
        face_edges.max(dim=-1)[0] * N_verts + face_edges.min(dim=-1)[0]
    )  # [N_faces * dim]
    edges, face_edges_indices = torch.unique(
        face_edges_flags, sorted=False, return_inverse=True
    )  # [N_edges], [N_faces * dim]
    edges = torch.stack((edges // N_verts, edges % N_verts), dim=-1)  # [N_edges, 2]
    # Note face_edges_indices can be served as the scatter index

    N_edges, _ = edges.shape
    edge_adjacent_faces = torch.zeros(
        (N_edges, 2), device=faces.device, dtype=faces.dtype
    )
    edge_adjacent_faces[..., 0].scatter_add_(
        dim=0, index=face_edges_indices, src=face_edges_face_idx
    )
    edge_adjacent_faces[..., 1].scatter_(
        dim=0, index=face_edges_indices, src=face_edges_face_idx
    )
    edge_adjacent_faces[..., 0] = (
        edge_adjacent_faces[..., 0] - edge_adjacent_faces[..., 1]
    )

    face_edges_indices = face_edges_indices.to(faces.dtype)

    return edges, face_edges_indices, edge_adjacent_faces


def div_Js(
    face_Js: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_areas: torch.Tensor = None,
    face_normals: torch.Tensor = None,
    keepdim: bool = True,
) -> torch.Tensor:
    """
    Compute approximate surface divergence of Js on each triangular face.

    Args:
        face_Js (torch.Tensor): [N_faces, 3], current density at vertices.
        verts (torch.Tensor):   [N_verts, 3], vertex coordinates.
        faces (torch.Tensor):   [N_faces, 3], vertex indices of each triangle.

    Returns:
        torch.Tensor: [N_faces, (1)], divergence values.
    """
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]

    if face_areas is None:
        face_areas = compute_face_areas(vertices=verts, faces=faces, keepdim=keepdim)

    if face_normals is None:
        face_normals = compute_face_normals(vertices=verts, faces=faces)

    # Edge vectors
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    # Edge lengths
    l0 = torch.linalg.norm(e0, dim=-1, keepdim=keepdim)  # opposite v0
    l1 = torch.linalg.norm(e1, dim=-1, keepdim=keepdim)  # opposite v1
    l2 = torch.linalg.norm(e2, dim=-1, keepdim=keepdim)  # opposite v2

    t0 = F.normalize(
        torch.cross(e0, face_normals, dim=-1), dim=-1
    )  # outward normal to edge opposite v0
    t1 = F.normalize(torch.cross(e1, face_normals, dim=-1), dim=-1)  # opposite v1
    t2 = F.normalize(torch.cross(e2, face_normals, dim=-1), dim=-1)  # opposite v2

    # Flux of Js across edges
    flux = (
        torch.sum(face_Js * t0, dim=1, keepdim=keepdim) * l0
        + torch.sum(face_Js * t1, dim=1, keepdim=keepdim) * l1
        + torch.sum(face_Js * t2, dim=1, keepdim=keepdim) * l2
    )

    # Divergence = flux / area
    div = flux / (face_areas)

    return div
