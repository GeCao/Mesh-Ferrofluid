import os
import numpy as np
import torch
import imageio
import soft_renderer as sr
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List

from src.utils import compute_face_normals


def _render_mesh_to_file_2d(
    img_dir: str,
    file_label: str,
    verts: torch.Tensor,
    faces: torch.Tensor,
    vert_phi: torch.Tensor,
    show_normals: bool = True,
):
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape
    assert dim == 2

    face_normals = None
    if show_normals and face_normals is None:
        face_normals = compute_face_normals(vertices=verts, faces=faces)

    if vert_phi is None:
        batch_size, n_ch = 1, dim
        # Use face normal as color
        if face_normals is None:
            face_normals = compute_face_normals(vertices=verts, faces=faces)
        textures = torch.stack(
            (
                face_normals[..., 0],
                torch.zeros_like(face_normals[..., 1]),
                face_normals[..., 1],
            ),
            dim=-1,
        )  # [N_faces, 3]
        textures = textures[None, :, :].abs()  # [B, N_faces, 3]

        np_phi_min = np.zeros((batch_size,), dtype=np.float32)
        np_phi_max = np.ones((batch_size,), dtype=np.float32)
        np_textures = textures.cpu().numpy()  # [B, N_faces, 3]
    else:
        batch_size, N_verts, n_ch = vert_phi.shape
        face_vert_phi = torch.index_select(
            vert_phi, dim=-2, index=faces.flatten().to(torch.int64)
        )  # [B, N_faces * dim, n_ch]
        face_phi = face_vert_phi.reshape(batch_size, N_faces, dim, n_ch).sum(dim=-2)

        if n_ch == dim:
            # Simply use it as R0B
            phi_max, _ = torch.max(face_phi.abs().reshape(batch_size, -1), dim=1)
            textures = torch.stack(
                (
                    face_phi[..., 0],
                    torch.zeros_like(face_phi[..., 1]),
                    face_phi[..., 1],
                ),
                dim=-1,
            )  # [B, N_faces, 3]
            textures = textures.abs() / phi_max.reshape(batch_size, 1, 1)

            np_phi_min = np.zeros((batch_size,), dtype=np.float32)
            np_phi_max = phi_max.flatten().cpu().numpy()
            np_textures = textures.cpu().numpy()  # [B, N_faces, 3]
        elif n_ch == 1:
            # Map it with viridis colormap
            phi_min, _ = torch.min(face_phi, dim=1)  # [B, n_ch=1]
            phi_max, _ = torch.max(face_phi, dim=1)  # [B, n_ch=1]
            face_rgb = (face_phi.squeeze(-1) - phi_min) / (phi_max - phi_min)
            np_face_rgb = face_rgb.cpu().numpy()  # [B, N_faces]
            np_textures = cm.viridis(np_face_rgb.flatten())

            np_phi_min = phi_min.flatten().cpu().numpy()
            np_phi_max = phi_max.flatten().cpu().numpy()
            np_textures = np_textures.reshape(batch_size, N_faces, 4)
        else:
            raise NotImplementedError(
                "We only accpet n_ch = 1/3 for visualization of phi on Mesh"
            )

    # Prepare 2D Mesh (lines) for plt
    face_verts = torch.index_select(
        input=verts, dim=-2, index=faces.flatten().to(torch.int64)
    )  # [N_faces * dim, dim]
    face_verts = face_verts.reshape(N_faces, dim, dim)
    np_lines = face_verts.cpu().numpy()

    # Prepare xlim and ylim for plt
    np_AABB_min = face_verts.reshape(-1, dim).min(dim=0)[0].cpu().numpy()
    np_AABB_max = face_verts.reshape(-1, dim).max(dim=0)[0].cpu().numpy()
    np_AABB_len = np_AABB_max - np_AABB_min
    np_AABB_eps = np_AABB_len * 0.1
    np_AABB_eps[np_AABB_eps == 0] = np_AABB_eps.max()

    # Prepare 2D normals (lines) for plt
    if show_normals:
        face_centers = face_verts.mean(dim=-2)  # [N_faces, dim]
        len_normals = float(np_AABB_eps.max()) * 0.1
        np_face_centers = face_centers.cpu().numpy()
        np_scaled_face_normals = len_normals * face_normals.cpu().numpy()

    filepath_list = []
    for batch_idx in range(batch_size):
        figure, axes = plt.subplots(nrows=1, ncols=1)
        lc = LineCollection(np_lines, colors=np_textures[batch_idx], linewidths=4)
        axes.add_collection(lc)

        if show_normals:
            plt.quiver(
                np_face_centers[..., 0],
                np_face_centers[..., 1],
                np_scaled_face_normals[..., 0],
                np_scaled_face_normals[..., 1],
                angles="xy",
                scale_units="xy",
                scale=0.1,
                color="black",
            )

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(np_AABB_min[0] - np_AABB_eps[0], np_AABB_max[0] + np_AABB_eps[0])
        plt.ylim(np_AABB_min[1] - np_AABB_eps[1], np_AABB_max[1] + np_AABB_eps[1])
        plt.title(f"phi")
        # plt.legend()

        # Add color bar
        if n_ch == 1:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            norm = Normalize(vmin=np_phi_min[batch_idx], vmax=np_phi_max[batch_idx])
            sm = cm.ScalarMappable(norm=norm, cmap="binary")
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label("phi")

        if batch_size == 1:
            filepath = os.path.join(img_dir, f"{file_label}.png")
        else:
            filepath = os.path.join(img_dir, f"{file_label}_{batch_idx}.png")

        plt.savefig(filepath, pad_inches=0.05, bbox_inches="tight")
        filepath_list.append(filepath)
        plt.close()

    return filepath_list


def _render_mesh_to_file_3d(
    img_dir: str,
    file_label: str,
    verts: torch.Tensor,
    faces: torch.Tensor,
    vert_phi: torch.Tensor,
    show_normals: bool = True,
    rasterizer: sr.SoftRasterizer = None,
    eye: List[float] = [0, 0, -3],
) -> List[str]:
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape
    assert dim == 3

    fov = 30  # degree
    transform = sr.LookAt(eye=eye, viewing_angle=fov)
    lighting = sr.Lighting()

    if vert_phi is None:
        # Use face normal as color
        face_normals = compute_face_normals(vertices=verts, faces=faces)
        textures = face_normals[None, :, None, :].abs()  # [B, N_faces, 1, n_ch=3]
        batch_size, N_faces, _, n_ch = textures.shape
        texture_type = "surface"
    else:
        batch_size, N_verts, n_ch = vert_phi.shape
        if n_ch == 3:
            # Simply use it as RGB
            phi_max, _ = torch.max(vert_phi.abs().reshape(batch_size, -1), dim=1)
            textures = vert_phi.abs() / phi_max.reshape(batch_size, 1, 1)
        elif n_ch == 1:
            # Map it with viridis colormap
            phi_min, _ = torch.min(vert_phi.reshape(batch_size, -1), dim=1)
            phi_min = phi_min.reshape(batch_size, 1, 1)
            phi_max, _ = torch.max(vert_phi.reshape(batch_size, -1), dim=1)
            phi_max = phi_max.reshape(batch_size, 1, 1)
            vert_rgb = (vert_phi - phi_min) / (phi_max - phi_min)
            vert_r = torch.clamp(1.5 - torch.abs(4 * vert_rgb - 3), 0, 1)
            vert_g = torch.clamp(1.5 - torch.abs(4 * vert_rgb - 2), 0, 1)
            vert_b = torch.clamp(1.5 - torch.abs(4 * vert_rgb - 1), 0, 1)
            textures = torch.cat((vert_r, vert_g, vert_b), dim=-1)
        else:
            raise NotImplementedError(
                "We only accpet n_ch = 1/3 for visualization of phi on Mesh"
            )
        texture_type = "vertex"

    filepath_list = []
    for batch_idx in range(batch_size):
        mesh_ = sr.Mesh(
            vertices=verts,
            faces=faces,
            textures=textures[batch_idx],
            texture_type=texture_type,
        )

        # render mesh
        mesh = lighting(mesh_)
        mesh = transform(mesh)
        images = rasterizer(mesh)  # [B, n_ch, H, W]

        np_image = images[0].permute((1, 2, 0)).detach().cpu().numpy() * 255
        if batch_size == 1:
            filepath = os.path.join(img_dir, f"{file_label}.png")
        else:
            filepath = os.path.join(img_dir, f"{file_label}_{batch_idx}.png")

        imageio.imwrite(filepath, np_image.astype(np.uint8))
        filepath_list.append(filepath)

    return filepath_list


def render_mesh_to_file(
    img_dir: str,
    file_label: str,
    verts: torch.Tensor,
    faces: torch.Tensor,
    vert_phi: torch.Tensor = None,
    rasterizer: sr.SoftRasterizer = None,
    eye: List[float] = [0, 0, 3],
    show_normals: bool = True,
) -> List[str]:
    """Render 2d/3d Mesh to file,
    If the texture has not be assigned, we will use normals instead.

    Args:
        img_dir (str): The directory of the images
        verts (torch.Tensor): [N_verts, dim]
        faces (torch.Tensor): [N_faces, dim]
        vert_phi (torch.Tensor): [B, N_verts, 1/dim]
        rasterizer (sr.SoftRasterizer): An initialized differentiable rasterizer, only for 3D
    Returns:
        str: the rendered files
    """
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape

    if dim == 2:
        return _render_mesh_to_file_2d(
            img_dir=img_dir,
            file_label=file_label,
            verts=verts,
            faces=faces,
            vert_phi=vert_phi,
            show_normals=show_normals,
        )
    elif dim == 3:
        return _render_mesh_to_file_3d(
            img_dir=img_dir,
            file_label=file_label,
            verts=verts,
            faces=faces,
            vert_phi=vert_phi,
            show_normals=show_normals,
            rasterizer=rasterizer,
            eye=eye,
        )


def save_heatmap(
    tensor_input: torch.Tensor,
    filename: str,
    title: str,
    mask: torch.Tensor = None,
    vmax: float = None,
):
    """Save 2D heatmap

    Args:
        tensor_input (torch.Tensor): [H, W, (1)]
        filename (str): file name to be saved
        title (str): title of this plot
        mask (torch.Tensor, optional): [H, W, (1)]. Defaults to None.
        vmax (float, optional): used for cmap. Defaults to None.
    """
    if mask is not None:
        tensor_input[mask] = torch.inf
    np_img = tensor_input.cpu().numpy()

    # np_img = [H, W, C]
    fig, ax = plt.subplots(1, 1)
    # plt.pcolormesh(new_gains)
    plt.imshow(np_img, origin="lower", cmap="viridis", vmax=vmax)
    plt.colorbar()
    plt.axis("off")
    plt.title(title)
    plt.savefig(filename, pad_inches=0, bbox_inches="tight")
    plt.close()


def vis_vert_Js(save_path: str, vert_Js: torch.Tensor, verts: torch.Tensor):
    """Visualize (3D) surface current density (Js)
    Args:
        save_path: str
        vert_Js: torch.Tensor = [N_verts, dim]
        verts: torch.Tensor = [N_verts, dim]
    """
    N_verts, dim = verts.shape

    fig = plt.figure(figsize=(5, 4))
    axs = fig.add_subplot(1, 1, 1, projection="3d")

    # edges, _, _ = get_edge_unique(verts=verts, faces=faces)
    # edge_verts = torch.index_select(
    #     verts, dim=-2, index=edges.flatten().to(torch.int64)
    # )
    # edge_verts = edge_verts.reshape(edges.shape[-2], 2, dim)
    # np_edge_verts = edge_verts.cpu().numpy()
    # edge_lc = Line3DCollection(np_edge_verts, colors="black", linewidths=0.001)
    # axs.add_collection3d(edge_lc)

    np_verts = verts.cpu().numpy()
    np_color = vert_Js.real.norm(dim=-1).cpu().numpy()
    im = axs.scatter(
        np_verts[:, 0],
        np_verts[:, 1],
        np_verts[:, 2],
        cmap="viridis",
        s=3,
        c=np_color,
    )

    # colorbar
    clb = plt.colorbar(im)
    clb.ax.set_title(r"$J_{s}$")

    axs.set_title(f"ray tracing inside a mesh")
    axs.set_xlabel(f"X")
    axs.set_ylabel(f"Y")
    axs.set_zlabel(f"Z")
    # axs.set_xlim(-0.1, 0.1)
    # axs.set_ylim(-0.1, 0.1)
    # axs.set_zlim(-0.1, 0.1)
    # axs.set_xticklabels([])
    # axs.set_yticklabels([])
    # axs.set_zticklabels([])
    axs.axis("equal")
    # axs.axis("off")
    plt.savefig(save_path, pad_inches=0.2, bbox_inches="tight")
    plt.close()
