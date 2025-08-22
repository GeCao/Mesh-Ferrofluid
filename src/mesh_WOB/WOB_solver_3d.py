import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian
from matplotlib import cm
from matplotlib.colors import Normalize

from src.mesh_WOB import AbstractWOBSolver
from src.utils import (
    Logger,
    compute_face_areas,
    compute_face_normals,
    compute_vert_areas,
    compute_vert_normals,
    get_vertices_from_index,
    compute_panel_relation,
    UniformSampleHemisphereOnSurface,
    UniformSampleSphereOnSurface,
    UniformSampleCosHemisphereOnSurface,
    GetRayQueryDepthMap,
    get_edge_unique,
    G,
    gradG_y,
    LoadPointCloudFromMesh,
)


class MeshWOBSolver3d(AbstractWOBSolver):
    rank = 3

    def __init__(
        self,
        wavenumber: float,
        logger: Logger,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(MeshWOBSolver3d, self).__init__(
            wavenumber=wavenumber,
            logger=logger,
            dtype=dtype,
            device=device,
        )

    def solve_WOB(
        self,
        vulkan_ray_tracer,
        verts: torch.Tensor,
        faces: torch.Tensor,
        Einc_func,
        Hinc_func,
        max_depth: int = 10,
        N_samples: int = 1024,
    ) -> List[torch.Tensor]:
        """Mesh-based WOB solver: To solve Helmholtz problem
        Reference: Ryusuke Sugimoto, Terry Chen, Yiti Jiang, Christopher Batty, Toshiya Hachisuka, A Practical Walk-on-Boundary Method for Boundary Value Problems, 2023.
        https://arxiv.org/abs/2305.04403

        Args:
            verts (torch.Tensor): [N_verts, dim]
            faces (torch.Tensor): [N_faces, dim]
            max_depth (int): The maximum bounces per ray.
            N_samples (int): The number of samples per surface.

        Returns:
            torch.Tensor: [B, N_Verts, dim=3], The solved Js
        """
        N_verts, dim = verts.shape[-2:]
        N_faces, dim = faces.shape[-2:]
        batch_size = 1

        device = verts.device
        dtype = verts.dtype

        mesh = Meshes(verts=verts[None], faces=faces[None])

        face_areas = compute_face_areas(vertices=verts, faces=faces, keepdim=True)
        face_normals = compute_face_normals(vertices=verts, faces=faces)

        total_area = face_areas.sum().item()

        vert_Js = torch.zeros((batch_size, N_verts, dim), dtype=dtype, device=device)
        vert_Js = vert_Js + 1j * vert_Js

        x0 = torch.index_select(verts, dim=-2, index=faces.flatten().to(torch.int64))
        x0 = x0.reshape(N_faces, dim, dim).mean(dim=-2)  # [N_faces, dim]
        n0 = face_normals + 0  # [N_faces, dim]
        Js_index = faces.flatten()  # [N_faces * dim]
        hit_face_idx = torch.linspace(
            0, N_faces - 1, N_faces, dtype=faces.dtype, device=device
        )  # [N_faces,]

        # f_idx = 0
        # x0 = torch.index_select(
        #     verts, dim=-2, index=faces[f_idx].flatten().to(torch.int64)
        # )
        # x0 = x0.mean(dim=0, keepdim=True)  # [N_test, dim]
        # n0 = face_normals[f_idx, None]  # [N_test, dim]
        # Js_index = faces[f_idx].flatten()  # [N_test * dim]
        # hit_face_idx = f_idx + torch.zeros_like(x0[..., 0]).to(faces.dtype)  # [N_test,]

        N_test = x0.shape[0]
        N = N_test * N_samples
        xi = x0.reshape(N_test, 1, dim).repeat(1, N_samples, 1)
        ni = n0.reshape(N_test, 1, dim).repeat(1, N_samples, 1)
        xi = xi.reshape(N, dim)
        ni = ni.reshape(N, dim)
        hit_face_idx = hit_face_idx[:, None].repeat(1, N_samples).reshape(N, 1)
        valid = torch.ones((N, 1), dtype=torch.bool, device=device)
        lines = [xi + 0]
        ns = [ni + 0]
        pdfs = torch.zeros((N, 0), dtype=dtype, device=device)
        for depth in range(max_depth):
            # ray_d, pdf = UniformSampleCosHemisphereOnSurface(
            #     normal=-ni, N_sample_per_surface=1
            # )
            # ray_d = ray_d.reshape(N, dim)
            # pdf = pdf.reshape(N, 1)

            # Sampling points
            pts, pts_normals = LoadPointCloudFromMesh(meshes=mesh, num_pts_samples=N)
            pts = pts.reshape(N, dim)
            pts_normals = pts_normals.reshape(N, dim)
            ray_d = F.normalize(pts - xi, dim=-1)
            hit_depth = (pts - xi).norm(dim=-1, keepdim=True)
            pdf = (1 / total_area) + torch.zeros_like(hit_depth)
            if depth >= 1:
                pdf = pdf / N_samples  # Pretend you have N_samples contributions.

            # hit_depth, new_hit_face_idx = GetRayQueryDepthMap(
            #     vulkan_ray_tracer=vulkan_ray_tracer,
            #     ray_o=xi.reshape(N, dim),
            #     ray_d=ray_d.reshape(N, 1, dim),
            # )
            # hit_depth = hit_depth.reshape(N, 1)
            # new_hit_face_idx = new_hit_face_idx.reshape(N, 1)

            # print(f"hit_depth = {hit_depth}")
            # print(f"new hit_face_idx = {new_hit_face_idx}")

            # Valid only when hit point is detected
            # curr_valid = torch.logical_and(valid[:, -1:], new_hit_face_idx >= 0)
            # curr_valid = curr_valid & (hit_face_idx != new_hit_face_idx)
            curr_valid = valid[:, -1:] & (hit_depth > 1e-6)
            valid = torch.cat((valid, curr_valid), dim=-1)  # [N, depth + 1]

            # Record pdf
            pdfs = torch.cat((pdfs, pdf), dim=-1)

            hit_depth[hit_depth < 0] = 0  # As default
            # hit_face_idx[hit_face_idx < 0] = 0  # As default

            # Next round
            # xi = xi + hit_depth * ray_d  # [N, dim]
            # ni = torch.where(
            #     curr_valid.repeat(1, dim),
            #     torch.index_select(
            #         face_normals, dim=-2, index=hit_face_idx.flatten().to(torch.int64)
            #     ),
            #     torch.Tensor([[0.0, 0.0, 1.0]]).repeat(N, 1).to(dtype).to(device),
            # )  # [N, dim]
            xi = pts + 0
            ni = torch.where(
                curr_valid.repeat(1, dim),
                pts_normals,
                torch.Tensor([[0.0, 0.0, 1.0]]).repeat(N, 1).to(dtype).to(device),
            )  # [N, dim]
            # hit_face_idx = new_hit_face_idx + 0

            # Record lines
            lines.append(xi + 0)  # [N, dim]
            ns.append(ni + 0)  # [N, dim]

            self.InfoLog(
                f"[{depth}/{max_depth}] valid ray percentage = {curr_valid.sum() / curr_valid.numel() * 100}%"
            )

        lines = torch.stack(lines, dim=1)  # [N, depth + 1, dim]
        lines = lines.reshape(N_test, N_samples, max_depth + 1, dim)

        ns = torch.stack(ns, dim=1)  # [N, depth + 1, dim]
        ns = ns.reshape(N_test, N_samples, max_depth + 1, dim)

        pdfs = pdfs.reshape(N_test, N_samples, max_depth, 1)

        valid = valid.reshape(N_test, N_samples, max_depth + 1, 1)

        for depth in range(max_depth - 1, -1, -1):
            # [max_depth - 1, ..., 1, 0]
            x_prev = lines[..., depth, :]  # [N_test, N_samples, dim]
            x_after = lines[..., depth + 1, :]  # [N_test, N_samples, dim]

            n_prev = ns[..., depth, :]  # [N_test, N_samples, dim]
            n_after = ns[..., depth + 1, :]  # [N_test, N_samples, dim]

            valid_prev = valid[..., depth, :].repeat(1, 1, dim)
            valid_after = valid[..., depth + 1, :].repeat(1, 1, dim)

            pdf = pdfs[..., depth, :]  # [N_test, N_samples, 1]
            inv_pdf = torch.where(pdf > 1e-2, 1 / pdf, torch.zeros_like(pdf))

            if depth == max_depth - 1:
                # The last step.
                test_Js = torch.cross(
                    n_after + 1j * torch.zeros_like(n_after),
                    Hinc_func(self.wavenumber, x_after),
                    dim=-1,
                )  # [N_tests, N_samples, dim]
                test_Js[~valid_after] = 0

            # The recursive steps
            tmp_gradGy = gradG_y(wavenumber=self.wavenumber, p1=x_prev, p2=x_after)
            tmp_gradGx = -tmp_gradGy
            tmp_test_Js = torch.cross(test_Js, tmp_gradGx, dim=-1)
            # same_face = torch.logical_or(
            #     ((F.normalize(tmp_face_Js.real, dim=-1) * n_prev).sum(dim=-1) - 1).abs()
            #     < 0.1,
            #     ((F.normalize(tmp_face_Js.imag, dim=-1) * n_prev).sum(dim=-1) - 1).abs()
            #     < 0.1,
            # )
            # same_face = same_face[..., None].repeat(1, 1, dim)
            tmp_test_Js = torch.cross(
                tmp_test_Js, n_prev + 1j * torch.zeros_like(n_prev), dim=-1
            )
            tmp_test_Js = 2 * tmp_test_Js * inv_pdf
            tmp_test_Js = tmp_test_Js + 2 * torch.cross(
                n_prev + 1j * torch.zeros_like(n_prev),
                Hinc_func(self.wavenumber, x_prev),
                dim=-1,
            )  # [N_test, N_samples, dim]
            test_Js = torch.where(valid_prev, tmp_test_Js, test_Js)

            # if depth > 0:
            #     # release the pretended N_samples contribution
            #     test_Js = test_Js / N_samples

        # Take mean value
        test_Js = test_Js.mean(dim=-2)  # [N_test, dim]
        test_Js = test_Js.reshape(1, N_test, dim).repeat(
            batch_size, 1, 1
        )  # [B, N_test, dim]
        Js_src = (
            test_Js[..., None, :]
            .repeat(1, 1, dim, 1)
            .reshape(batch_size, N_test * dim, dim)
        )
        vert_Js.scatter_add_(
            dim=-2,
            index=Js_index.reshape(1, N_test * dim, 1)
            .repeat(batch_size, 1, dim)
            .to(torch.int64),
            src=Js_src / dim,
        )

        return vert_Js, lines, test_Js

    def vis_rays(
        self,
        save_path: str,
        ray_lines: torch.Tensor,
        verts: torch.Tensor,
        faces: torch.Tensor,
    ):
        N_verts, dim = verts.shape

        fig = plt.figure(figsize=(5, 4))
        axs = fig.add_subplot(1, 1, 1, projection="3d")

        ray_lines = torch.stack((ray_lines[..., :-1, :], ray_lines[..., 1:, :]), dim=-2)
        ray_lines = ray_lines.reshape(-1, ray_lines.shape[-2], dim)

        np_ray_lines = ray_lines.cpu().numpy()
        colors = np.linspace(0, ray_lines.shape[0] - 1, ray_lines.shape[0])
        norm = Normalize(vmin=0, vmax=ray_lines.shape[0] - 1)
        colors = cm.viridis(norm(colors))
        ray_lc = Line3DCollection(np_ray_lines, colors=colors, linewidths=0.1)
        axs.add_collection3d(ray_lc)

        edges, _, _ = get_edge_unique(verts=verts, faces=faces)
        edge_verts = torch.index_select(
            verts, dim=-2, index=edges.flatten().to(torch.int64)
        )
        edge_verts = edge_verts.reshape(edges.shape[-2], 2, dim)
        np_edge_verts = edge_verts.cpu().numpy()
        edge_lc = Line3DCollection(np_edge_verts, colors="black", linewidths=0.001)
        axs.add_collection3d(edge_lc)

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
        plt.savefig(
            os.path.join(save_path, "WOB_vis.png"), pad_inches=0, bbox_inches="tight"
        )
        plt.close()
