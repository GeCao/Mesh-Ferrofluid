import torch
from pytorch3d.structures import Meshes

from src.utils import (
    get_gaussion_integration_points_and_weights,
    single_integration_on_panels_3d,
    compute_face_areas,
    compute_face_normals,
    compute_vert_normals,
    get_vertices_from_index,
    grad_G_y_3d,
    linear_interplate_from_unit_panel_to_general,
    BEMType,
    OrderType,
)


class MeshHelmholtzDecomposition(object):
    def __init__(
        self,
        gaussQR: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.gaussQR = gaussQR
        self.device = device
        self.dtype = dtype

        self.gaussian_points_1d, self.gaussian_weights_1d = (
            get_gaussion_integration_points_and_weights(
                gaussQR, device=device, dtype=dtype
            )
        )

        gaussQR2 = gaussQR * gaussQR

        # Generate number(r1, r2)
        r1 = self.gaussian_points_1d[None, :]
        r2 = self.gaussian_points_1d[:, None] * r1
        r1 = r1.repeat(gaussQR, 1)

        r1 = r1.reshape((gaussQR2, 1))
        r2 = r2.reshape((gaussQR2, 1))

        weights = self.gaussian_weights_1d * self.gaussian_weights_1d[:, None]
        weights = weights.reshape((gaussQR2, 1))  # [gaussQR2, 1]

        self.r1 = r1
        self.r2 = r2
        self.gauss_weights = weights

    def solve(
        self, verts: torch.Tensor, faces: torch.Tensor, vert_vel: torch.Tensor
    ) -> torch.Tensor:
        """Mesh-based Helmholtz Decomposition
        Related algorithm can be refered by: https://dl.acm.org/doi/pdf/10.1145/3414685.3417799

        Args:
            verts (torch.Tensor): [..., N_verts, dim=3]
            faces (torch.Tensor): [..., N_faces, dim=3]
            vel (torch.Tensor): [..., N_verts, dim=3]

        Returns:
            torch.Tensor: [..., N_Verts, dim=3], The updated vertices
        """
        device = vert_vel.device
        dtype = vert_vel.dtype
        N_verts, dim = verts.shape[-2:]
        N_faces, dim = faces.shape[-2:]
        batch_size, N_verts, dim = vert_vel.shape

        face_areas = compute_face_areas(vertices=verts, faces=faces, keepdim=True)
        face_normals = compute_face_normals(vertices=verts, faces=faces)
        vert_normals = compute_vert_normals(vertices=verts, faces=faces)
        face_verts = get_vertices_from_index(verts, index=faces.flatten())
        face_verts = face_verts.reshape(N_faces, dim, dim)
        face_vel = torch.index_select(vert_vel, dim=-2, index=faces.flatten())
        face_vel = face_vel.reshape(batch_size, N_faces, dim, dim)

        # 1. calculate: u1 = -nabla(phi)
        integrand_weights = single_integration_on_panels_3d(
            dim=dim,
            order_type=int(OrderType.PLANAR),
            face_areas=face_areas,
            gauss_weights=self.gauss_weights,
            r1=self.r1,
            r2=self.r2,
        )  # [N_faces, 1, GaussQR2, 1]

        y1 = face_verts[..., 0, :].reshape(N_faces, 1, 1, dim)
        y2 = face_verts[..., 1, :].reshape(N_faces, 1, 1, dim)
        y3 = face_verts[..., 2, :].reshape(N_faces, 1, 1, dim)
        y = linear_interplate_from_unit_panel_to_general(
            r1=self.r1, r2=self.r2, x1=y1, x2=y2, x3=y3
        )  # [N_faces, 1, GaussQR2, dim]
        G_gradx = grad_G_y_3d(
            x=y, y=verts.reshape(N_verts, 1, 1, 1, dim), BEM_type=int(BEMType.LAPLACE)
        )  # [N_verts, N_faces, 1, GaussQR2, dim]

        face_vel1 = face_vel[..., 0, :].reshape(batch_size, 1, N_faces, 1, 1, dim)
        face_vel2 = face_vel[..., 1, :].reshape(batch_size, 1, N_faces, 1, 1, dim)
        face_vel3 = face_vel[..., 2, :].reshape(batch_size, 1, N_faces, 1, 1, dim)
        uy = linear_interplate_from_unit_panel_to_general(
            r1=self.r1, r2=self.r2, x1=face_vel1, x2=face_vel2, x3=face_vel3
        )  # [B, 1, N_faces, 1, GaussQR2, dim]

        n_dot_vel = (face_normals[:, None, None, :] * uy).sum(
            dim=-1, keepdim=True
        )  # [N_faces, 1, GaussQR2, 1]
        u1 = (
            (integrand_weights * n_dot_vel * G_gradx).sum(dim=2).sum(dim=-2)
        )  # [B, N_verts, 1, dim]
        u1 = u1.reshape(batch_size, N_verts, dim)

        # 2. calculate: u2 = nabla \times A
        n_cross_vel = torch.cross(
            face_normals[None, None, :, None, None, :].repeat(
                batch_size, 1, 1, 1, self.gaussQR**2, 1
            ),
            uy,
            dim=-1,
        )  # [B, 1, N_faces, 1, GaussQR2, dim]
        n_cross_vel_cross_Ggradx = torch.cross(
            n_cross_vel.repeat(1, N_verts, 1, 1, 1, 1),
            G_gradx[None].repeat(batch_size, 1, 1, 1, 1, 1),
            dim=-1,
        )  # [B, N_verts, N_faces, 1, GaussQR2, dim]
        u2 = (
            (n_cross_vel_cross_Ggradx * integrand_weights).sum(dim=2).sum(dim=-2)
        )  # [B, N_verts, 1, dim]
        u2 = u2.reshape(batch_size, N_verts, dim)

        # 3. Update according to Da et al, 2016 (explicit-format)
        # In Da et al, 2020, they tried implicit-format for this.
        vert_un = (vert_normals * (u1 + u2)).sum(
            dim=-1, keepdim=True
        )  # [B, N_verts, 1]

        I_mat = torch.eye(dim, device=device, dtype=dtype)  # [3, 3]
        P_mat = I_mat - vert_normals[:, :, None] @ vert_normals[:, None, :]
        new_vel = (P_mat @ vert_vel[..., None]).squeeze(-1) + vert_un * vert_normals

        return new_vel
