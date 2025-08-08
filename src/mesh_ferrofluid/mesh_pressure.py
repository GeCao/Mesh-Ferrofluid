import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian

from src.mesh_BEM import MeshBEMSolver3d
from src.utils import (
    compute_face_areas,
    compute_face_normals,
    compute_vert_normals,
    get_vertices_from_index,
    compute_mesh_vert_tangent_gradient,
    compute_mean_curvature,
    VertexType,
    BEMType,
    BoundaryType,
    Logger,
)


class MeshPressure(object):
    def __init__(
        self,
        density_fluid: float,
        gravity_strength: float,
        surface_tension: torch.Tensor,
        contact_angle: torch.Tensor,
        gaussQR: int,
        order_type: int,
        logger: Logger,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.gaussQR = gaussQR
        self.order_type = order_type
        self.logger = logger
        self.device = device
        self.dtype = dtype

        self.density_fluid = density_fluid
        self.gravity = torch.Tensor([0, -gravity_strength, 0]).to(device).to(dtype)
        self.surface_tension = surface_tension
        self.contact_angle = contact_angle

        self.BEM_solver = MeshBEMSolver3d(
            gaussQR=gaussQR,
            order_type=order_type,
            BEM_type=int(BEMType.LAPLACE),
            wavenumber=1.0,
            logger=self.logger,
            device=device,
            dtype=dtype,
        )

    def solve(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        dt: float,
        vel: torch.Tensor,
        vert_bc_flags: torch.Tensor,
        P_mag: torch.Tensor = None,
    ) -> torch.Tensor:
        """Mesh-based pressure solver
        Related algorithm can be refered by: https://dl.acm.org/doi/pdf/10.1145/3414685.3417799

        Args:
            verts (torch.Tensor): [..., N_verts, dim=3]
            faces (torch.Tensor): [..., N_faces, dim=3]
            dt (float): Time step
            vel (torch.Tensor): [..., N_verts, dim=3]
            vert_bc_flags (torch.Tensor): [..., N_verts, 1]
            P_mag (torch.Tensor): [..., N_verts, 1]

        Returns:
            torch.Tensor: [..., N_Verts, dim=3], The updated vertices
        """
        device = vel.device
        dtype = vel.dtype
        N_verts, dim = verts.shape[-2:]
        N_faces, dim = faces.shape[-2:]
        batch_size, N_verts, dim = vel.shape
        vel_solid = 0.0

        mesh = Meshes(verts=verts[None], faces=faces[None])

        density_fluid = self.density_fluid
        gravity = self.gravity
        surface_tension = self.surface_tension
        contact_angle = self.contact_angle

        face_areas = compute_face_areas(vertices=verts, faces=faces, keepdim=True)
        face_normals = compute_face_normals(vertices=verts, faces=faces)
        vert_normals = compute_vert_normals(vertices=verts, faces=faces)
        face_verts = get_vertices_from_index(verts, index=faces.flatten())
        face_verts = face_verts.reshape(N_faces, dim, dim)

        # 1. Pack the Dirichlet/Neumann boundary Conditions
        Neumann_vert_dpdn = (density_fluid / dt) * (
            (vel - vel_solid) * vert_normals
        ).sum(
            dim=-1, keepdim=True
        )  # [B, N_verts, 1]
        Neumann_vert_dpdn[vert_bc_flags == int(BoundaryType.DIRICHLET)] = 0

        # Combine the components for Dirichlet
        # The [gravity]
        Dirichlet_vert_p = -density_fluid * (gravity * verts).sum(dim=-1, keepdim=True)
        # The [Magnetic force]
        if P_mag is not None:
            Dirichlet_vert_p = Dirichlet_vert_p - P_mag
        # The [surface tension]
        surface_L, inv_areas = cot_laplacian(
            verts, faces.to(torch.int64)
        )  # [N_verts, N_verts] sparse matrix in ji index
        surface_L = surface_L.unsqueeze(-1)
        inv_areas = inv_areas.reshape(N_verts, 1)
        # dp_{i} = 0.5 * area_i * (cot_alpha_ji + cot_beta_ji) * (pj - pi)
        laplace_v = (
            0.5 * inv_areas * (surface_L * (verts[:, None, :] - verts)).sum(dim=0)
        )  # [N_verts, 3]
        laplace_v = laplace_v.to_dense()
        H = 0.5 * torch.norm(laplace_v, dim=-1)  # scalar mean curvature
        signs = torch.sign((laplace_v * vert_normals).sum(dim=1))
        kappa = H * signs  # signed mean curvature
        kappa = kappa.reshape(N_verts, 1)
        # kappa = compute_mean_curvature(vertices=verts, faces=faces)
        Dirichlet_vert_p = Dirichlet_vert_p + surface_tension * kappa
        Dirichlet_vert_p = Dirichlet_vert_p.reshape(1, N_verts, 1)
        Dirichlet_vert_p = Dirichlet_vert_p.repeat(batch_size, 1, 1)
        Dirichlet_vert_p[vert_bc_flags == int(BoundaryType.NEUMANN)] = 0

        # 2. Solve the poisson equation by BEM
        vert_dpdn, vert_p = self.BEM_solver.solve(
            verts=verts,
            faces=faces,
            vert_bc_Dirichlet=Dirichlet_vert_p,
            vert_bc_Neumann=Neumann_vert_dpdn,
            vert_bc_flags=vert_bc_flags,
            Amat_scaler=1.0,
        )  # [B, N_verts, 1], [B, N_verts, 1]

        # 3.1 Calculate the regular grad_p from pressure
        vert_gradp_normal = vert_dpdn * vert_normals  # [B, N_vertss, dim]
        vert_gradp_tangent = compute_mesh_vert_tangent_gradient(
            vert_p=vert_p, verts=verts, faces=faces, face_normals=face_normals
        )
        vert_gradp = vert_gradp_normal + vert_gradp_tangent
        print(f"dpdn = {vert_gradp_normal.norm(dim=-1).max()}")
        print(f"dpdt = {vert_gradp_tangent.norm(dim=-1).max()}")

        # 3.2 Calculate the triple junction grad_p from pressure
        triple_junction_detected = (
            vert_bc_flags == int(VertexType.TRIPLE_JUNCTION)
        ).sum().item() > 0
        if triple_junction_detected:
            # Triple junction (Air-Fluid-Obstacle) is not smooth
            self.ErrorLog("Not implemented for triple-junction points")

        new_vel = vel - (dt / density_fluid) * vert_gradp
        print(f"new_vel = {new_vel.min()}, {new_vel.max()}")
        # exit(0)

        return new_vel

    def InfoLog(self, *args, **kwargs):
        return self.logger.InfoLog(*args, **kwargs)

    def WarnLog(self, *args, **kwargs):
        return self.logger.WarnLog(*args, **kwargs)

    def ErrorLog(self, *args, **kwargs):
        return self.logger.ErrorLog(*args, **kwargs)
