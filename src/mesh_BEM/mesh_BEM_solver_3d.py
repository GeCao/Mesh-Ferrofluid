import torch
from typing import List
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian

from src.mesh_BEM import AbstractBEMSolver, SingleLayer3d, DoubleLayer3d, RhsLayer3d
from src.utils import (
    Logger,
    compute_face_areas,
    compute_face_normals,
    compute_vert_areas,
    get_vertices_from_index,
    compute_panel_relation,
    BoundaryType,
    VertexType,
    BEMType,
)


class MeshBEMSolver3d(AbstractBEMSolver):
    rank = 3

    def __init__(
        self,
        gaussQR: int,
        order_type: int,
        BEM_type: int,
        wavenumber: float,
        logger: Logger,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(MeshBEMSolver3d, self).__init__(
            gaussQR=gaussQR,
            order_type=order_type,
            BEM_type=BEM_type,
            wavenumber=wavenumber,
            logger=logger,
            dtype=dtype,
            device=device,
        )

        self.single_layer = SingleLayer3d(
            gaussQR=self.gaussQR,
            order_type=self.order_type,
            BEM_type=self.BEM_type,
            wavenumber=self.wavenumber,
            logger=self.logger,
            device=device,
            dtype=dtype,
        )

        self.double_layer = DoubleLayer3d(
            gaussQR=self.gaussQR,
            order_type=self.order_type,
            BEM_type=self.BEM_type,
            wavenumber=self.wavenumber,
            logger=self.logger,
            device=device,
            dtype=dtype,
        )

        self.rhs_layer = RhsLayer3d(
            parent=self,
            gaussQR=self.gaussQR,
            order_type=self.order_type,
            BEM_type=self.BEM_type,
            wavenumber=self.wavenumber,
            logger=self.logger,
            device=device,
            dtype=dtype,
        )

    def solve(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        vert_bc_Dirichlet: torch.Tensor,
        vert_bc_Neumann: torch.Tensor,
        vert_bc_flags: torch.Tensor,
        Amat_scaler: float = None,
    ) -> List[torch.Tensor]:
        """Mesh-based BEM solver
        Reference: S. A. Sauter and C. Schwab, Boundary Element Methods, Springer Ser. Comput. Math. 39, Springer-Verlag, Berlin, 2011.
        https://link.springer.com/content/pdf/10.1007/978-3-540-68093-2.pdf

        Args:
            verts (torch.Tensor): [..., N_verts, dim]
            faces (torch.Tensor): [..., N_faces, dim]
            vert_bc_Dirichlet (torch.Tensor): [B, N_verts, 1]
            vert_bc_Neumann (torch.Tensor): [B, N_verts, 1]
            vert_bc_flags (torch.Tensor): [B, N_verts, 1]

        Returns:
            List[torch.Tensor]:
                [B, N_Verts, dim=3], The solved unknown Neumann (Dirichlet bc)
                [B, N_Verts, dim=3], The solved unknown Dirichlet (Neumann bc)
        """
        N_verts, dim = verts.shape[-2:]
        N_faces, dim = faces.shape[-2:]
        batch_size, N_verts, _ = vert_bc_flags.shape

        device = verts.device
        dtype = verts.dtype

        face_areas = compute_face_areas(vertices=verts, faces=faces, keepdim=True)
        face_normals = compute_face_normals(vertices=verts, faces=faces)
        vert_areas = compute_vert_areas(vertices=verts, faces=faces)
        face_verts = get_vertices_from_index(verts, index=faces.flatten())
        face_verts = face_verts.reshape(N_faces, dim, dim)
        panel_relations = compute_panel_relation(faces=faces)

        solved_Neumann_list = []
        solved_Dirichlet_list = []
        for batch_idx in range(batch_size):
            vert_is_Dirichlet = vert_bc_flags[batch_idx] == int(BoundaryType.DIRICHLET)
            vert_is_Neumann = vert_bc_flags[batch_idx] == int(BoundaryType.NEUMANN)

            N_GammaD = vert_is_Dirichlet.sum()
            N_GammaN = vert_is_Neumann.sum()

            self.InfoLog(
                f"Batch [{batch_idx}]: (Vertex attached) number of Dirichlet B.C. = {N_GammaD}, "
                f"number of Neumann B.C. = {N_GammaN}, "
                f"number of whole zone = {vert_bc_flags[batch_idx].numel()}"
            )

            if N_GammaD > 0:
                self.single_layer.compute_Vmat(
                    verts=verts,
                    faces=faces,
                    face_areas=face_areas,
                    face_normals=face_normals,
                    panel_relations=panel_relations,
                )
                V_mat = self.single_layer.get_Vmat_tight(verts=verts, faces=faces)

            self.double_layer.compute_Kmat(
                verts=verts,
                faces=faces,
                face_areas=face_areas,
                face_normals=face_normals,
                panel_relations=panel_relations,
            )
            K_mat = self.double_layer.get_Kmat_tight(verts=verts, faces=faces)

            if N_GammaN > 0 and self.BEM_type != int(BEMType.HELMHOLTZ_MOM):
                self.hypersingular_layer.compute_Dmat(
                    verts=verts,
                    faces=faces,
                    face_areas=face_areas,
                    face_normals=face_normals,
                    panel_relations=panel_relations,
                )
                D_mat = self.hypersingular_layer.get_Dmat_tight(
                    verts=verts, faces=faces
                )

            self.rhs_layer.compute_rhs(
                verts=verts,
                faces=faces,
                vert_bc_Dirichlet=vert_bc_Dirichlet,
                vert_bc_Neumann=vert_bc_Neumann,
                vert_bc_flags=vert_bc_flags,
            )
            rhs_vec = self.rhs_layer.get_rhs_vec().flatten()

            if N_GammaD != 0 and N_GammaN == 0:
                # Full Dirichlet B.C.
                A_mat = V_mat
            elif N_GammaD == 0 and N_GammaN != 0:
                # Full Neumann B.C.
                if self.BEM_type == int(BEMType.HELMHOLTZ_MOM):
                    A_mat = K_mat  # For MOM only
                    A_mat[
                        (
                            [i for i in range(K_mat.shape[0])],
                            [i for i in range(K_mat.shape[1])],
                        )
                    ] = -0.5 * vert_areas.flatten().to(K_mat.dtype)
                else:
                    A_mat = D_mat  # For BEM
            elif N_GammaD != 0 and N_GammaN != 0:
                # Mix Dirichlet/Neumann B.C.
                V_mask = torch.logical_and(
                    vert_bc_Dirichlet.reshape(N_verts, 1).repeat(1, N_verts),
                    vert_bc_Dirichlet.reshape(1, N_verts).repeat(N_verts, 1),
                )  # Dx Dy
                K_mask = torch.logical_and(
                    vert_bc_Dirichlet.reshape(N_verts, 1).repeat(1, N_verts),
                    vert_bc_Neumann.reshape(1, N_verts).repeat(N_verts, 1),
                )  # Dx, Ny
                D_mask = torch.logical_and(
                    vert_bc_Neumann.reshape(N_verts, 1).repeat(1, N_verts),
                    vert_bc_Neumann.reshape(1, N_verts).repeat(N_verts, 1),
                )  # Nx Ny

                # Note: Our V/K/D matrix All in sequence of [N_faces_y, N_faces_x]
                # This is all fine for V_mat and D_mat, as they are axisymmetric.
                # However, we have to transpose K_mat to make it works as [N_faces_x, N_faces_y]
                # -------------------   ------   ------
                # | Dx Dy |  Dx Ny  |   | Dy |   | Dx |
                # |-------|---------|   |----|   |----|
                # | Nx Dy |  Nx Ny  | @ | Ny | = | Nx |
                # |       |         |   |    |   |    |
                # |-------|---------|   |----|   |----|
                #        Amat           solved     rhs
                V_mat_masked = V_mat[V_mask].reshape(N_GammaD, N_GammaD)
                K_mat_masked = K_mat.permute((1, 0))[K_mask].reshape(N_GammaD, N_GammaN)
                D_mat_masked = D_mat[D_mask].reshape(N_GammaN, N_GammaN)

                A_mat = torch.cat(
                    (
                        torch.cat((V_mat_masked, -K_mat_masked), dim=1),
                        torch.cat((K_mat_masked.permute((1, 0)), D_mat_masked), dim=1),
                    ),
                    dim=0,
                )  # [D + N, D + N]

            if Amat_scaler is not None:
                A_mat = Amat_scaler * A_mat

            this_solved = torch.linalg.solve(A_mat, rhs_vec)
            if N_GammaD != 0 and N_GammaN == 0:
                # Dirichlet B.C., try to solve Neumann
                solved_Neumann_list.append(this_solved.reshape(N_verts, 1))
                solved_Dirichlet_list.append(
                    vert_bc_Dirichlet[batch_idx].reshape(N_verts, 1)
                )
            elif N_GammaD == 0 and N_GammaN != 0:
                # Neumann B.C., try to solve Dirichlet
                solved_Neumann_list.append(
                    vert_bc_Neumann[batch_idx].reshape(N_verts, 1)
                )
                solved_Dirichlet_list.append(this_solved.reshape(N_verts, 1))
            elif N_GammaD != 0 and N_GammaN != 0:
                # Mix B.C., try to solve both Neumann and Dirichlet
                vert_bc_Neumann[batch_idx][vert_is_Dirichlet] = this_solved[0:N_GammaD]
                vert_bc_Dirichlet[batch_idx][vert_is_Neumann] = this_solved[N_GammaD:]
                solved_Neumann_list.append(
                    vert_bc_Neumann[batch_idx].reshape(N_verts, 1)
                )
                solved_Dirichlet_list.append(
                    vert_bc_Dirichlet[batch_idx].reshape(N_verts, 1)
                )

        solved_Neumann = torch.stack(solved_Neumann_list, dim=0)
        solved_Dirichlet = torch.stack(solved_Dirichlet_list, dim=0)

        return solved_Neumann, solved_Dirichlet
