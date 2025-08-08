import torch
from typing import List, Tuple

from src.utils import (
    PanelRelationType,
    LayerType,
    OrderType,
    BEMType,
    get_vertices_from_index,
    shape_function,
    linear_interplate_from_unit_panel_to_general,
    G_3d,
    grad_G_y_3d,
    compute_mesh_face_curl,
)


def get_rands_and_jacobian_3d(
    xsi: torch.Tensor, eta1: torch.Tensor, eta2: torch.Tensor, eta3: torch.Tensor
) -> Tuple[List[torch.Tensor]]:
    """Get r1, r2 based on panel relations

    Args:
        xsi (torch.Tensor): [GaussQR2, 1]
        eta1 (torch.Tensor): [GaussQR2, 1]
        eta2 (torch.Tensor): [GaussQR2, 1]
        eta3 (torch.Tensor): [GaussQR2, 1]

    Returns:
        Tuple[List[torch.Tensor]]:
            r1_x: [r1_x_separate, r1_x_same_vertex, r1_x_same_edge, r1_x_same_face]
            r2_x: ...
            r1_y: ...
            r2_y: ...
            jacobian: ...
        Every single Tensor are in shape of [GaussQR2, N_Ds]
    """
    xsi = xsi.reshape(-1, 1)
    eta1 = eta1.reshape(-1, 1)
    eta2 = eta2.reshape(-1, 1)
    eta3 = eta3.reshape(-1, 1)

    r1_x = [None for i in range(4)]
    r2_x = [None for i in range(4)]
    r1_y = [None for i in range(4)]
    r2_y = [None for i in range(4)]
    jacobian = [None for i in range(4)]

    # 0. In default, the panels are all separated.
    rands_separate = torch.cat((xsi, eta1 * xsi, eta2, eta2 * eta3), dim=-1)
    jacobian_separate = xsi * eta2
    r1_x[int(PanelRelationType.SEPARATE)] = rands_separate[..., 0:1]
    r2_x[int(PanelRelationType.SEPARATE)] = rands_separate[..., 1:2]
    r1_y[int(PanelRelationType.SEPARATE)] = rands_separate[..., 2:3]
    r2_y[int(PanelRelationType.SEPARATE)] = rands_separate[..., 3:4]
    jacobian[int(PanelRelationType.SEPARATE)] = jacobian_separate

    # 1. Sometimes, these two panels might share the same vertex
    rands_samevertex_D1 = torch.cat(
        (xsi, xsi * eta1, xsi * eta2, xsi * eta2 * eta3), dim=-1
    )
    jacobian_samevertex_D1 = xsi * xsi * xsi * eta2

    rands_samevertex_D2 = torch.cat(
        (xsi * eta2, xsi * eta2 * eta3, xsi, xsi * eta1), dim=-1
    )
    jacobian_samevertex_D2 = xsi * xsi * xsi * eta2

    rands_samevertex = torch.stack(
        (rands_samevertex_D1, rands_samevertex_D2), dim=-2
    )  # [GaussQR2, 2, 4]
    jacobian_samevertex = torch.cat(
        (jacobian_samevertex_D1, jacobian_samevertex_D2), dim=-1
    )  # [GaussQR2, 2]
    r1_x[int(PanelRelationType.SAME_VERTEX)] = rands_samevertex[..., 0]
    r2_x[int(PanelRelationType.SAME_VERTEX)] = rands_samevertex[..., 1]
    r1_y[int(PanelRelationType.SAME_VERTEX)] = rands_samevertex[..., 2]
    r2_y[int(PanelRelationType.SAME_VERTEX)] = rands_samevertex[..., 3]
    jacobian[int(PanelRelationType.SAME_VERTEX)] = jacobian_samevertex

    # 2. Sometimes, these two panels might share the same edge
    w_sameedge_D1 = torch.cat(
        (xsi, -xsi * eta1 * eta2, xsi * eta1 * (1.0 - eta2), xsi * eta1 * eta3),
        dim=-1,
    )
    rands_sameedge_D1 = torch.stack(
        (
            w_sameedge_D1[..., 0],
            w_sameedge_D1[..., 3],
            w_sameedge_D1[..., 0] + w_sameedge_D1[..., 1],
            w_sameedge_D1[..., 2],
        ),
        dim=-1,
    )
    jacobian_sameedge_D1 = xsi * xsi * xsi * eta1 * eta1

    w_sameedge_D2 = torch.cat(
        (
            xsi,
            -xsi * eta1 * eta2 * eta3,
            xsi * eta1 * eta2 * (1.0 - eta3),
            xsi * eta1,
        ),
        dim=-1,
    )
    rands_sameedge_D2 = torch.stack(
        (
            w_sameedge_D2[..., 0],
            w_sameedge_D2[..., 3],
            w_sameedge_D2[..., 0] + w_sameedge_D2[..., 1],
            w_sameedge_D2[..., 2],
        ),
        dim=-1,
    )
    jacobian_sameedge_D2 = xsi * xsi * xsi * eta1 * eta1 * eta2

    w_sameedge_D3 = torch.cat(
        (
            xsi * (1.0 - eta1 * eta2),
            xsi * eta1 * eta2,
            xsi * eta1 * eta2 * eta3,
            xsi * eta1 * (1.0 - eta2),
        ),
        dim=-1,
    )
    rands_sameedge_D3 = torch.stack(
        (
            w_sameedge_D3[..., 0],
            w_sameedge_D3[..., 3],
            w_sameedge_D3[..., 0] + w_sameedge_D3[..., 1],
            w_sameedge_D3[..., 2],
        ),
        dim=-1,
    )
    jacobian_sameedge_D3 = xsi * xsi * xsi * eta1 * eta1 * eta2

    w_sameedge_D4 = torch.cat(
        (
            xsi * (1.0 - eta1 * eta2 * eta3),
            xsi * eta1 * eta2 * eta3,
            xsi * eta1,
            xsi * eta1 * eta2 * (1.0 - eta3),
        ),
        dim=-1,
    )
    rands_sameedge_D4 = torch.stack(
        (
            w_sameedge_D4[..., 0],
            w_sameedge_D4[..., 3],
            w_sameedge_D4[..., 0] + w_sameedge_D4[..., 1],
            w_sameedge_D4[..., 2],
        ),
        dim=-1,
    )
    jacobian_sameedge_D4 = xsi * xsi * xsi * eta1 * eta1 * eta2

    w_sameedge_D5 = torch.cat(
        (
            xsi * (1.0 - eta1 * eta2 * eta3),
            xsi * eta1 * eta2 * eta3,
            xsi * eta1 * eta2,
            xsi * eta1 * (1.0 - eta2 * eta3),
        ),
        dim=-1,
    )
    rands_sameedge_D5 = torch.stack(
        (
            w_sameedge_D5[..., 0],
            w_sameedge_D5[..., 3],
            w_sameedge_D5[..., 0] + w_sameedge_D5[..., 1],
            w_sameedge_D5[..., 2],
        ),
        dim=-1,
    )
    jacobian_sameedge_D5 = xsi * xsi * xsi * eta1 * eta1 * eta2

    rands_sameedge = torch.stack(
        (
            rands_sameedge_D1,
            rands_sameedge_D2,
            rands_sameedge_D3,
            rands_sameedge_D4,
            rands_sameedge_D5,
        ),
        dim=-2,
    )  # [GaussQR2, 5, 4]
    jacobian_sameedge = torch.cat(
        (
            jacobian_sameedge_D1,
            jacobian_sameedge_D2,
            jacobian_sameedge_D3,
            jacobian_sameedge_D4,
            jacobian_sameedge_D5,
        ),
        dim=-1,
    )  # [GaussQR2, 5]
    r1_x[int(PanelRelationType.SAME_EDGE)] = rands_sameedge[..., 0]
    r2_x[int(PanelRelationType.SAME_EDGE)] = rands_sameedge[..., 1]
    r1_y[int(PanelRelationType.SAME_EDGE)] = rands_sameedge[..., 2]
    r2_y[int(PanelRelationType.SAME_EDGE)] = rands_sameedge[..., 3]
    jacobian[int(PanelRelationType.SAME_EDGE)] = jacobian_sameedge

    # 3. Sometime, they are just exactly the same panel.
    mats_coincide = (
        torch.Tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, -1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, -1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0, -1.0],
                ],
            ],
        )
        .to(xsi.device)
        .to(xsi.dtype)
    )  # 3 x [4, 4]
    mats_coincide = mats_coincide.reshape(3, 1, 4, 4)
    w_sameface = torch.stack(
        (xsi, xsi * eta1, xsi * eta1 * eta2, xsi * eta1 * eta2 * eta3),
        dim=-2,
    )  # [GaussQR2, 4, 1]
    xz = (mats_coincide @ w_sameface).squeeze(-1)  # [3, GaussQR2, 4]
    xz = xz.permute((1, 0, 2))  # [GaussQR2, 3, 4]
    rands_sameface = torch.stack(
        (xz[..., 0], xz[..., 1], xz[..., 0] - xz[..., 2], xz[..., 1] - xz[..., 3]),
        dim=-1,
    )  # [GaussQR2, 3, 4]
    jacobian_sameface = xsi * xsi * xsi * eta1 * eta1 * eta2  # [GaussQR2, 1]

    r1_x[int(PanelRelationType.SAME_FACE)] = rands_sameface[..., 0]
    r2_x[int(PanelRelationType.SAME_FACE)] = rands_sameface[..., 1]
    r1_y[int(PanelRelationType.SAME_FACE)] = rands_sameface[..., 2]
    r2_y[int(PanelRelationType.SAME_FACE)] = rands_sameface[..., 3]
    jacobian[int(PanelRelationType.SAME_FACE)] = jacobian_sameface

    return r1_x, r2_x, r1_y, r2_y, jacobian


def double_integration_on_panels_3d(
    BEM_type: int,
    order_type: int,
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_areas: torch.Tensor,
    face_normals: torch.Tensor,
    gauss_weightsx: torch.Tensor,
    gauss_weightsy: torch.Tensor,
    rands_x: torch.Tensor,
    rands_y: torch.Tensor,
    panel_relations: torch.Tensor,
    wavenumber: float = 1.0,
    layer_type: int = int(LayerType.SINGLE_LAYER),
) -> torch.Tensor:
    """Get Double Integration on two triangles, where
    int_{Tau_x} int_{Tau_y} (func) dydx
        = (2 * area_x) (2 * area_y) * int_{unit_panel} int_{unit_panel} (func) (Jacobian) dr1 dr2

    Get Integration points and weights for a general triangle (Use Duffy Transform)
    ### Perspective from Duffy Transform
     y
     ^
    1|__________
     |          |
     |  *    *  |
     |          |     <- This refers GaussQR = 2, you will get GaussQR*GaussQR=4 points and 4 weights
     |  *    *  |
    0-----------|1--->x
     0

     ============= TO =============

     y
     ^
    1|
     |   /|
     |  / |
     | /  |           <- Points: (x, y) := (x, x * y)
     |/   |           <- Weights:   w  To  w
    0|----|1--->x     <- Jaobian:   1  To  x
     0

     ============= TO =============
      r2
     ^
    1|                                      x2
     |   /|                                 /|
     |  / |                                / |
     | /  |                      ->       /  |
     |/   |                              /   |
    0|----|1--->r1                    x3/____|x1
     0

    - How to project a unit triangle (x1, x2, x3) to a general one?
    - If we choose (0, 0)->x1, (1, 0)->x2, (1, 1)->x3
    - x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3

    ==============================
    To sum up:
        Sampled_point   = (1 - r1) * x1  +  (r1 - r2) * x2  +  r2 * x3
        Corres_weight   = w * 2 * Area
        Jacobian        = r1
        phase functions = 1 - r1, r1 - r2, r2

    !!!!!!!!!!!!!!
    However, if these two panels has overlaps, such like common vertices, even the same panel.
    For instance, the same panel:
    Reference: S. A. Sauter and C. Schwab, Boundary Element Methods, Springer Ser. Comput. Math. 39, Springer-Verlag, Berlin, 2011.
    https://link.springer.com/content/pdf/10.1007/978-3-540-68093-2.pdf
    See this algorithm from chapter 5.2.1

    Similarly, besides the coincide case we have mentioned above,
    other approaches such like common vertices/edges can be refered by chapter 5.2.2 and 5.2.3

    Args:
        face_areas (torch.Tensor): [N_faces, 1]
        face_normals (torch.Tensor): [N_faces, dim]
        gauss_weightsx (torch.Tensor): [GaussQR2, 1]
        gauss_weightsy (torch.Tensor): [GaussQR2, 1]
        rands_x (torch.Tensor): [GaussQR2, 2] consists of r1 and r2
        rands_y (torch.Tensor): [GaussQR2, 2] consists of r1 and r2
        panel_relations (torch.Tensor): [N_faces, N_faces]

    Returns:
        torch.Tensor: [N_faces, dim, gaussQR2, 1]
    """
    N_faces, dim = faces.shape
    N_verts, dim = verts.shape

    face_verts = get_vertices_from_index(verts, index=faces.flatten())
    face_verts = face_verts.reshape(N_faces, dim, dim)

    face_face_y = face_verts[:, None, ...].repeat(1, N_faces, 1, 1)
    face_face_x = face_verts[None].repeat(N_faces, 1, 1, 1)
    panel_relations_expanded = panel_relations[:, :, None, None].repeat(1, 1, dim, dim)

    GaussQR2 = gauss_weightsx.numel()
    GaussQR2 = gauss_weightsy.numel()
    GaussQR4 = GaussQR2 * GaussQR2
    Ng = GaussQR4

    # Generate number(xsi, eta1, eta2, eta3)
    xsi = rands_x[..., 0:1].reshape(1, GaussQR2).repeat(GaussQR2, 1).reshape(Ng, 1)
    eta1 = rands_x[..., 1:2].reshape(1, GaussQR2).repeat(GaussQR2, 1).reshape(Ng, 1)

    eta2 = rands_y[..., 0:1].reshape(GaussQR2, 1).repeat(1, GaussQR2).reshape(Ng, 1)
    eta3 = rands_y[..., 1:2].reshape(GaussQR2, 1).repeat(1, GaussQR2).reshape(Ng, 1)

    # Scale your weight
    weights_x = gauss_weightsx.reshape(1, GaussQR2).repeat(GaussQR2, 1).reshape(Ng, 1)
    weights_y = gauss_weightsy.reshape(GaussQR2, 1).repeat(1, GaussQR2).reshape(Ng, 1)
    weights_yx = (weights_y * weights_x).reshape(Ng, 1)  # [Ng, 1]
    area_fix = (face_areas * face_areas[:, None]).reshape(N_faces, N_faces)
    area_fix = 4 * area_fix  # Only use for 3D triangles
    face_face_normaly = face_normals[:, None, :].repeat(1, N_faces, 1)

    # Get your r1, r2 for each triangles.
    # In order of (r1_x, r2_x, r1_y, r2_y)
    r1_x_list, r2_x_list, r1_y_list, r2_y_list, jacobian_list = (
        get_rands_and_jacobian_3d(xsi=xsi, eta1=eta1, eta2=eta2, eta3=eta3)
    )

    if order_type == int(OrderType.PLANAR):
        N_phiy, N_phix = 1, 1
        V_mat_mask = panel_relations[:, :, None, None]
    elif order_type == int(OrderType.LINEAR):
        N_phiy, N_phix = dim, dim
        V_mat_mask = panel_relations_expanded

    V_mat = torch.zeros(
        (N_faces, N_faces, N_phiy, N_phix), device=verts.device, dtype=verts.dtype
    )
    if BEMType.use_complex(BEM_type=BEM_type):
        V_mat = V_mat + 1j * V_mat

    panel_relation_cases = [
        int(PanelRelationType.SEPARATE),
        int(PanelRelationType.SAME_VERTEX),
        int(PanelRelationType.SAME_EDGE),
        int(PanelRelationType.SAME_FACE),
    ]
    for panel_relation in panel_relation_cases:
        r1_x = r1_x_list[panel_relation]  # [Ng, ND]
        r2_x = r2_x_list[panel_relation]  # [Ng, ND]
        r1_y = r1_y_list[panel_relation]  # [Ng, ND]
        r2_y = r2_y_list[panel_relation]  # [Ng, ND]
        jacobian = jacobian_list[panel_relation]  # [Ng, ND]

        phix = shape_function(dim=dim, order_type=order_type, r1=r1_x, r2=r2_x)
        phix = torch.stack(phix, dim=-3)  # [N_phix, Ng, ND]

        phiy = shape_function(dim=dim, order_type=order_type, r1=r1_y, r2=r2_y)
        phiy = torch.stack(phiy, dim=-3)  # [N_phiy, Ng, ND]
        phiy = phiy[..., None, :, :]  # [N_phiy, 1, Ng, ND]

        phiyx = phiy * phix  # [N_phiy, N_phix, Ng, ND]

        area_fix_i = area_fix[panel_relations == panel_relation]
        N_cells = area_fix_i.numel()
        tmp_normals = face_face_normaly[
            panel_relations[..., None].repeat(1, 1, dim) == panel_relation
        ]
        tmp_normals = tmp_normals.reshape(N_cells, 1, 1, dim)
        triangle_x = face_face_x[panel_relations_expanded == panel_relation]
        triangle_x = triangle_x.reshape(N_cells, dim, 1, 1, dim)
        x0 = triangle_x[:, 0]  # [N_cells, 1, 1, dim]
        x1 = triangle_x[:, 1]  # [N_cells, 1, 1, dim]
        x2 = triangle_x[:, 2]  # [N_cells, 1, 1, dim]
        triangle_y = face_face_y[panel_relations_expanded == panel_relation]
        triangle_y = triangle_y.reshape(N_cells, dim, 1, 1, dim)
        y0 = triangle_y[:, 0]  # [N_cells, 1, 1, dim]
        y1 = triangle_y[:, 1]  # [N_cells, 1, 1, dim]
        y2 = triangle_y[:, 2]  # [N_cells, 1, 1, dim]

        x = linear_interplate_from_unit_panel_to_general(
            r1=r1_x.unsqueeze(-1), r2=r2_x.unsqueeze(-1), x1=x0, x2=x1, x3=x2
        )  # [N_cells, Ng, ND, dim]
        y = linear_interplate_from_unit_panel_to_general(
            r1=r1_y.unsqueeze(-1), r2=r2_y.unsqueeze(-1), x1=y0, x2=y1, x3=y2
        )  # [N_cells, Ng, ND, dim]

        if layer_type == int(LayerType.SINGLE_LAYER):
            fxy = G_3d(x=x, y=y, k=wavenumber, BEM_type=BEM_type, keepdim=False)
        elif layer_type == int(LayerType.DOUBLE_LAYER):
            fxy = (
                grad_G_y_3d(x=x, y=y, k=wavenumber, BEM_type=BEM_type) * tmp_normals
            ).sum(dim=-1, keepdim=False)
        elif layer_type == int(LayerType.HYPERSINGULAR_LAYER):
            fxy = G_3d(x=x, y=y, k=wavenumber, BEM_type=BEM_type, keepdim=False)
            raise NotImplementedError("Not implemented for hypersingular")

        area_fix_i = area_fix_i.reshape(N_cells, 1, 1, 1, 1)
        # [N_cells, 1, 1, Ng, ND]
        integrand = fxy[:, None, None, ...] * area_fix_i
        # [N_cells, N_phiy, N_phix, Ng, ND]
        integrand = integrand * (phiyx * weights_yx * jacobian)
        integrand = integrand.sum(dim=-2).sum(dim=-1)  # [N_cells, N_phiy, N_phix]
        V_mat[V_mat_mask == panel_relation] = integrand.flatten()

    return V_mat
