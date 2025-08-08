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
    G_2d,
    grad_G_y_2d,
    compute_mesh_face_curl,
)


def get_rands_and_jacobian_2d(
    xsi: torch.Tensor, eta: torch.Tensor
) -> Tuple[List[torch.Tensor]]:
    """Get r1 based on panel relations

    Args:
        xsi (torch.Tensor): [GaussQR, 1]
        eta (torch.Tensor): [GaussQR, 1]

    Returns:
        Tuple[List[torch.Tensor]]:
            r1_x: [r1_x_separate, r1_x_same_vertex, r1_x_same_edge, r1_x_same_face]
            r1_y: ...
            jacobian: ...
        Every single Tensor are in shape of [GaussQR, N_Ds]
    """
    xsi = xsi.reshape(-1, 1)
    eta = eta.reshape(-1, 1)

    r1_x = [None for i in range(4)]
    r1_y = [None for i in range(4)]
    jacobian = [None for i in range(4)]

    # 0. In default, the panels are all separated.
    r1_x[int(PanelRelationType.SEPARATE)] = xsi
    r1_y[int(PanelRelationType.SEPARATE)] = eta
    jacobian[int(PanelRelationType.SEPARATE)] = 1.0

    # 1. Sometimes, these two panels might share the same vertex
    r1_x[int(PanelRelationType.SAME_VERTEX)] = xsi
    r1_y[int(PanelRelationType.SAME_VERTEX)] = eta
    jacobian[int(PanelRelationType.SAME_VERTEX)] = 1.0

    # 2. Sometime, they are just exactly the same panel.
    r1_x[int(PanelRelationType.SAME_EDGE)] = xsi
    r1_y[int(PanelRelationType.SAME_EDGE)] = xsi * eta
    jacobian[int(PanelRelationType.SAME_EDGE)] = xsi

    return r1_x, r1_y, jacobian


def double_integration_on_panels_2d(
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
        = (area_x) (area_y) * int_{unit_panel} int_{unit_panel} (func) (Jacobian) dr1 dr2

    0---*----*--|1--->x     <- This refers GaussQR = 2, you will get GaussQR=2 points and 2 weights

    ============= TO =============

    0---*----*--|1--->x

                                            x2
                                            /
    0---*----*--|1--->x                    /
                                 ->       /
                                         /
                                      x1/


    - How to project a unit line to a general one?
    - If we choose (0,)->x1, (1)->x2,
    - x (r1) = (1 - r1) * x1 + r2 * x2

    ==============================
    To sum up:
        Sampled_point   = (1 - r1) * x1  +  r2 * x2
        Corres_weight   = w * Area
        Jacobian        = r1
        phase functions = 1 - r1, r2

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
        gauss_weightsx (torch.Tensor): [GaussQR, 1]
        gauss_weightsy (torch.Tensor): [GaussQR, 1]
        rands_x (torch.Tensor): [GaussQR, 1] consists of r1 only
        rands_y (torch.Tensor): [GaussQR, 1] consists of r1 only
        panel_relations (torch.Tensor): [N_faces, N_faces]

    Returns:
        torch.Tensor: [N_faces, dim, gaussQR, 1]
    """
    N_faces, dim = faces.shape
    N_verts, dim = verts.shape

    face_verts = get_vertices_from_index(verts, index=faces.flatten())
    face_verts = face_verts.reshape(N_faces, dim, dim)

    face_face_y = face_verts[:, None, ...].repeat(1, N_faces, 1, 1)
    face_face_x = face_verts[None].repeat(N_faces, 1, 1, 1)
    panel_relations_expanded = panel_relations[:, :, None, None].repeat(1, 1, dim, dim)

    GaussQR = gauss_weightsx.numel()
    GaussQR = gauss_weightsy.numel()
    GaussQR2 = GaussQR * GaussQR
    Ng = GaussQR2

    # Generate number(xsi, eta)
    xsi = rands_x[..., 0:1].reshape(1, GaussQR).repeat(GaussQR, 1).reshape(Ng, 1)

    eta = rands_y[..., 0:1].reshape(GaussQR, 1).repeat(1, GaussQR).reshape(Ng, 1)

    # Scale your weight
    weights_x = gauss_weightsx.reshape(1, GaussQR).repeat(GaussQR, 1).reshape(Ng, 1)
    weights_y = gauss_weightsy.reshape(GaussQR, 1).repeat(1, GaussQR).reshape(Ng, 1)
    weights_yx = (weights_y * weights_x).reshape(Ng, 1)  # [Ng, 1]
    area_fix = (face_areas * face_areas[:, None]).reshape(N_faces, N_faces)
    face_face_normaly = face_normals[:, None, :].repeat(1, N_faces, 1)

    # Get your r1 for each triangles.
    # In order of (r1_x, r1_y)
    r1_x_list, r1_y_list, jacobian_list = get_rands_and_jacobian_2d(xsi=xsi, eta=eta)

    # Get the curl if you need it:
    if layer_type == int(LayerType.HYPERSINGULAR_LAYER):
        pass

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
    ]
    for panel_relation in panel_relation_cases:
        r1_x = r1_x_list[panel_relation]  # [Ng, ND]
        r1_y = r1_y_list[panel_relation]  # [Ng, ND]
        jacobian = jacobian_list[panel_relation]  # [Ng, ND]

        phix = shape_function(dim=dim, order_type=order_type, r1=r1_x)
        phix = torch.stack(phix, dim=-3)  # [N_phix, Ng, ND]

        phiy = shape_function(dim=dim, order_type=order_type, r1=r1_y)
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
        triangle_y = face_face_y[panel_relations_expanded == panel_relation]
        triangle_y = triangle_y.reshape(N_cells, dim, 1, 1, dim)
        y0 = triangle_y[:, 0]  # [N_cells, 1, 1, dim]
        y1 = triangle_y[:, 1]  # [N_cells, 1, 1, dim]

        x = linear_interplate_from_unit_panel_to_general(
            r1=r1_x.unsqueeze(-1), r2=None, x1=x0, x2=x1, x3=None
        )  # [N_cells, Ng, ND, dim]
        y = linear_interplate_from_unit_panel_to_general(
            r1=r1_y.unsqueeze(-1), r2=None, x1=y0, x2=y1, x3=None
        )  # [N_cells, Ng, ND, dim]

        if layer_type == int(LayerType.SINGLE_LAYER):
            fxy = G_2d(x=x, y=y, k=wavenumber, BEM_type=BEM_type, keepdim=False)
        elif layer_type == int(LayerType.DOUBLE_LAYER):
            fxy = (
                grad_G_y_2d(x=x, y=y, k=wavenumber, BEM_type=BEM_type) * tmp_normals
            ).sum(dim=-1, keepdim=False)
        elif layer_type == int(LayerType.HYPERSINGULAR_LAYER):
            fxy = G_2d(x=x, y=y, k=wavenumber, BEM_type=BEM_type, keepdim=False)
            raise NotImplementedError("Not implemented for hypersingular")

        area_fix_i = area_fix_i.reshape(N_cells, 1, 1, 1, 1)
        # [N_cells, 1, 1, Ng, ND]
        integrand = fxy[:, None, None, ...] * area_fix_i
        # [N_cells, N_phiy, N_phix, Ng, ND]
        integrand = integrand * (phiyx * weights_yx * jacobian)
        integrand = integrand.sum(dim=-2).sum(dim=-1)  # [N_cells, N_phiy, N_phix]
        V_mat[V_mat_mask == panel_relation] = integrand.flatten()

    return V_mat
