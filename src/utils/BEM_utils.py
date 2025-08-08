import math
import torch
import torch.nn.functional as F
from typing import List

from src.utils import BEMType, OrderType


def Hankel(order: int, kind: int, z: torch.Tensor) -> torch.Tensor:
    """Hankel function
    Args:
        order: (int)
        kind (int)
        z (torch.Tensor)
    Returns:
        torch.Tensor: Hankel(z)
    """
    theta = math.pi / 4.0 + order * (math.pi / 2.0)

    if kind == 1:
        hankel_z = torch.cos(z - theta) + 1j * torch.sin(z - theta)
        hankel_z = hankel_z * torch.sqrt(2.0 / math.pi / z)
    if kind == 2:
        hankel_z = torch.cos(-(z - theta)) + 1j * torch.sin(-(z - theta))
        hankel_z = hankel_z * torch.sqrt(2.0 / math.pi / z)
    else:
        raise NotImplementedError(
            f"Not implemented for Hankel function: kind = {kind}, order = 0"
        )

    return hankel_z


def G_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    k: float = 1,
    BEM_type: int = int(BEMType.LAPLACE),
    keepdim: bool = True,
) -> torch.Tensor:
    """G(x, y) = 1 / (4 * PI * |x - y|)

    Args:
        x (torch.Tensor): [..., dim=3]
        y (torch.Tensor): [..., dim=3]
        k (float, optional): The wavenumber. Defaults to 1.
        BEM_type (int, optional): Solving Laplace equation or Helmholtz?. Defaults to int(BEMType.LAPLACE).
        keepdim (bool, optional): Defaults to True.

    Returns:
        torch.Tensor: G(x, y) -> [..., (1)]
    """
    dim = x.shape[-1]
    assert dim == 2

    distance = (x - y).norm(dim=-1, keepdim=keepdim)
    if BEM_type == int(BEMType.LAPLACE):
        Gxy = (1.0 / 2.0 / math.pi) * torch.log(1.0 / distance)
    elif BEMType.belong_Helmholtz(BEM_type=BEM_type):
        # In usual BEM:  (1j / 4) * H_{0}^{1}(kd)  ---> exp(-1j wt)
        # In EM Physics: (-1j / 4) * H_{0}^{2}(kd) ---> exp(1j wt)
        kind_type = 2
        hankel_z = Hankel(order=0, kind=kind_type, z=k * distance)
        if kind_type == 1:
            Gxy = 0.25 * (-hankel_z.imag + 1j * hankel_z.real)
        elif kind_type == 2:
            Gxy = 0.25 * (hankel_z.imag - 1j * hankel_z.real)

    return Gxy


def G_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    k: float = 1,
    BEM_type: int = int(BEMType.LAPLACE),
    keepdim: bool = True,
) -> torch.Tensor:
    """G(x, y) = 1 / (4 * PI * |x - y|)

    Args:
        x (torch.Tensor): [..., dim=3]
        y (torch.Tensor): [..., dim=3]
        k (float, optional): The wavenumber. Defaults to 1.
        BEM_type (int, optional): Solving Laplace equation or Helmholtz?. Defaults to int(BEMType.LAPLACE).
        keepdim (bool, optional): Defaults to True.

    Returns:
        torch.Tensor: G(x, y) -> [..., (1)]
    """
    dim = x.shape[-1]
    assert dim == 3

    distance = (x - y).norm(dim=-1, keepdim=keepdim)
    if BEM_type == int(BEMType.LAPLACE):
        Gxy = (1.0 / 4.0 / math.pi) / distance
    elif BEMType.belong_Helmholtz(BEM_type=BEM_type):
        Gxy = (
            (1.0 / 4.0 / math.pi)
            / distance
            * (torch.cos(k * distance) + 1j * torch.sin(k * distance))
        )

    return Gxy


def grad_G_y_2d(
    x: torch.Tensor, y: torch.Tensor, k: float = 1, BEM_type: int = int(BEMType.LAPLACE)
) -> torch.Tensor:
    """\partial_G(x, y)/\partial_y

    Args:
        x (torch.Tensor): [..., dim=3]
        y (torch.Tensor): [..., dim=3]
        k (float, optional): The wavenumber. Defaults to 1.
        BEM_type (int, optional): Solving Laplace equation or Helmholtz?. Defaults to int(BEMType.LAPLACE).

    Returns:
        torch.Tensor: grad_G(x, y)_at_y -> [..., dim=3]
    """
    dim = x.shape[-1]
    assert dim == 2

    distance = (x - y).norm(dim=-1, keepdim=True)
    direc = F.normalize(x - y, dim=-1)
    if BEM_type == int(BEMType.LAPLACE):
        grad_Gy = (1.0 / 2.0 / math.pi) * direc / distance
    elif BEMType.belong_Helmholtz(BEM_type=BEM_type):
        # In usual BEM:  (1j / 4) * H_{1}^{1}(kd)  ---> exp(-1j wt)
        # In EM Physics: (-1j / 4) * H_{1}^{2}(kd) ---> exp(1j wt)
        kind_type = 2
        hankel_z = Hankel(order=1, kind=kind_type, z=k * distance)
        if kind_type == 1:
            grad_Gy = 0.25 * (-hankel_z.imag + 1j * hankel_z.real) * (k * direc)
        elif kind_type == 2:
            grad_Gy = 0.25 * (hankel_z.imag - 1j * hankel_z.real) * (k * direc)

    return grad_Gy


def grad_G_y_3d(
    x: torch.Tensor, y: torch.Tensor, k: float = 1, BEM_type: int = int(BEMType.LAPLACE)
) -> torch.Tensor:
    """\partial_G(x, y)/\partial_y

    Args:
        x (torch.Tensor): [..., dim=3]
        y (torch.Tensor): [..., dim=3]
        k (float, optional): The wavenumber. Defaults to 1.
        BEM_type (int, optional): Solving Laplace equation or Helmholtz?. Defaults to int(BEMType.LAPLACE).

    Returns:
        torch.Tensor: grad_G(x, y)_at_y -> [..., dim=3]
    """
    dim = x.shape[-1]
    assert dim == 3

    distance = (x - y).norm(dim=-1, keepdim=True)
    distance_sqr = distance * distance
    direc = F.normalize(x - y, dim=-1)
    if BEM_type == int(BEMType.LAPLACE):
        grad_Gy = (1.0 / 4.0 / math.pi) * direc / distance_sqr
    elif BEMType.belong_Helmholtz(BEM_type=BEM_type):
        grad_Gy = (
            (1.0 / 4.0 / math.pi)
            * direc
            / distance_sqr
            * (1 - 1j * k * distance)
            * (torch.cos(k * distance) + 1j * torch.sin(k * distance))
        )

    return grad_Gy


def shape_function(
    dim: int, order_type: int, r1: torch.Tensor, r2: torch.Tensor = None
) -> List[torch.Tensor]:
    """Calculate shape function
    Providing 2D: x(r1)     = (1 - r1) * x1 + r1 * x2
    Providing 3D: x(r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3

    Args:
        dim (int): 2 or 3
        r1 (torch.Tensor): [..., 1]
        r2 (torch.Tensor): [..., 1]

    Raises:
        RuntimeError: We only accept 2D/3D for calculating shape function

    Returns:
        List[torch.Tensor]: dim x [torch.Tensor]
    """
    # Default as 1 to cancel shape function term
    result = 0.0 * r1 + 1.0  # Avoid type check, this equals as: result = 1.0

    if dim == 2:
        if order_type == int(OrderType.PLANAR):
            return (torch.ones_like(r1),)
        elif order_type == int(OrderType.LINEAR):
            return 1 - r1, r1
    elif dim == 3:
        if order_type == int(OrderType.PLANAR):
            return (torch.ones_like(r1),)
        elif order_type == int(OrderType.LINEAR):
            return 1 - r1, r1 - r2, r2
    else:
        raise RuntimeError("We only accept 2D/3D for calculating shape function")


def linear_interplate_from_unit_panel_to_general(
    r1: torch.Tensor,
    r2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
) -> torch.Tensor:
    """Given randomly sampled (r1, r2) pair, get the real vertice position from triangle
    r2
    ^
    |1                                     x2
    |   /|                                 /|
    |  / |                                / |
    | /  |                      ->       /  |
    |/   |                              /   |
    |----|1--->r1                    x3/____|x1
    0

    - How to project a unit triangle (x1, x2, x3) to a general one?
    - If we choose (0, 0)->x1, (1, 0)->x2, (1, 1)->x3
    - x(r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3

    Args:
        r1 (torch.Tensor): [..., 1]
        r2 (torch.Tensor): [..., 1]
        x1 (torch.Tensor): [N_faces, dim]
        x2 (torch.Tensor): [N_faces, dim]
        x3 (torch.Tensor): [N_faces, dim]

    Returns:
        torch.Tensor: Interpolated x(r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3
    """
    dim = x1.shape[-1]
    psi_lists = shape_function(dim=dim, order_type=int(OrderType.LINEAR), r1=r1, r2=r2)

    result = x1
    if dim == 2:
        result = psi_lists[0] * x1 + psi_lists[1] * x2
    elif dim == 3:
        result = psi_lists[0] * x1 + psi_lists[1] * x2 + psi_lists[2] * x3

    return result


def single_integration_on_panels_3d(
    dim: int,
    order_type: int,
    face_areas: torch.Tensor,
    gauss_weights: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor = None,
) -> torch.Tensor:
    """Get Integration on a single triangle, where
    int_{Tau_x} (func) dx
        = (2 * area_x) * int_{unit_panel} (func) (Jacobian) dr1 dr2

    ============= TO =============
    r2
    ^
    |1                                     x2
    |   /|                                 /|
    |  / |                                / |
    | /  |                      ->       /  |
    |/   |                              /   |
    |----|1--->r1                    x3/____|x1
    0

    - How to project a unit triangle (x1, x2, x3) to a general one?
    - If we choose (0, 0)->x1, (1, 0)->x2, (1, 1)->x3
    - x (r1, r2) = (1 - r1) * x1 + (r1 - r2) * x2 + r2 * x3

    Args:
        dim (int): 2 or 3
        face_areas (torch.Tensor): [N_faces, dim]
        gauss_weights (torch.Tensor): [GaussQR2, 1]
        r1 (torch.Tensor): [GaussQR2, 1]
        r2 (torch.Tensor): [GaussQR2, 1]

    Returns:
        torch.Tensor: [N_faces, dim, gaussQR2, 1]
    """
    assert dim == 2 or dim == 3

    GaussQR2 = gauss_weights.numel()

    r1_y = r1.reshape(GaussQR2, 1)  # [gaussQR2, 1]
    r2_y = r2.reshape(GaussQR2, 1)  # [gaussQR2, 1]

    weights_y = (
        2 * face_areas[..., None, None, :] * gauss_weights.reshape(GaussQR2, 1)
    )  # [N_faces, 1, gaussQR2, 1]

    jacobian = r1_y

    phiy = shape_function(
        dim=dim, order_type=order_type, r1=r1_y, r2=r2_y
    )  # Nphi x [gaussQR2, 1]
    phiy = torch.stack(phiy, dim=-3)  # [Nphi, gaussQR2, 1]

    integrand = phiy * weights_y * jacobian  # [N_faces, Nphi, gaussQR2, 1]

    return integrand


def pack_matrix_tight(
    order_type: int,
    unpacked_matrix: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Get The matrix in tight format
    Our matrix is saved with the format of: [N_faces, N_phiy, N_faces, N_phix]
    However, to use it in our BEM solver, it has to be in format of [N_verts, N_verts]

    Args:
        unpacked_matrix (torch.Tensor): [N_facesy, N_facesx, N_phiy, N_phix]
        verts (torch.Tensor): [N_verts, dim]
        faces (torch.Tensor): [N_faces, dim]

    Returns:
        torch.Tensor: [N_verts, N_verts]
    """
    N_verts, dim = verts.shape
    N_faces, dim = faces.shape
    device = verts.device
    dtype = verts.dtype
    mat_dtype = unpacked_matrix.dtype

    mat_output = torch.zeros((N_verts * N_verts), device=device, dtype=mat_dtype)
    if order_type == int(OrderType.PLANAR):
        # N_phiy, N_phix = 1, 1
        mat_src = unpacked_matrix.reshape(N_faces, N_faces, 1, 1)
        mat_src = mat_src.repeat(1, 1, dim, dim) / (dim * dim)
        # N_phiy, N_phix = dim, dim
    elif order_type == int(OrderType.LINEAR):
        mat_src = unpacked_matrix  # [N_facesy, N_facesx, dimy, dimx]

    mat_src = mat_src.permute((0, 2, 1, 3))  # [N_facesy, dimy, N_facesx, dimx]
    mat_src = mat_src.flatten()

    mat_index = faces.reshape(N_faces * dim, 1) * N_verts + faces.flatten()
    mat_index = mat_index.flatten().to(torch.int64)

    mat_output.scatter_add_(dim=-1, index=mat_index, src=mat_src)

    mat_output = mat_output.reshape(N_verts, N_verts)

    return mat_output


def get_gaussion_integration_points_and_weights(
    N: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """Calcuate Gaussian Points and weights for 1D integration

    Args:
        N (int): order
        device (torch.device, optional): Defaults to torch.device("cpu").
        dtype (torch.dtype, optional): Defaults to torch.float32.

    Raises:
        NotImplementedError: The order cannot be larger than 12

    Returns:
        List[torch.Tensor]:
            Points (torch.Tensor): [N,] from 0 to 1
            Weights (torch.Tensor): [N,] sum as 1
    """
    if N == 1:
        np_points = torch.Tensor([0.0]).to(dtype).to(device)
        np_weights = torch.Tensor([2.0]).to(dtype).to(device)
    elif N == 2:
        np_points = (
            torch.Tensor([-0.5773502691896257, 0.5773502691896257]).to(dtype).to(device)
        )
        np_weights = torch.Tensor([1.0, 1.0]).to(dtype).to(device)
    elif N == 3:
        np_points = (
            torch.Tensor([-0.7745966692414834, 0.0, 0.7745966692414834])
            .to(dtype)
            .to(device)
        )
        np_weights = (
            torch.Tensor([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])
            .to(dtype)
            .to(device)
        )
    elif N == 4:
        np_points = (
            torch.Tensor(
                [
                    -0.8611363115940526,
                    -0.3399810435848563,
                    0.3399810435848563,
                    0.8611363115940526,
                ]
            )
            .to(dtype)
            .to(device)
        )
        np_weights = (
            torch.Tensor(
                [
                    0.3478548451374538,
                    0.6521451548625461,
                    0.6521451548625461,
                    0.3478548451374538,
                ]
            )
            .to(dtype)
            .to(device)
        )
    elif N == 5:
        np_points = (
            torch.Tensor(
                [
                    -0.9061798459386640,
                    -0.5384693101056831,
                    0.0,
                    0.5384693101056831,
                    0.9061798459386640,
                ]
            )
            .to(dtype)
            .to(device)
        )
        np_weights = (
            torch.Tensor(
                [
                    0.2369268850561891,
                    0.4786286704993665,
                    0.5688888888888889,
                    0.4786286704993665,
                    0.2369268850561891,
                ]
            )
            .to(dtype)
            .to(device)
        )
    elif N == 6:
        np_points = (
            torch.Tensor(
                [
                    -0.9324695142031521,
                    -0.6612093864662645,
                    -0.2386191860831969,
                    0.2386191860831969,
                    0.6612093864662645,
                    0.9324695142031521,
                ]
            )
            .to(dtype)
            .to(device)
        )
        np_weights = (
            torch.Tensor(
                [
                    0.1713244923791704,
                    0.3607615730481386,
                    0.4679139345726910,
                    0.4679139345726910,
                    0.3607615730481386,
                    0.1713244923791704,
                ]
            )
            .to(dtype)
            .to(device)
        )
    elif N == 7:
        np_points = (
            torch.Tensor(
                [
                    -0.9491079123427585,
                    -0.7415311855993945,
                    -0.4058451513773972,
                    0.0,
                    0.4058451513773972,
                    0.7415311855993945,
                    0.9491079123427585,
                ]
            )
            .to(dtype)
            .to(device)
        )
        np_weights = (
            torch.Tensor(
                [
                    0.1294849661688697,
                    0.2797053914892766,
                    0.3818300505051189,
                    0.4179591836734694,
                    0.3818300505051189,
                    0.2797053914892766,
                    0.1294849661688697,
                ]
            )
            .to(dtype)
            .to(device)
        )
    elif N == 8:
        np_points = (
            torch.Tensor(
                [
                    -0.9602898564975363,
                    -0.7966664774136267,
                    -0.5255324099163290,
                    -0.1834346424956498,
                    0.1834346424956498,
                    0.5255324099163290,
                    0.7966664774136267,
                    0.9602898564975363,
                ]
            )
            .to(dtype)
            .to(device)
        )
        np_weights = (
            torch.Tensor(
                [
                    0.1012285362903763,
                    0.2223810344533745,
                    0.3137066458778873,
                    0.3626837833783620,
                    0.3626837833783620,
                    0.3137066458778873,
                    0.2223810344533745,
                    0.1012285362903763,
                ]
            )
            .to(dtype)
            .to(device)
        )
    elif N == 12:
        np_points = (
            torch.Tensor(
                [
                    -0.9815606342467192,
                    -0.9041172563704749,
                    -0.7699026741943047,
                    -0.5873179542866175,
                    -0.3678314989981802,
                    -0.1252334085114689,
                    0.1252334085114689,
                    0.3678314989981802,
                    0.5873179542866175,
                    0.7699026741943047,
                    0.9041172563704749,
                    0.9815606342467192,
                ]
            )
            .to(dtype)
            .to(device)
        )
        np_weights = (
            torch.Tensor(
                [
                    0.0471753363865118,
                    0.1069393259953184,
                    0.1600783285433462,
                    0.2031674267230659,
                    0.2334925365383548,
                    0.2491470458134028,
                    0.2491470458134028,
                    0.2334925365383548,
                    0.2031674267230659,
                    0.1600783285433462,
                    0.1069393259953184,
                    0.0471753363865118,
                ]
            )
            .to(dtype)
            .to(device)
        )
    else:
        raise NotImplementedError("Please refer N as 2~8 for gaussion interation")

    return [(np_points + 1.0) / 2.0, np_weights / 2.0]
