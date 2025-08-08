import torch

from src.utils import create_2d_meshgrid_tensor, create_3d_meshgrid_tensor


def create_droplet(
    rho: torch.Tensor, center: torch.Tensor, radius: float, rho_liquid: float
) -> torch.Tensor:
    """Create a droplet on this rho grid

    Args:
        rho (torch.Tensor): [B, 1, (D), H, W]
        center (torch.Tensor): Tensor([x, y, z])
        radius (float): The radius of droplet
        rho_liquid (float): The density of droplet

    Raises:
        RuntimeError: Density has to be in 2D/3D, ie. format of [B, 1, (D), H, W]

    Returns:
        torch.Tensor: rho grid with droplet
    """
    device = rho.device
    dtype = rho.dtype

    if rho.dim() == 4:
        dim = 2
        meshgrid = create_2d_meshgrid_tensor(rho.shape, device=device, dtype=dtype)
        dist = (meshgrid - center[..., None, None]).norm(dim=1, keepdim=True)
    elif rho.dim() == 5:
        dim = 3
        meshgrid = create_3d_meshgrid_tensor(rho.shape, device=device, dtype=dtype)
        dist = (meshgrid - center[..., None, None, None]).norm(dim=1, keepdim=True)
    else:
        raise RuntimeError(
            "Density has to be in 2D/3D, ie. format of [B, 1, (D), H, W]"
        )

    is_droplet = dist < radius
    rho[is_droplet] = rho_liquid

    return rho


def create_cube(
    rho: torch.Tensor, center: torch.Tensor, half_size: float, rho_liquid: float
) -> torch.Tensor:
    """Create a droplet on this rho grid

    Args:
        rho (torch.Tensor): [B, 1, (D), H, W]
        center (torch.Tensor): Tensor([x, y, z])
        half_size (float): The half size (radius) of box
        rho_liquid (float): The density of droplet

    Raises:
        RuntimeError: Density has to be in 2D/3D, ie. format of [B, 1, (D), H, W]

    Returns:
        torch.Tensor: rho grid with droplet
    """
    device = rho.device
    dtype = rho.dtype

    if rho.dim() == 4:
        dim = 2
        meshgrid = create_2d_meshgrid_tensor(rho.shape, device=device, dtype=dtype)
        dist, _ = (meshgrid - center[..., None, None]).max(dim=1, keepdim=True)
    elif rho.dim() == 5:
        dim = 3
        meshgrid = create_3d_meshgrid_tensor(rho.shape, device=device, dtype=dtype)
        dist, _ = (meshgrid - center[..., None, None, None]).max(dim=1, keepdim=True)
    else:
        raise RuntimeError(
            "Density has to be in 2D/3D, ie. format of [B, 1, (D), H, W]"
        )

    is_droplet = dist < half_size
    rho[is_droplet] = rho_liquid

    return rho
