import torch


class MeshAdvection(object):
    def __init__(self) -> None:
        pass

    def solve(self, verts: torch.Tensor, dt: float, vel: torch.Tensor) -> torch.Tensor:
        """Mesh-based advection

        Args:
            verts (torch.Tensor): [..., N_verts, dim=3]
            dt (float): time step
            vel (torch.Tensor): [..., N_verts, dim=3]

        Returns:
            torch.Tensor: [..., N_Verts, dim=3], The updated vertices
        """
        # Advection by update vertices directly
        verts = verts + vel * dt

        return verts
