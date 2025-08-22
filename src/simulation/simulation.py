import os
import math
import torch
from typing import List, Dict

from src.mesh_ferrofluid import MeshAdvection, MeshHelmholtzDecomposition, MeshPressure
from src.mesh_BEM import MeshBEMSolver2d, MeshBEMSolver3d
from src.mesh_WOB import MeshWOBSolver2d, MeshWOBSolver3d
from src.utils import Logger, c0


class EMSimulationParameters(object):
    def __init__(
        self,
        dim: int,
        dt: float,
        dx: float,
        simulation_size: List[int],
        freq: float,
        save_path: str,
        condMax: float = None,
        Z0: float = 1.0,
        BEM_params: Dict[str, int] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.dim = dim
        self.dt = dt
        self.dx = dx
        self.simulation_size = simulation_size
        self.freq = freq
        self.condMax = condMax
        self.Z0 = Z0
        self.save_path = save_path
        self.dtype = dtype
        self.device = device

        if BEM_params is not None:
            self.gaussQR = BEM_params["gaussQR"]
            self.BEM_type = BEM_params["BEM_type"]
            self.order_type = BEM_params["order_type"]


class FluidSimulationParameters(object):
    def __init__(
        self,
        dim: int = 2,
        dt: float = 1.0,
        simulation_size: List[int] = [1, 1, 256, 256],
        physics_type: int = 0,
        axisymmetric_type: int = 0,
        density_gas: float = 0.0,
        density_fluid: float = 1.0,
        gravity_strength: float = 0.0,
        LBM_parmas: Dict[str, float] = None,
        multiphase_params: Dict[str, float] = None,
        surface_only_params: Dict[str, float] = None,
        k: float = 1.0,
        dtype=torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.dim = dim
        self.dtype = dtype
        self.dt = dt
        self.device = device

        self.frame = 0
        self.time_per_frame = 0
        self.frame_length = 1.0
        self.time_total = 0

        if dim == 2 and len(simulation_size) != 4:
            raise ValueError(
                "For 2d simulation simulation size should have 4 parameters B x C x H x W"
            )

        self.simulation_size = simulation_size
        self.physics_type = physics_type
        self.axisymmetric_type = axisymmetric_type

        self.density_gas = density_gas
        self.density_fluid = density_fluid
        self.gravity_strength = gravity_strength

        # For multi-phase fluid simulation only
        if multiphase_params is not None:
            self.density_gas = multiphase_params["density_gas"]
            self.surface_tension = multiphase_params["surface_tension"]
            self.contact_angle = multiphase_params["contact_angle"]

            # Sometimes you actually want to optimize the CA or surface_tension:
            self.surface_tension = (
                torch.Tensor([self.surface_tension]).to(device).to(dtype)
            )
            self.contact_angle = torch.Tensor([self.contact_angle]).to(device).to(dtype)

        if LBM_parmas is not None:
            self.Q = LBM_parmas["Q"]
            self.tau = LBM_parmas["tau"]
            self.rho_gas = LBM_parmas["rho_gas"]
            self.rho_fluid = LBM_parmas["rho_fluid"]
            self.kappa = LBM_parmas["kappa"]
            self.tau_g = LBM_parmas["tau_g"]
            self.tau_f = LBM_parmas["tau_f"]

        if surface_only_params is not None:
            self.gaussQR = surface_only_params["gaussQR"]
            self.order_type = surface_only_params["order_type"]

        self.k = k  # susceptibilty

    def step(self):
        """Advances the simulation one time step"""
        self.time_per_frame += self.dt
        self.time_total += self.dt

        if self.time_per_frame >= self.frame_length:
            self.frame += 1

            # re-calculate total time to prevent drift
            self.time_total = self.frame * self.frame_length
            self.time_per_frame = 0

    def get_dx(self):
        return 1.0 / max(self.simulation_size)

    def is_2d(self):
        return self.dim == 2

    def is_3d(self):
        return self.dim == 3

    def set_device(self, device: str = "cuda"):
        if device not in ["cuda", "cpu"]:
            raise ValueError(
                "Set_device: device {} must be either cuda or cpu.".format(device)
            )
        self.device = torch.device(device)


class SimulationRunner(object):
    def __init__(
        self,
        EM_parameters: EMSimulationParameters = None,
        fluid_parameters: FluidSimulationParameters = None,
        logger: Logger = None,
    ) -> None:
        self.EM_parameters = EM_parameters
        self.fluid_parameters = fluid_parameters
        self.logger = logger

        self.clock = 0
        self.curr_time = 0.0

    def step(self, dt: float):
        self.clock = self.clock + 1
        self.curr_time += dt

    def create_mesh_advection(self):
        return MeshAdvection()

    def create_mesh_Helmholtz_decomposition(self):
        return MeshHelmholtzDecomposition(
            gaussQR=self.fluid_parameters.gaussQR,
            device=self.fluid_parameters.device,
            dtype=self.fluid_parameters.dtype,
        )

    def create_mesh_pressure_solver(self):
        return MeshPressure(
            density_fluid=self.fluid_parameters.density_fluid,
            gravity_strength=self.fluid_parameters.gravity_strength,
            surface_tension=self.fluid_parameters.surface_tension,
            contact_angle=self.fluid_parameters.contact_angle,
            gaussQR=self.fluid_parameters.gaussQR,
            order_type=self.fluid_parameters.order_type,
            logger=self.logger,
            device=self.fluid_parameters.device,
            dtype=self.fluid_parameters.dtype,
        )

    def create_mesh_BEM_solver(self):
        if self.EM_parameters.dim == 2:
            return MeshBEMSolver2d(
                gaussQR=self.EM_parameters.gaussQR,
                order_type=self.EM_parameters.order_type,
                BEM_type=self.EM_parameters.BEM_type,
                wavenumber=2 * math.pi * self.EM_parameters.freq / c0,
                logger=self.logger,
                dtype=self.EM_parameters.dtype,
                device=self.EM_parameters.device,
            )
        elif self.EM_parameters.dim == 3:
            return MeshBEMSolver3d(
                gaussQR=self.EM_parameters.gaussQR,
                order_type=self.EM_parameters.order_type,
                BEM_type=self.EM_parameters.BEM_type,
                wavenumber=2 * math.pi * self.EM_parameters.freq / c0,
                logger=self.logger,
                dtype=self.EM_parameters.dtype,
                device=self.EM_parameters.device,
            )

    def create_mesh_WOB_solver(self):
        if self.EM_parameters.dim == 2:
            return MeshWOBSolver2d(
                wavenumber=2 * math.pi * self.EM_parameters.freq / c0,
                logger=self.logger,
                dtype=self.EM_parameters.dtype,
                device=self.EM_parameters.device,
            )
        elif self.EM_parameters.dim == 3:
            return MeshWOBSolver3d(
                wavenumber=2 * math.pi * self.EM_parameters.freq / c0,
                logger=self.logger,
                dtype=self.EM_parameters.dtype,
                device=self.EM_parameters.device,
            )
