# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Scattering from a sphere using a combined direct formulation
# ## Background
#
# In this tutorial, we will solve the problem of scattering from the unit sphere $\Omega$ using a combined integral formulation and an incident wave defined by
#
# $$
# u^{\text{inc}}(\mathbf x) = \mathrm{e}^{\mathrm{i} k x}.
# $$
#
# where $\mathbf x = (x, y, z)$.
#
# The PDE is given by the Helmholtz equation:
#
# $$
# \Delta u + k^2 u = 0, \quad \text{ in } \mathbb{R}^3 \backslash \Omega,
# $$
#
# where $u=u^\text{s}+u^\text{inc}$ is the total acoustic field and $u^\text{s}$ satisfies the Sommerfeld radiation condition
#
# $$
# \frac{\partial u^\text{s}}{\partial r}-\mathrm{i}ku^\text{s}=o(r^{-1})
# $$
#
# for $r:=|\mathbf{x}|\rightarrow\infty$.
#
# From Green's representation formula, one can derive that
#
# $$
# u(\mathbf x) = u^\text{inc}-\int_{\Gamma}g(\mathbf x,\mathbf y)\frac{\partial u}{\partial\nu}(\mathbf y)\mathrm{d}\mathbf{y}.
# $$
#
# Here, $g(\mathbf x, \mathbf y)$ is the acoustic Green's function given by
#
# $$
# g(\mathbf x, \mathbf y):=\frac{\mathrm{e}^{\mathrm{i} k |\mathbf{x}-\mathbf{y}|}}{4 \pi |\mathbf{x}-\mathbf{y}|}.
# $$
#
# The problem has therefore been reduced to computing the normal derivative $u_\nu:=\frac{\partial u}{\partial\nu}$ on the boundary $\Gamma$. This is achieved using the following boundary integral equation formulation.
#
# $$
# (\tfrac12\mathsf{Id} + \mathsf{K}' - \mathrm{i} \eta \mathsf{V}) u_\nu(\mathbf{x}) = \frac{\partial u^{\text{inc}}}{\partial \nu}(\mathbf{x}) - \mathrm{i} \eta u^{\text{inc}}(\mathbf{x}), \quad \mathbf{x} \in \Gamma.
# $$
#
# where $\mathsf{Id}$, $\mathsf{K}'$ and $\mathsf{V}$ are identity, adjoint double layer and single layer <a href='https://bempp_cl.com/2017/07/11/available_operators/'>boundary operators</a>. More details of the derivation of this formulation and its properties can be found in the article <a href='http://journals.cambridge.org/action/displayAbstract?fromPage=online&aid=8539370&fileId=S0962492912000037' target='new'>Chandler-Wilde <em>et al</em> (2012)</a>.
#
# ## Implementation
#
# First we import the Bempp module and NumPy.

import os
import sys
import math
import bempp_cl.api
import numpy as np
import torch
import pathlib
from typing import List

sys.path.append("../")
from src.utils import (
    GetRayQueryDepthMap,
    LoadSingleMesh,
    initialize_vulkan_ray_querier,
    mkdir,
    create_2d_meshgrid_tensor,
    c0,
)


def mesh_normalize(
    verts: torch.Tensor, scale: float = 1.0, normalize: bool = True
) -> torch.Tensor:
    if normalize:
        vert_min, _ = verts.min(dim=0)
        vert_max, _ = verts.max(dim=0)
        vert_center = 0.5 * (vert_min + vert_max)
        verts = (verts - vert_center) / (vert_max - vert_min).max() * 2
        # verts = verts * 2 - 1  # [-1, 1]

    verts = verts * scale
    verts = verts
    return verts


# We define the wavenumber

freq = 10e9
k = 2 * math.pi * freq / c0

# The following command creates a sphere mesh.

# grid = bempp_cl.api.shapes.regular_sphere(3)

dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
path = pathlib.Path(__file__).parent.absolute()
save_path = f"{path}/../save"
mkdir(save_path)
save_path = f"{save_path}/data_BEMpp_Helmholtz_{dim}d/"
mkdir(save_path)
asset_path = f"{path}/../assets/"
grid = bempp_cl.api.import_grid(f"{asset_path}/nasaAlmond_v2.ply")

# As basis functions, we use piecewise constant functions over the elements of the mesh. The corresponding space is initialised as follows.

piecewise_const_space = bempp_cl.api.function_space(grid, "DP", 0)

# We now initialise the <a href='https://bempp_cl.com/2017/07/11/operators/'>boundary operators</a>.
# A boundary operator always takes at least three space arguments: a domain space, a range space and the test space (dual to the range). In this example we only work on the space $\mathcal{L}^2(\Gamma)$ and we can choose all spaces to be identical.

identity = bempp_cl.api.operators.boundary.sparse.identity(
    piecewise_const_space, piecewise_const_space, piecewise_const_space
)
adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
    piecewise_const_space, piecewise_const_space, piecewise_const_space, k
)
slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
    piecewise_const_space, piecewise_const_space, piecewise_const_space, k
)

# Standard arithmetic operators can be used to create linear combinations of boundary operators.

lhs = 0.5 * identity + adlp - 1j * k * slp

# We now form the right-hand side by defining a <a href='https://bempp_cl.com/2017/07/11/grid-functions/'>GridFunction</a> using Python callable.


# +
# @bempp_cl.api.complex_callable
# def combined_data(x, n, domain_index, result):
#     result[0] = 1j * k * np.exp(1j * k * x[0]) * (n[0] - 1)


@bempp_cl.api.complex_callable
def combined_data(x, n, domain_index, result):
    azumith = 30 / 180.0 * np.pi
    # result[0] = (
    #     1j
    #     * k
    #     * np.exp(1j * (k * x[0] * np.cos(azumith) + k * x[1] * np.sin(azumith)))
    #     * (n[0] * np.cos(azumith) + n[1] * np.sin(azumith) - 1)
    # )
    result[0] = (
        1j
        * k
        * np.exp(1j * (k * x[0] * np.cos(azumith) + k * x[1] * np.sin(azumith)))
        * (n[2] - 1)
    )


grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, fun=combined_data)
# -

# We can now use GMRES to solve the problem.

from bempp_cl.api.linalg import gmres

neumann_fun, info = gmres(lhs, grid_fun, tol=1e-5)

# `gmres` returns a grid function `neumann_fun` and an integer `info`. When everything works fine info is equal to 0.
#
# At this stage, we have the surface solution of the integral equation. Now we will evaluate the solution in the domain of interest. We define the evaluation points as follows.

# +
res = [64, 64, 64]
meshgrid = create_2d_meshgrid_tensor([1, 1, *res[-2:]], device=device, dtype=dtype)
meshgrid = meshgrid.reshape(2, -1).permute((1, 0))
meshgrid = torch.cat((meshgrid, torch.zeros_like(meshgrid[..., 0:1])), dim=-1)
meshgrid = mesh_normalize(meshgrid, scale=0.3)  # [Nm, dim=3]
u_evaluated = np.zeros((meshgrid.shape[0]), dtype=np.complex128)
u_evaluated[:] = np.nan  # [Nm,]

# -

# This will generate a grid of points in the $x$-$y$ plane.
#
# Then we create a single layer potential operator and use it to evaluate the solution at the evaluation points. The variable ``idx`` allows to compute the solution only at points located outside the unit circle of the plane. We use a single layer potential operator to evaluate the solution at the observation points.

# find mask
verts, faces = LoadSingleMesh(
    obj_path=f"/home/moritz/homework/Mesh-Ferrofluid/assets/nasaAlmond_v2.ply",
    device=device,
    dtype=dtype,
)
N_verts, dim = verts.shape
N_faces, dim = faces.shape
ray_querier = initialize_vulkan_ray_querier(verts=verts, faces=faces)
Nm = meshgrid.shape[0]
dim = 3
pm_rayd = (
    torch.Tensor(
        [[[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]]
    )
    .to(dtype)
    .to(device)
    .repeat(Nm, 1, 1)
)
pm_depth, _ = GetRayQueryDepthMap(
    vulkan_ray_tracer=ray_querier, ray_o=meshgrid.reshape(Nm, dim), ray_d=pm_rayd
)
pm_depth = pm_depth.reshape(Nm, 6, 1)
mask = (pm_depth >= 0).sum(dim=-2) == pm_depth.shape[-2]  # [Nm, 1]
mask = mask | (((pm_depth < 0.01) & (pm_depth >= 0)).sum(dim=-2) > 0)
mask = mask.reshape(Nm)
extent_mask = mask.reshape(Nm, 1).repeat(1, dim)

from bempp_cl.api.operators.potential import helmholtz as helmholtz_potential

points = meshgrid[~extent_mask].reshape(-1, dim)

slp_pot = helmholtz_potential.single_layer(
    piecewise_const_space, points.permute((1, 0)).cpu().numpy(), k
)
# result = np.real(np.exp(1j * k * points[:, 0]) - slp_pot.evaluate(neumann_fun))
result = np.real(-slp_pot.evaluate(neumann_fun))
u_evaluated[~mask.cpu().numpy()] = result.flat
# -

# We now plot the slice of the domain solution.

# +
try:
    get_ipython().run_line_magic("matplotlib", "inline")
    ipython = True
except NameError:
    ipython = False

u_evaluated = u_evaluated.reshape((res[-2], res[-1]))

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 8))
plt.imshow(np.real(u_evaluated), extent=[-0.3, 0.3, -0.3, 0.3], cmap="viridis")
plt.xlabel("Y")
plt.ylabel("X")
plt.colorbar()
plt.title("Scattering from the NasaAlmond, solution in plane z=0")
if not ipython:
    plt.savefig(f"{save_path}/example-helmholtz_combined_exterior.png")
# -
