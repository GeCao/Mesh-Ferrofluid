#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define PANEL_TYPE_SAME 3
#define PANEL_TYPE_COMMOM_EDGE 2
#define PANEL_TYPE_COMMON_VERTEX 1
#define PANEL_TYPE_SEPARATE 0

template <typename scalar_t>
__device__ void cross(const scalar_t *t_vec, const scalar_t *b_vec, scalar_t *n_vec)
{
    int dim = 3;
    for (int axis = 0; axis < dim; ++axis)
    {
        int axis_next = (axis + 1) % dim;
        int axis_next_next = (axis + 2) % dim;
        n_vec[axis] = (t_vec[axis_next] * b_vec[axis_next_next] - t_vec[axis_next_next] * b_vec[axis_next]);
    }
}

template <typename scalar_t>
__device__ void calculate_Einc(const scalar_t wavenumber, const scalar_t *pos, scalar_t *Einc_real, scalar_t *Einc_imag)
{
    scalar_t azimuth = 30 / 180.0 * M_PI;
    scalar_t k[3] = {wavenumber * cos(azimuth), wavenumber * sin(azimuth), 0.0};
    scalar_t kx = k[0] * pos[0] + k[1] * pos[1] + k[2] * pos[2];

    Einc_real[0] = 0;
    Einc_real[1] = 0;
    Einc_real[2] = cos(-kx);

    Einc_imag[0] = 0;
    Einc_imag[1] = 0;
    Einc_imag[2] = sin(-kx);
}

template <typename scalar_t>
__device__ void calculate_Hinc(const scalar_t wavenumber, const scalar_t *pos, scalar_t *Hinc_real, scalar_t *Hinc_imag)
{
    /*
    """A self-defined function for calculate incident magnetic field H^{inc}

    Args:
        wavenumber (scalar_t): wave number
        pos (torch.Tensor): [x, y, z]

    Returns:
        torch.Tensor: [x, y, z]
    """
    */
    scalar_t azimuth = 30 / 180.0 * M_PI;
    scalar_t k[3] = {wavenumber * cos(azimuth), wavenumber * sin(azimuth), 0.0};
    scalar_t kx = k[0] * pos[0] + k[1] * pos[1] + k[2] * pos[2];

    scalar_t H_direc = azimuth - 0.5 * M_PI;
    scalar_t eps_r = 1.0, mu_r = 1.0;
    scalar_t eta0 = 1.0; // 376.73;
    scalar_t eta = sqrt(mu_r / eps_r) * eta0;

    Hinc_real[0] = (1 / eta) * cos(-kx) * cos(H_direc);
    Hinc_real[1] = (1 / eta) * cos(-kx) * sin(H_direc);
    Hinc_real[2] = 0;

    Hinc_imag[0] = (1 / eta) * sin(-kx) * cos(H_direc);
    Hinc_imag[1] = (1 / eta) * sin(-kx) * sin(H_direc);
    Hinc_imag[2] = 0;
}

template <typename scalar_t>
__device__ void G(const scalar_t wavenumber, const scalar_t *p1, const scalar_t *p2, scalar_t &G_real, scalar_t &G_imag)
{
    /*3D Green function for Helmholtz function

    G = exp(-1j * k * r) / (4 * PI * r)

    Args:
        wavenumber (float): wave number
        p1 (torch.Tensor): [x, y, z], the position of r
        p2 (torch.Tensor): [x, y, z], the position of r_prime

    Returns:
        torch.Tensor: G(p1, p2)
    */
    scalar_t r_vec[3] = {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
    scalar_t dist = sqrt(r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2]);
    scalar_t inv_dist = 1 / dist;
    scalar_t kx = wavenumber * dist;

    G_real = cos(-kx) / (4 * M_PI) * inv_dist;
    G_imag = sin(-kx) / (4 * M_PI) * inv_dist;
}

template <typename scalar_t>
__device__ void gradG_y(
    const scalar_t wavenumber,
    const scalar_t *p1,
    const scalar_t *p2,
    scalar_t *grad_G_real,
    scalar_t *grad_G_imag)
{
    /*\partial_G \partial_p2 for Helmholtz function

    Args:
        wavenumber (float): wave number
        p1 (torch.Tensor): [x, y, z], the position of r
        p2 (torch.Tensor): [x, y, z], the position of r_prime

    Returns:
        torch.Tensor: grad^{prime} G(r, r_prime)
    */
    scalar_t r_vec[3] = {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
    scalar_t dist = sqrt(r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2]);
    scalar_t inv_dist = 1 / dist;
    scalar_t kx = wavenumber * dist;

    for (int axis = 0; axis < 3; ++axis)
    {
        grad_G_real[axis] = (cos(-kx) - kx * sin(-kx)) / (4 * M_PI) * inv_dist * inv_dist * inv_dist * r_vec[axis];
        grad_G_imag[axis] = (sin(-kx) + kx * cos(-kx)) / (4 * M_PI) * inv_dist * inv_dist * inv_dist * r_vec[axis];
    }
}

template <typename scalar_t>
__device__ void gradG_x(
    const scalar_t wavenumber,
    const scalar_t *p1,
    const scalar_t *p2,
    scalar_t *grad_G_real,
    scalar_t *grad_G_imag)
{
    /*\partial_G \partial_p1 for Helmholtz function

    Args:
        wavenumber (float): wave number
        p1 (torch.Tensor): [x, y, z], the position of r
        p2 (torch.Tensor): [x, y, z], the position of r_prime

    Returns:
        torch.Tensor: grad G(r, r_prime)
    */
    gradG_y(wavenumber, p1, p2, grad_G_real, grad_G_imag);
    for (int axis = 0; axis < 3; ++axis)
    {
        grad_G_real[axis] = -grad_G_real[axis];
        grad_G_imag[axis] = -grad_G_imag[axis];
    }
}

__device__ bool is_Tn_plus(const int32_t v1_idx, const int32_t v2_idx)
{
    /* To define this face is Tn+ or Tn- for RWG function.
    While an edge (v1 -> v2) is typically attached to 2 triangles.
    For example:
        face 1: v0 -> v1 -> v2, (anti-clockwise)
        face 2: v3 -> v2 -> v1, (anti-clockwise)
    There always exist one face is v1->v2, and another one is v2->v1
    In this function, if the former v_idx is smaller, I treat it as Tn+
    */
    return v1_idx < v2_idx ? true : false;
}

template <typename scalar_t>
__device__ void RWG(
    const scalar_t *x,
    const scalar_t *verts,
    const scalar_t area,
    const int32_t v0_idx,
    const int32_t v1_idx,
    const int32_t v2_idx,
    scalar_t *RWG_result)
{
    /*RWG Function (Note v0, v1, v2 always are anti-clockwise on the triangle face)

    Args:
        x (torch.Tensor): [x, y, z], a sampled point on this triangle face
        verts (torch.Tensor): [N_verts, dim=3]
        area (scalar_t): the area of this triangle face
        v0_idx (int32_t): The vert_idx of stand-alone vertex
        v1_idx (int32_t): The 1st vert_idx of edge
        v2_idx (int32_t): The 2nd vert_idx of edge

    Returns:
        torch.Tensor: RWG(x)
    */
    int dim = 3;
    bool Tn_plus = is_Tn_plus(v1_idx, v2_idx);

    scalar_t v0[3] = {verts[v0_idx * dim + 0], verts[v0_idx * dim + 1], verts[v0_idx * dim + 2]};
    scalar_t v1[3] = {verts[v1_idx * dim + 0], verts[v1_idx * dim + 1], verts[v1_idx * dim + 2]};
    scalar_t v2[3] = {verts[v2_idx * dim + 0], verts[v2_idx * dim + 1], verts[v2_idx * dim + 2]};

    scalar_t edge[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
    scalar_t edge_len = sqrt(edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]);

    scalar_t RWG_scalar = edge_len / (2 * area);
    for (int axis = 0; axis < dim; ++axis)
    {
        RWG_result[axis] = Tn_plus ? x[axis] - v0[axis] : v0[axis] - x[axis];
        RWG_result[axis] = RWG_result[axis] * RWG_scalar;
    }
}

template <typename scalar_t>
__device__ scalar_t div_RWG(
    const scalar_t *verts,
    const scalar_t area,
    const int32_t v0_idx,
    const int32_t v1_idx,
    const int32_t v2_idx)
{
    /*div(RWG(x)) Function (Note v0, v1, v2 always are anti-clockwise on the triangle face)

    Args:
        verts (torch.Tensor): [N_verts, dim=3]
        area (scalar_t): the area of this triangle face
        v0_idx (int32_t): The vert_idx of stand-alone vertex
        v1_idx (int32_t): The 1st vert_idx of edge
        v2_idx (int32_t): The 2nd vert_idx of edge

    Returns:
        scalar_t: div(RWG(x))
    */
    int dim = 3;
    bool Tn_plus = is_Tn_plus(v1_idx, v2_idx);

    // scalar_t v0[3] = {verts[v0_idx * dim + 0], verts[v0_idx * dim + 1], verts[v0_idx * dim + 2]};
    scalar_t v1[3] = {verts[v1_idx * dim + 0], verts[v1_idx * dim + 1], verts[v1_idx * dim + 2]};
    scalar_t v2[3] = {verts[v2_idx * dim + 0], verts[v2_idx * dim + 1], verts[v2_idx * dim + 2]};

    scalar_t edge[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
    scalar_t edge_len = sqrt(edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]);

    scalar_t RWG_scalar = edge_len / (2 * area);
    scalar_t div_RWG_result = Tn_plus ? RWG_scalar * 2 : -RWG_scalar * 2;

    return div_RWG_result;
}

template <typename scalar_t>
__global__ void kernel_create_MOM_rhs_3d_forward(
    const scalar_t *verts,
    const int32_t *faces,
    const scalar_t *face_areas,
    const scalar_t *face_normals,
    const scalar_t *gaussian_points_1d,
    const scalar_t *gaussian_weights_1d,
    scalar_t *face_edge_rhs_real,
    scalar_t *face_edge_rhs_imag,
    scalar_t wavenumber,
    int N_verts,
    int N_faces,
    int batch_size,
    int N_GaussQR)
{
    /*Create MOM-based 3D RHS
    Args:
        verts: torch.Tensor = [N_verts, dim=3]
        faces: torch.Tensor = [N_faces, dim=3]
        face_areas: torch.Tensor = [N_faces, 1]
        face_normals: torch.Tensor = [N_faces, dim]
        gaussian_points_1d: torch.Tensor = [N_GaussQR]
        gaussian_weights_1d: torch.Tensor = [N_GaussQR]
    Return:
        face_edge_rhs: torch.Tensor = [B, N_faces, dim, 1]
    */
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim = 3;

    if (idx >= batch_size * N_faces * dim)
    {
        return;
    }

    const int batch_idx = idx / (N_faces * dim);
    const int face_idx = idx % (N_faces * dim) / dim;
    const int edge_idx = idx % dim;

    scalar_t area = face_areas[face_idx];
    scalar_t normal[3] = {face_normals[face_idx * dim + 0], face_normals[face_idx * dim + 1], face_normals[face_idx * dim + 2]};
    int32_t v0_idx = faces[face_idx * dim + edge_idx];
    int32_t v1_idx = faces[face_idx * dim + (edge_idx + 1) % dim];
    int32_t v2_idx = faces[face_idx * dim + (edge_idx + 2) % dim];
    scalar_t v0[3] = {verts[v0_idx * dim + 0], verts[v0_idx * dim + 1], verts[v0_idx * dim + 2]};
    scalar_t v1[3] = {verts[v1_idx * dim + 0], verts[v1_idx * dim + 1], verts[v1_idx * dim + 2]};
    scalar_t v2[3] = {verts[v2_idx * dim + 0], verts[v2_idx * dim + 1], verts[v2_idx * dim + 2]};
    scalar_t edge[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
    scalar_t edge_len = sqrt(edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]);

    scalar_t integration_real = 0.0, integration_imag = 0.0;
    int N_GaussQR2 = N_GaussQR * N_GaussQR;
    for (int i = 0; i < N_GaussQR2; ++i)
    {
        scalar_t r1 = gaussian_points_1d[i / N_GaussQR];
        scalar_t r2 = gaussian_points_1d[i % N_GaussQR] * r1;
        scalar_t weight = gaussian_weights_1d[i / N_GaussQR] * gaussian_weights_1d[i % N_GaussQR] * (2 * area);
        scalar_t jacobian = r1;

        scalar_t x[3] = {0.0, 0.0, 0.0};
        for (int axis = 0; axis < dim; ++axis)
        {
            x[axis] = (1 - r1) * v0[axis] + (r1 - r2) * v1[axis] + r2 * v2[axis];
        }

        scalar_t RWG1[3] = {0.0, 0.0, 0.0};
        RWG(x, verts, area, v0_idx, v1_idx, v2_idx, RWG1);

        scalar_t Einc_real[3], Einc_imag[3], Hinc_real[3], Hinc_imag[3];
        calculate_Einc(wavenumber, x, Einc_real, Einc_imag);
        calculate_Hinc(wavenumber, x, Hinc_real, Hinc_imag);

        for (int axis = 0; axis < dim; ++axis)
        {
            int axis_next = (axis + 1) % dim;
            int axis_next_next = (axis + 2) % dim;
            // EFIE
            integration_real += weight * jacobian * RWG1[axis] * (Einc_real[axis]);
            integration_imag += weight * jacobian * RWG1[axis] * (Einc_imag[axis]);
            // MFIE
            integration_real += weight * jacobian * RWG1[axis] * (normal[axis_next] * Hinc_real[axis_next_next] - Hinc_real[axis_next] * normal[axis_next_next]);
            integration_imag += weight * jacobian * RWG1[axis] * (normal[axis_next] * Hinc_imag[axis_next_next] - Hinc_imag[axis_next] * normal[axis_next_next]);
        }
    }
    face_edge_rhs_real[idx] = integration_real;
    face_edge_rhs_imag[idx] = integration_imag;
}

template <typename scalar_t>
__global__ void kernel_create_MOM_Zmat_3d_forward(
    const scalar_t *verts,
    const int32_t *faces,
    const scalar_t *face_areas,
    const scalar_t *face_normals,
    const scalar_t *gaussian_points_1d_x,
    const scalar_t *gaussian_weights_1d_x,
    const scalar_t *gaussian_points_1d_y,
    const scalar_t *gaussian_weights_1d_y,
    scalar_t *face_edge_Zmat_real,
    scalar_t *face_edge_Zmat_imag,
    scalar_t wavenumber,
    int N_verts,
    int N_faces,
    int batch_size,
    int Nx_GaussQR,
    int Ny_GaussQR)
{
    /*Create MOM-based 3D Zmat
    Args:
        verts: torch.Tensor = [N_verts, dim=3]
        faces: torch.Tensor = [N_faces, dim=3]
        face_areas: torch.Tensor = [N_faces, 1]
        face_normals: torch.Tensor = [N_faces, dim]
        gaussian_points_1d_x: torch.Tensor = [Nx_GaussQR]
        gaussian_weights_1d_x: torch.Tensor = [Nx_GaussQR]
        gaussian_points_1d_y: torch.Tensor = [Ny_GaussQR]
        gaussian_weights_1d_y: torch.Tensor = [Ny_GaussQR]
    Return:
        face_edge_Zmat: torch.Tensor = [B, N_faces * dim, N_faces * dim, 1]
    */
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim = 3;

    if (idx >= batch_size * N_faces * N_faces * dim * dim)
    {
        return;
    }

    const int batch_idx = idx / (N_faces * N_faces * dim * dim);
    const int face_edge1_idx = idx % (N_faces * N_faces * dim * dim) / (N_faces * dim);
    const int face_edge2_idx = idx % (N_faces * dim);

    const int f1 = face_edge1_idx / dim;
    const int f2 = face_edge2_idx / dim;

    const int edge1_idx = face_edge1_idx % dim;
    const int edge2_idx = face_edge2_idx % dim;

    face_edge_Zmat_real[idx] = 0.0;
    face_edge_Zmat_imag[idx] = 0.0;

    int panel_type = (f1 == f2) ? PANEL_TYPE_SAME : PANEL_TYPE_SEPARATE;

    scalar_t area1 = face_areas[f1], area2 = face_areas[f2];
    scalar_t normal1[3] = {face_normals[f1 * dim + 0], face_normals[f1 * dim + 1], face_normals[f1 * dim + 2]};
    scalar_t normal2[3] = {face_normals[f2 * dim + 0], face_normals[f2 * dim + 1], face_normals[f2 * dim + 2]};

    int32_t edge1_v0_idx = faces[f1 * dim + edge1_idx];
    int32_t edge1_v1_idx = faces[f1 * dim + (edge1_idx + 1) % dim];
    int32_t edge1_v2_idx = faces[f1 * dim + (edge1_idx + 2) % dim];
    int32_t edge2_v0_idx = faces[f2 * dim + edge2_idx];
    int32_t edge2_v1_idx = faces[f2 * dim + (edge2_idx + 1) % dim];
    int32_t edge2_v2_idx = faces[f2 * dim + (edge2_idx + 2) % dim];

    scalar_t f1_v0[3] = {verts[edge1_v0_idx * dim + 0], verts[edge1_v0_idx * dim + 1], verts[edge1_v0_idx * dim + 2]};
    scalar_t f1_v1[3] = {verts[edge1_v1_idx * dim + 0], verts[edge1_v1_idx * dim + 1], verts[edge1_v1_idx * dim + 2]};
    scalar_t f1_v2[3] = {verts[edge1_v2_idx * dim + 0], verts[edge1_v2_idx * dim + 1], verts[edge1_v2_idx * dim + 2]};
    scalar_t f2_v0[3] = {verts[edge2_v0_idx * dim + 0], verts[edge2_v0_idx * dim + 1], verts[edge2_v0_idx * dim + 2]};
    scalar_t f2_v1[3] = {verts[edge2_v1_idx * dim + 0], verts[edge2_v1_idx * dim + 1], verts[edge2_v1_idx * dim + 2]};
    scalar_t f2_v2[3] = {verts[edge2_v2_idx * dim + 0], verts[edge2_v2_idx * dim + 1], verts[edge2_v2_idx * dim + 2]};

    scalar_t div_RWG1 = div_RWG(verts, area1, edge1_v0_idx, edge1_v1_idx, edge1_v2_idx);
    scalar_t div_RWG2 = div_RWG(verts, area2, edge2_v0_idx, edge2_v1_idx, edge2_v2_idx);

    scalar_t integration_real = 0.0, integration_imag = 0.0;
    int Nx_GaussQR2 = Nx_GaussQR * Nx_GaussQR, Ny_GaussQR2 = Ny_GaussQR * Ny_GaussQR;
    for (int i = 0; i < Nx_GaussQR2; ++i)
    {
        scalar_t r1 = gaussian_points_1d_x[i / Nx_GaussQR];
        scalar_t r2 = gaussian_points_1d_x[i % Nx_GaussQR] * r1;
        scalar_t weight1 = gaussian_weights_1d_x[i / Nx_GaussQR] * gaussian_weights_1d_x[i % Nx_GaussQR] * (2 * area1);
        scalar_t jacobian1 = r1;

        scalar_t x[3] = {0.0, 0.0, 0.0};
        for (int axis = 0; axis < dim; ++axis)
        {
            x[axis] = (1 - r1) * f1_v0[axis] + (r1 - r2) * f1_v1[axis] + r2 * f1_v2[axis];
        }

        scalar_t RWG1[3] = {0.0, 0.0, 0.0};
        RWG(x, verts, area1, edge1_v0_idx, edge1_v1_idx, edge1_v2_idx, RWG1);

        // Part 1: RWG_m * RWG_n / 2, only valid when you are on the same panel.
        if (panel_type == PANEL_TYPE_SAME)
        {
            scalar_t RWG2[3] = {0.0, 0.0, 0.0};
            RWG(x, verts, area2, edge2_v0_idx, edge2_v1_idx, edge2_v2_idx, RWG2);

            for (int axis = 0; axis < dim; ++axis)
            {
                integration_real += weight1 * jacobian1 * 0.5 * RWG1[axis] * RWG2[axis]; // MFIE
            }
        }
        // Part 2 & 3: n x K(RWG), L(RWG).
        scalar_t K_RWG_real[3] = {0.0, 0.0, 0.0};
        scalar_t K_RWG_imag[3] = {0.0, 0.0, 0.0};
        scalar_t L1_RWG_real[3] = {0.0, 0.0, 0.0};
        scalar_t L1_RWG_imag[3] = {0.0, 0.0, 0.0};
        scalar_t L2_RWG_real = 0.0;
        scalar_t L2_RWG_imag = 0.0;
        for (int j = 0; j < Ny_GaussQR2; ++j)
        {
            scalar_t r3 = gaussian_points_1d_y[j / Ny_GaussQR];
            scalar_t r4 = gaussian_points_1d_y[j % Ny_GaussQR] * r3;
            scalar_t weight2 = gaussian_weights_1d_y[j / Ny_GaussQR] * gaussian_weights_1d_y[j % Ny_GaussQR] * (2 * area2);
            scalar_t jacobian2 = r3;

            scalar_t y[3] = {0.0, 0.0, 0.0};
            for (int axis = 0; axis < dim; ++axis)
            {
                y[axis] = (1 - r3) * f2_v0[axis] + (r3 - r4) * f2_v1[axis] + r4 * f2_v2[axis];
            }

            scalar_t RWG2[3] = {0.0, 0.0, 0.0};
            RWG(y, verts, area2, edge2_v0_idx, edge2_v1_idx, edge2_v2_idx, RWG2);

            scalar_t G_real = 0.0, G_imag = 0.0;
            G(wavenumber, x, y, G_real, G_imag);

            scalar_t gradG_x_real[3] = {0.0, 0.0, 0.0};
            scalar_t gradG_x_imag[3] = {0.0, 0.0, 0.0};
            gradG_x(wavenumber, x, y, gradG_x_real, gradG_x_imag);

            for (int axis = 0; axis < dim; ++axis)
            {
                int axis_next = (axis + 1) % dim;
                int axis_next_next = (axis + 2) % dim;
                // MFIE
                K_RWG_real[axis] += weight2 * jacobian2 * (RWG2[axis_next] * gradG_x_real[axis_next_next] - RWG2[axis_next_next] * gradG_x_real[axis_next]);
                K_RWG_imag[axis] += weight2 * jacobian2 * (RWG2[axis_next] * gradG_x_imag[axis_next_next] - RWG2[axis_next_next] * gradG_x_imag[axis_next]);
                // EFIE
                L1_RWG_imag[axis] += wavenumber * weight2 * jacobian2 * (RWG2[axis] * G_real);
                L1_RWG_real[axis] += -wavenumber * weight2 * jacobian2 * (RWG2[axis] * G_imag);
            }
            // EFIE
            L2_RWG_imag += -1.0 / wavenumber * weight2 * jacobian2 * (div_RWG2 * G_real);
            L2_RWG_real += 1.0 / wavenumber * weight2 * jacobian2 * (div_RWG2 * G_imag);
        }
        scalar_t n_cross_K_RWG_real[3] = {0.0, 0.0, 0.0};
        scalar_t n_cross_K_RWG_imag[3] = {0.0, 0.0, 0.0};
        cross(normal1, K_RWG_real, n_cross_K_RWG_real);
        cross(normal1, K_RWG_imag, n_cross_K_RWG_imag);
        for (int axis = 0; axis < dim; ++axis)
        {
            integration_real += weight1 * jacobian1 * RWG1[axis] * (n_cross_K_RWG_real[axis] + L1_RWG_real[axis]);
            integration_imag += weight1 * jacobian1 * RWG1[axis] * (n_cross_K_RWG_imag[axis] + L1_RWG_imag[axis]);
        }
        integration_real += weight1 * jacobian1 * div_RWG1 * L2_RWG_real;
        integration_imag += weight1 * jacobian1 * div_RWG1 * L2_RWG_imag;
    }
    face_edge_Zmat_real[idx] = integration_real;
    face_edge_Zmat_imag[idx] = integration_imag;
}

template <typename scalar_t>
__global__ void kernel_interpolate_Icoeff_to_Js_3d_forward(
    const scalar_t *verts,
    const int32_t *faces,
    const int32_t *face_edge_indices,
    const scalar_t *face_areas,
    const scalar_t *edge_I_coeff_real,
    const scalar_t *edge_I_coeff_imag,
    scalar_t *face_vert_Js_real,
    scalar_t *face_vert_Js_imag,
    int N_verts,
    int N_faces,
    int N_edges,
    int batch_size)
{
    /*Interpolate MOM-based I_coeff to Js
    Args:
        verts: torch.Tensor = [N_verts, dim=3]
        faces: torch.Tensor = [N_faces, dim=3]
        face_edge_indices: torch.Tensor = [N_faces * dim], the edge indices
        face_areas: torch.Tensor = [N_faces, 1]
        edge_I_coeff_real: torch.Tensor = [B, N_edges, 1]
        edge_I_coeff_imag: torch.Tensor = [B, N_edges, 1]
    Return:
        face_vert_Js: torch.Tensor = [B, N_faces, dim, dim]
    */
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim = 3;

    if (idx >= batch_size * N_faces)
    {
        return;
    }

    const int batch_idx = idx / N_faces;
    const int face_idx = idx % N_faces;

    for (int axis_idx = 0; axis_idx < dim; ++axis_idx)
    {
        const int edge_idx = face_edge_indices[face_idx * dim + axis_idx];

        const int v0_idx = faces[face_idx * dim + axis_idx]; // This is the stand-alone vertex.
        const int v1_idx = faces[face_idx * dim + (axis_idx + 1) % dim];
        const int v2_idx = faces[face_idx * dim + (axis_idx + 2) % dim];

        // scalar_t v0[3] = {verts[v0_idx * dim + 0], verts[v0_idx * dim + 1], verts[v0_idx * dim + 2]};
        scalar_t v1[3] = {verts[v1_idx * dim + 0], verts[v1_idx * dim + 1], verts[v1_idx * dim + 2]};
        scalar_t v2[3] = {verts[v2_idx * dim + 0], verts[v2_idx * dim + 1], verts[v2_idx * dim + 2]};

        scalar_t area = face_areas[face_idx];

        scalar_t RWG1[3] = {0.0, 0.0, 0.0};
        RWG(v1, verts, area, v0_idx, v1_idx, v2_idx, RWG1);
        scalar_t RWG2[3] = {0.0, 0.0, 0.0};
        RWG(v2, verts, area, v0_idx, v1_idx, v2_idx, RWG2);

        for (int axis = 0; axis < dim; ++axis)
        {
            // v1
            face_vert_Js_real[batch_idx * (N_faces * dim * dim) + face_idx * (dim * dim) + ((axis_idx + 1) % dim) * dim + axis] += RWG1[axis] * edge_I_coeff_real[batch_idx * N_edges + edge_idx];
            face_vert_Js_imag[batch_idx * (N_faces * dim * dim) + face_idx * (dim * dim) + ((axis_idx + 1) % dim) * dim + axis] += RWG1[axis] * edge_I_coeff_imag[batch_idx * N_edges + edge_idx];

            // v2
            face_vert_Js_real[batch_idx * (N_faces * dim * dim) + face_idx * (dim * dim) + ((axis_idx + 2) % dim) * dim + axis] += RWG2[axis] * edge_I_coeff_real[batch_idx * N_edges + edge_idx];
            face_vert_Js_imag[batch_idx * (N_faces * dim * dim) + face_idx * (dim * dim) + ((axis_idx + 2) % dim) * dim + axis] += RWG2[axis] * edge_I_coeff_imag[batch_idx * N_edges + edge_idx];
        }
    }
}

void cu_create_MOM_rhs_3d_forward(
    const torch::Tensor &verts,
    const torch::Tensor &faces,
    const torch::Tensor &face_areas,
    const torch::Tensor &face_normals,
    const torch::Tensor &gaussian_points_1d,
    const torch::Tensor &gaussian_weights_1d,
    torch::Tensor &face_edge_rhs_real,
    torch::Tensor &face_edge_rhs_imag,
    double wavenumber)
{
    /*Create MOM-based 3D RHS
    Args:
        verts: torch.Tensor = [N_verts, dim=3]
        faces: torch.Tensor = [N_faces, dim=3]
        face_areas: torch.Tensor = [N_faces, 1]
        face_normals: torch.Tensor = [N_faces, 1]
        gaussian_points_1d: torch.Tensor = [N_GaussQR]
        gaussian_weights_1d: torch.Tensor = [N_GaussQR]
    Return:
        face_edge_rhs: torch.Tensor = [B, N_faces, dim, 1]
    */
    int batch_size = face_edge_rhs_real.size(0);
    int N_verts = verts.size(0);
    int N_faces = faces.size(0);
    int dim = verts.size(1);
    int N_GaussQR = gaussian_weights_1d.numel();

    const int threads = 512;
    const dim3 blocks((batch_size * N_faces * dim - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(verts.scalar_type(), "create_MOM_rhs_3d_forward", ([&]
                                                                                  { kernel_create_MOM_rhs_3d_forward<scalar_t><<<blocks, threads>>>(
                                                                                        (const scalar_t *)verts.data_ptr(),
                                                                                        (const int32_t *)faces.data_ptr(),
                                                                                        (const scalar_t *)face_areas.data_ptr(),
                                                                                        (const scalar_t *)face_normals.data_ptr(),
                                                                                        (const scalar_t *)gaussian_points_1d.data_ptr(),
                                                                                        (const scalar_t *)gaussian_weights_1d.data_ptr(),
                                                                                        (scalar_t *)face_edge_rhs_real.data_ptr(),
                                                                                        (scalar_t *)face_edge_rhs_imag.data_ptr(),
                                                                                        wavenumber,
                                                                                        N_verts,
                                                                                        N_faces,
                                                                                        batch_size,
                                                                                        N_GaussQR); }));
}

void cu_create_MOM_Zmat_3d_forward(
    const torch::Tensor &verts,
    const torch::Tensor &faces,
    const torch::Tensor &face_areas,
    const torch::Tensor &face_normals,
    const torch::Tensor &gaussian_points_1d_x,
    const torch::Tensor &gaussian_weights_1d_x,
    const torch::Tensor &gaussian_points_1d_y,
    const torch::Tensor &gaussian_weights_1d_y,
    torch::Tensor &face_edge_Zmat_real,
    torch::Tensor &face_edge_Zmat_imag,
    double wavenumber)
{
    /*Create MOM-based 3D Zmat
    Args:
        verts: torch.Tensor = [N_verts, dim=3]
        faces: torch.Tensor = [N_faces, dim=3]
        face_areas: torch.Tensor = [N_faces, 1]
        face_normals: torch.Tensor = [N_faces, 1]
        gaussian_points_1d_x: torch.Tensor = [Nx_GaussQR]
        gaussian_weights_1d_x: torch.Tensor = [Nx_GaussQR]
        gaussian_points_1d_y: torch.Tensor = [Ny_GaussQR]
        gaussian_weights_1d_y: torch.Tensor = [Ny_GaussQR]
    Return:
        Zmat: torch.Tensor = [B, N_edges, N_edges, 1]
    */
    int batch_size = face_edge_Zmat_imag.size(0);
    int N_verts = verts.size(0);
    int N_faces = faces.size(0);
    int dim = verts.size(1);
    int Nx_GaussQR = gaussian_weights_1d_x.numel();
    int Ny_GaussQR = gaussian_weights_1d_y.numel();

    const int threads = 512;
    const dim3 blocks((batch_size * N_faces * dim * N_faces * dim - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(verts.scalar_type(), "create_MOM_Zmat_3d_forward", ([&]
                                                                                   { kernel_create_MOM_Zmat_3d_forward<scalar_t><<<blocks, threads>>>(
                                                                                         (const scalar_t *)verts.data_ptr(),
                                                                                         (const int32_t *)faces.data_ptr(),
                                                                                         (const scalar_t *)face_areas.data_ptr(),
                                                                                         (const scalar_t *)face_normals.data_ptr(),
                                                                                         (const scalar_t *)gaussian_points_1d_x.data_ptr(),
                                                                                         (const scalar_t *)gaussian_weights_1d_x.data_ptr(),
                                                                                         (const scalar_t *)gaussian_points_1d_y.data_ptr(),
                                                                                         (const scalar_t *)gaussian_weights_1d_y.data_ptr(),
                                                                                         (scalar_t *)face_edge_Zmat_real.data_ptr(),
                                                                                         (scalar_t *)face_edge_Zmat_imag.data_ptr(),
                                                                                         wavenumber,
                                                                                         N_verts,
                                                                                         N_faces,
                                                                                         batch_size,
                                                                                         Nx_GaussQR,
                                                                                         Ny_GaussQR); }));
}

void cu_interpolate_Icoeff_to_Js_3d_forward(
    const torch::Tensor &verts,
    const torch::Tensor &faces,
    const torch::Tensor &face_edge_indices,
    const torch::Tensor &face_areas,
    const torch::Tensor &edge_I_coeff_real,
    const torch::Tensor &edge_I_coeff_imag,
    torch::Tensor &face_vert_Js_real,
    torch::Tensor &face_vert_Js_imag)
{
    /*Create MOM-based 3D Zmat
    Args:
        verts: torch.Tensor = [N_verts, dim=3]
        faces: torch.Tensor = [N_faces, dim=3]
        face_edge_indices: torch.Tensor = [N_faces * dim], the edge indices
        face_areas: torch.Tensor = [N_faces, 1]
        face_normals: torch.Tensor = [N_faces, 1]
        edge_I_coeff_real: torch.Tensor = [B, N_edges, 1]
        edge_I_coeff_image: torch.Tensor = [B, N_edges, 1]
    Return:
        face_vert_Js: torch.Tensor = [B, N_faces, dim, dim]
    */
    int batch_size = face_vert_Js_real.size(0);
    int N_verts = verts.size(0);
    int N_faces = faces.size(0);
    int N_edges = edge_I_coeff_imag.size(1);
    int dim = verts.size(1);

    const int threads = 512;
    const dim3 blocks((batch_size * N_faces - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(verts.scalar_type(), "interpolate_Icoeff_to_Js_3d_forward", ([&]
                                                                                            { kernel_interpolate_Icoeff_to_Js_3d_forward<scalar_t><<<blocks, threads>>>(
                                                                                                  (const scalar_t *)verts.data_ptr(),
                                                                                                  (const int32_t *)faces.data_ptr(),
                                                                                                  (const int32_t *)face_edge_indices.data_ptr(),
                                                                                                  (const scalar_t *)face_areas.data_ptr(),
                                                                                                  (const scalar_t *)edge_I_coeff_real.data_ptr(),
                                                                                                  (const scalar_t *)edge_I_coeff_imag.data_ptr(),
                                                                                                  (scalar_t *)face_vert_Js_real.data_ptr(),
                                                                                                  (scalar_t *)face_vert_Js_imag.data_ptr(),
                                                                                                  N_verts,
                                                                                                  N_faces,
                                                                                                  N_edges,
                                                                                                  batch_size); }));
}