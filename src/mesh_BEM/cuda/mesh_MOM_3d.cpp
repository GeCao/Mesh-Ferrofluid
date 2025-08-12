#include <torch/extension.h>

void cu_create_MOM_rhs_3d_forward(
    const torch::Tensor &verts,
    const torch::Tensor &faces,
    const torch::Tensor &face_areas,
    const torch::Tensor &face_normals,
    const torch::Tensor &gaussian_points_1d,
    const torch::Tensor &gaussian_weights_1d,
    torch::Tensor &face_edge_rhs_real,
    torch::Tensor &face_edge_rhs_imag,
    double wavenumber
);

void create_MOM_rhs_3d_forward(
    const torch::Tensor &verts,
    const torch::Tensor &faces,
    const torch::Tensor &face_areas,
    const torch::Tensor &face_normals,
    const torch::Tensor &gaussian_points_1d,
    const torch::Tensor &gaussian_weights_1d,
    torch::Tensor &face_edge_rhs_real,
    torch::Tensor &face_edge_rhs_imag,
    double wavenumber
) {
    cu_create_MOM_rhs_3d_forward(
        verts,
        faces,
        face_areas,
        face_normals,
        gaussian_points_1d,
        gaussian_weights_1d,
        face_edge_rhs_real,
        face_edge_rhs_imag,
        wavenumber);
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
    double wavenumber
);

void create_MOM_Zmat_3d_forward(
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
    double wavenumber
) {
    cu_create_MOM_Zmat_3d_forward(
        verts,
        faces,
        face_areas,
        face_normals,
        gaussian_points_1d_x,
        gaussian_weights_1d_x,
        gaussian_points_1d_y,
        gaussian_weights_1d_y,
        face_edge_Zmat_real,
        face_edge_Zmat_imag,
        wavenumber);
}

void cu_interpolate_Icoeff_to_Js_3d_forward(
    const torch::Tensor &verts,
    const torch::Tensor &faces,
    const torch::Tensor &face_edge_indices,
    const torch::Tensor &face_areas,
    const torch::Tensor &edge_I_coeff_real,
    const torch::Tensor &edge_I_coeff_imag,
    torch::Tensor &face_vert_Js_real,
    torch::Tensor &face_vert_Js_imag
);

void interpolate_Icoeff_to_Js_3d_forward(
    const torch::Tensor &verts,
    const torch::Tensor &faces,
    const torch::Tensor &face_edge_indices,
    const torch::Tensor &face_areas,
    const torch::Tensor &edge_I_coeff_real,
    const torch::Tensor &edge_I_coeff_imag,
    torch::Tensor &face_vert_Js_real,
    torch::Tensor &face_vert_Js_imag
) {
    cu_interpolate_Icoeff_to_Js_3d_forward(
        verts,
        faces,
        face_edge_indices,
        face_areas,
        edge_I_coeff_real,
        edge_I_coeff_imag,
        face_vert_Js_real,
        face_vert_Js_imag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_MOM_rhs_3d_forward", &create_MOM_rhs_3d_forward, "Forward: Create MOM-based RHS in 3D");
    m.def("create_MOM_Zmat_3d_forward", &create_MOM_Zmat_3d_forward, "Forward: Create MOM-based Zmat in 3D");
    m.def("interpolate_Icoeff_to_Js_3d_forward", &interpolate_Icoeff_to_Js_3d_forward, "Forward: Interpolate edge-based I_coeff to Js on 3D surface");
}

TORCH_LIBRARY(mesh_MOM_3d, m) {
    m.def("create_MOM_rhs_3d_forward", create_MOM_rhs_3d_forward);
    m.def("create_MOM_Zmat_3d_forward", create_MOM_Zmat_3d_forward);
    m.def("interpolate_Icoeff_to_Js_3d_forward", interpolate_Icoeff_to_Js_3d_forward);
}