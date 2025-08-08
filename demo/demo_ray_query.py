"""Copyright (C) 2025, Ge Cao
ACEM research group, https://acem.ece.illinois.edu/
All rights reserved.

This software is free for non-commercial, research and evaluation use
under the terms of the LICENSE.md file.

For inquiries contact  gecao2@illinois.edu
"""

import sys
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("../")


from src.vulkan_ray_tracing import VulkanRayQuery  # Our vulkan-based ray tracer
from src.utils import LoadMeshes, mkdir, ComputeFaceNormals, SampleSkydomRays


def GetRayQueryDepthMap(
    vulkan_ray_tracer, ray_o: torch.Tensor, ray_d: torch.Tensor, keepdim: bool = True
) -> torch.Tensor:
    """Given several camera points (ray_o), and ray directions,
    This function will return a depth image for each camera

    n_rays = height * width

    Args:
        vulkan_ray_tracer: VulkanRayQuery
        ray_o: torch.Tensor = [B, dim=3]
        ray_d: torch.Tensor = [(B), n_rays, dim=3]
        keepdim: bool, returns [B, n_rays, 1] when it's True, [B, n_rays] when False
    Returns:
        torch.Tensor = [B, n_rays, (1)], the depth map of this ray query.
        torch.Tensor = [B, n_rays, (3)], the face index map of this ray query.
    """
    dim = 3
    n_rays = ray_d.shape[-2]
    ray_d = ray_d + torch.zeros_like(ray_o).unsqueeze(-2)

    np_ray_o = ray_o.flatten().to(torch.float32).cpu().numpy()  # [B * dim]
    np_ray_d = ray_d.flatten().to(torch.float32).cpu().numpy()  # [B * HW * dim]
    np_geom_mat = vulkan_ray_tracer.QueryForNLOS(np_ray_o, np_ray_d)  # [B * HW * 4]
    np_geom_mat = np.asarray(np_geom_mat).reshape(*(ray_o.shape[:-1]), n_rays, 4)
    np_geom_face = np_geom_mat[..., 0:3]
    np_geom_depth = np_geom_mat[..., 3:4]
    geom_face = torch.from_numpy(np_geom_face).to(ray_o.device).to(torch.int32)
    geom_depth = torch.from_numpy(np_geom_depth).to(ray_o.device).to(ray_o.dtype)

    if not keepdim:
        geom_depth = geom_depth.squeeze(-1)

    return geom_depth, geom_face


def initialize_vulkan_ray_querier(
    scene_name: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    # Find the asset path and load some triangle mesh (vertices and faces) from them.
    # You will load all the triangle meshes from ../assets/{scene_name}/objs/*.{obj, ply, json, dae}
    asset_dir = "../assets/"
    scene_dir = os.path.join(asset_dir, scene_name)
    verts_list, faces_list, _, _, _, _ = LoadMeshes(
        asset_path=scene_dir,
        with_primitives=False,
        obj_folder="objs",
        device=device,
        dtype=dtype,
    )
    assert len(verts_list) == 1
    assert len(faces_list) == 1
    torch_verts = verts_list[0]
    torch_faces = faces_list[0]
    np_verts = torch_verts.flatten().to(torch.float32).cpu().numpy()  # [N_verts * 3]
    np_faces = torch_faces.flatten().to(torch.uint32).cpu().numpy()  # [N_faces * 3]

    # Initialize our ray querier
    print(f"\n=============== Start Initialize the Vulkan ray querier ===========\n")
    raygenShaderPath = "../src/vulkan_ray_tracing/spv/raygen.spv"
    missShaderPath = "../src/vulkan_ray_tracing/spv/miss.spv"
    chitShaderPath = "../src/vulkan_ray_tracing/spv/closesthit.spv"
    ray_querier = VulkanRayQuery.RayQueryApp(
        np_verts, np_faces, raygenShaderPath, missShaderPath, chitShaderPath
    )
    print(f"\n=============== Finished Initialize the Vulkan ray querier ===========\n")

    return ray_querier, torch_verts, torch_faces


def main(scene_name: str):
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the save directory if it does not exist.
    save_dir = "../save/"
    mkdir(save_dir)
    save_dir = save_dir + "ray_query/"
    mkdir(save_dir)

    # define your camera position (ray origin)
    ray_o = torch.Tensor([[0.0, -8.0, 10.0]]).to(device).to(dtype)
    N_cameras = ray_o.numel() // 3

    # For each camera, define the image size and ray directions.
    Width = 512  # The width of final depth image
    Height = Width // 2  # The height of final depth image
    n_rays = Height * Width  # The total pixels from the final image
    ray_d, theta, phi, _, _ = SampleSkydomRays(
        n_rays=n_rays, device=device, dtype=dtype
    )

    ray_querier, _, _ = initialize_vulkan_ray_querier(
        scene_name=scene_name, device=device, dtype=dtype
    )

    # Test FPS of this ray querier
    start_time = time.time()
    for i in range(10):
        # occlusion_matrix = ray_querier.QueryForLOS(txs, rxs)

        # 1. Ray Query
        # You might noticed this return matrix has 4 channels:
        # When the ray did not hit any point, it will give back to you: [0, 0, 0, -1]
        # When the ray did hit a scene vertex, it will give back to you: [f1, f2, f3, depth]
        ray_query_depth_map, _ = GetRayQueryDepthMap(
            vulkan_ray_tracer=ray_querier, ray_o=ray_o, ray_d=ray_d, keepdim=False
        )
        ray_query_depth_map = ray_query_depth_map.reshape(N_cameras, Height, Width)
    end_time = time.time()
    fps = 10 / (end_time - start_time)
    print(f"fps = {fps} for {Width}x{Height}")
    print(f"Because the frequent CPU-GPU communication, this is not the best speed.")
    print(f"We are still optimizing the speed ...")

    # The remaining for visualization
    depth_maps = {"RayQuery": ray_query_depth_map.cpu().numpy()}
    for camera_index in range(N_cameras):
        for key in depth_maps:
            plt.figure()
            fig, axs = plt.subplots(1, 1)
            im = plt.imshow(depth_maps[key][camera_index], cmap="jet")
            axs.set_title(f"Depth Map by {key}")
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = plt.colorbar(im, cax=cax)
            clb.ax.set_title("depth")
            save_filepath = os.path.join(save_dir, f"{key}_depth_map{camera_index}")
            plt.savefig(save_filepath, pad_inches=0, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    scene_name = "human"
    main(scene_name=scene_name)
