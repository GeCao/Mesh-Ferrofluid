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


from src.vulkan_ray_tracing import VulkanRayTracing  # Our vulkan-based ray tracer
from src.utils import LoadMeshes, mkdir, SampleSkydomRays, GetRayQueryDepthMap


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
    ray_querier = VulkanRayTracing.RayQueryApp(
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
            im = plt.imshow(depth_maps[key][camera_index], cmap="viridis")
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
