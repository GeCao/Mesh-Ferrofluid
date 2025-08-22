import sys
import numpy as np
import argparse
import torch
import pathlib
import torch.nn.functional as F

sys.path.append("../")
from src.utils import LoadSingleMesh, mkdir


def save_beast_in(filename, verts, faces):
    """
    Save a surface mesh to BEAST `.in` format.

    Args:
        filename (str): Path to output file
        verts (np.ndarray): Nx3 array of vertex coordinates
        faces (np.ndarray): Mx3 array of triangle indices (0-based or 1-based)
    """
    verts = np.asarray(verts)
    faces = np.asarray(faces)

    # Ensure faces are 1-based (BEAST expects 1-based indices)
    if np.min(faces) == 0:
        faces = faces + 1

    with open(filename, "w") as f:
        # Write number of vertices
        f.write(f"1\n")
        f.write(f"{verts.shape[0]} {faces.shape[0]}\n")
        # Write vertex coordinates
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        # Write number of faces
        # f.write(f"{faces.shape[0]}\n")
        # Write face indices
        for face in faces:
            f.write(f"{face[0]} {face[1]} {face[2]}\n")


# set up the path for saving
path = pathlib.Path(__file__).parent.absolute()
asset_path = f"{path}/../assets/"

# use cuda if exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Load a 3D asset
verts, faces = LoadSingleMesh(
    obj_path=f"{asset_path}/nasaAlmond_v3.ply", device=device, dtype=dtype
)
N_verts, dim = verts.shape
N_faces, dim = faces.shape
np_verts = verts.cpu().numpy()
np_faces = faces.cpu().numpy()

save_beast_in(f"{asset_path}/nasaAlmond_v3.in", np_verts, np_faces)
print("BEAST .in file saved!")
