import numpy as np
import math
import torch
import torch.nn.functional as F
import cv2
from typing import List

from src.utils import GenUVMode, GetThetaAndPhiFromRayDirection


class TaichiTexture(object):
    def __init__(self, filename: str, image: torch.Tensor = None) -> None:
        self.filename = filename
        self.image = image  # [0 - 1], [H, W, 4]

    def initialized(self):
        return self.image is not None

    def LoadImage(
        self,
        filepath: str,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ) -> torch.Tensor:
        np_image = cv2.imread(filepath)  # [0 - 255], [H, W, n_ch]
        np_image = np_image / 255
        np_image = np.concatenate(
            (
                np_image[..., 2:3],
                np_image[..., 1:2],
                np_image[..., 0:1],
                np_image[..., 3:],
            ),
            axis=-1,
        )
        if np_image.shape[-1] == 3:
            np_image = np.concatenate(
                (np_image, np.ones_like(np_image[..., 0:1])), axis=-1
            )

        self.image = torch.from_numpy(np_image).to(dtype).to(device)
        return self.image

    @staticmethod
    def GenerateUVs(
        verts: List[torch.Tensor],
        faces: List[torch.Tensor],
        mode: int = int(GenUVMode.SPHERICAL_HARMONICS),
    ) -> List[torch.Tensor]:
        vert_uvs = []
        if mode == int(GenUVMode.SPHERICAL_HARMONICS):
            for i in range(len(verts)):
                normalized_verts = F.normalize(verts[i], dim=-1)
                theta, phi = GetThetaAndPhiFromRayDirection(normalized_verts)
                theta = theta / math.pi
                phi = phi / (2.0 * math.pi)
                uv = torch.cat((theta, phi), dim=-1)
                vert_uvs.append(uv)
        elif mode == int(GenUVMode.PLANAR):
            for i in range(len(verts)):
                verts_min, _ = verts[i].min(dim=0)
                verts_max, _ = verts[i].max(dim=0)
                normalized_verts = (verts - verts_min) / (verts_max - verts_min)
                uv = normalized_verts[..., 0:2]
                vert_uvs.append(uv)
        elif mode == int(GenUVMode.BOX):
            for i in range(len(verts)):
                verts_min, _ = verts[i].min(dim=0)
                verts_max, _ = verts[i].max(dim=0)
                normalized_verts = (verts - verts_min) / (verts_max - verts_min)
                uv, _ = torch.sort(normalized_verts, dim=-1)
                uv = uv[..., 0:2]
                vert_uvs.append(uv)
        else:
            raise RuntimeError("Erro when generating UV for mesh: un-recognized mode")

        # from vertice UVs to face UVs
        face_uvs = [None for i in range(len(faces))]
        for i in range(len(face_uvs)):
            # face_uvs[i] = (
            #     vert_uvs[i][faces[i].flatten().tolist()]
            #     .reshape(faces[i].shape[0], 3, 2)
            #     .sum(dim=-2)
            # )
            face_uvs[i] = faces[i] + 0

        return vert_uvs, face_uvs
