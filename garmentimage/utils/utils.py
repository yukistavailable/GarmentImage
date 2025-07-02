from __future__ import annotations

import os
from typing import Dict, List, Union

from dotenv import load_dotenv
import numpy as np
import torch

load_dotenv(override=True)
GARMENT_IMAGE_RESOLUTION = int(os.getenv("GARMENT_IMAGE_RESOLUTION", 16))
print(f"GARMENT_IMAGE_RESOLUTION: {GARMENT_IMAGE_RESOLUTION}")
TEMPLATE_W = int(os.getenv("TEMPLATE_W", 512))

EdgeInfoType = List[Dict[str, Union[np.ndarray, Dict, int]]]
PanelToEdgeInfoType = Dict[str, EdgeInfoType]
Number = Union[int, float, np.floating]
GarmentImageType = Union[str, np.ndarray, torch.Tensor]


def curvature_3d_quadratic_bezier(p0, p1, p2, t) -> float:
    # first derivative B'(t)
    B1 = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)
    # second derivative B''(t)
    B2 = 2 * (p2 - 2 * p1 + p0)

    # B'(t) x B''(t)
    cross_val = np.cross(B1, B2)
    cross_norm = np.linalg.norm(cross_val)

    # ||B'(t)||
    B1_norm = np.linalg.norm(B1)

    # curvature k(t) = |B'(t) x B''(t)| / ||B'(t)||^3
    if B1_norm == 0:
        return 0.0
    else:
        return cross_norm / (B1_norm**3)
