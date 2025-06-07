from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from garmentimage.utils.seam import Seam
    from garmentimage.utils.template import Template

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


def visualize_np_garmentimage(
    np_garmentimage: np.ndarray,
    output_file_path: Optional[str] = None,
    markersize: int = 5,
    axis_off: bool = True,
    only_deformations: bool = False,
    deform_scale: float = 1.0,
) -> None:
    alpha = 0.5
    color = "gray"

    C, W, H = np_garmentimage.shape
    scale = Template.W / W

    for i in range(2):
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        tmp_output_file_path = None
        if output_file_path is not None:
            if i == 0:
                tmp_output_file_path = output_file_path.replace(".png", "_front.png")
            else:
                tmp_output_file_path = output_file_path.replace(".png", "_back.png")
        start_index = i * 17
        for x in range(W):
            for y in range(W):
                scaled_x = x * scale
                scaled_y = y * scale
                scaled_x_plus_1 = (x + 1) * scale
                scaled_y_plus_1 = (y + 1) * scale
                if np_garmentimage[start_index + 0, x, y] == 1:
                    x_axis_seam_type = np.argmax(
                        np_garmentimage[start_index + 9 : start_index + 13, x, y]
                    )
                    x_axis_linestyle = Seam.boundary_types_to_linestyle[
                        x_axis_seam_type
                    ]
                    x_axis_linecolor = Seam.boundary_types_to_color[x_axis_seam_type]
                    ax.plot(
                        [scaled_x, scaled_x_plus_1],
                        [scaled_y, scaled_y],
                        marker=None,
                        linestyle=x_axis_linestyle,
                        color=x_axis_linecolor,
                        alpha=1.0,
                        markersize=markersize,
                    )
                    y_axis_seam_type = np.argmax(
                        np_garmentimage[start_index + 13 : start_index + 17, x, y]
                    )
                    y_axis_linestyle = Seam.boundary_types_to_linestyle[
                        y_axis_seam_type
                    ]
                    y_axis_linecolor = Seam.boundary_types_to_color[y_axis_seam_type]
                    ax.plot(
                        [scaled_x, scaled_x],
                        [scaled_y, scaled_y_plus_1],
                        marker=None,
                        linestyle=y_axis_linestyle,
                        color=y_axis_linecolor,
                        alpha=1.0,
                        markersize=markersize,
                    )
                    deform_bottom = (
                        np_garmentimage[start_index + 1 : start_index + 3, x, y] * scale
                    )
                    deform_left = (
                        np_garmentimage[start_index + 3 : start_index + 5, x, y] * scale
                    )
                    deform_top = (
                        np_garmentimage[start_index + 5 : start_index + 7, x, y] * scale
                    )
                    deform_right = (
                        np_garmentimage[start_index + 7 : start_index + 9, x, y] * scale
                    )
                    x_vector = (deform_bottom + deform_top) / 2 * deform_scale
                    y_vector = (deform_left + deform_right) / 2 * deform_scale

                    origin = np.array([scaled_x, scaled_y])
                    dest = origin + x_vector + y_vector
                    mid = (origin + dest) / 2
                    new_origin = (
                        origin + np.array([scaled_x + scale / 2, scaled_y + scale / 2])
                    ) - mid
                    parallelogram_points = [
                        new_origin,
                        new_origin + x_vector,
                        new_origin + x_vector + y_vector,
                        new_origin + y_vector,
                    ]
                    polygon = patches.Polygon(
                        parallelogram_points,
                        closed=True,
                        edgecolor=color,
                        facecolor=color,
                        alpha=alpha,
                    )
                    ax.add_patch(polygon)

                else:
                    if x != 0 and np_garmentimage[start_index + 0, x - 1, y] == 1:
                        y_axis_seam_type = np.argmax(
                            np_garmentimage[start_index + 13 : start_index + 17, x, y]
                        )
                        y_axis_linestyle = Seam.boundary_types_to_linestyle[
                            y_axis_seam_type
                        ]
                        y_axis_linecolor = Seam.boundary_types_to_color[
                            y_axis_seam_type
                        ]
                        ax.plot(
                            [scaled_x, scaled_x],
                            [scaled_y, scaled_y_plus_1],
                            marker=None,
                            linestyle=y_axis_linestyle,
                            color=y_axis_linecolor,
                            alpha=1.0,
                            markersize=markersize,
                        )
                    if y != 0 and np_garmentimage[start_index + 0, x, y - 1] == 1:
                        x_axis_seam_type = np.argmax(
                            np_garmentimage[start_index + 9 : start_index + 13, x, y]
                        )
                        x_axis_linestyle = Seam.boundary_types_to_linestyle[
                            x_axis_seam_type
                        ]
                        x_axis_linecolor = Seam.boundary_types_to_color[
                            x_axis_seam_type
                        ]
                        ax.plot(
                            [scaled_x, scaled_x_plus_1],
                            [scaled_y, scaled_y],
                            marker=None,
                            linestyle=x_axis_linestyle,
                            color=x_axis_linecolor,
                            alpha=1.0,
                            markersize=markersize,
                        )

        ax.set_xticks(
            range(0, Template.W + 1),
            minor=True,
        )
        ax.set_yticks(
            range(0, Template.W + 1),
            minor=True,
        )
        if axis_off:
            ax.axis("off")
        if output_file_path is not None:
            dir_path = os.path.dirname(tmp_output_file_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(tmp_output_file_path)
            fig.savefig(tmp_output_file_path.replace(".png", ".svg"), format="svg")
        else:
            plt.show()
        # close the figure to prevent memory leak
        plt.close(fig)
