import json
import os
from typing import Any, Dict, Optional, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg

from garmentimage.utils.neural_tailor_converter import NeuralTailorConverter
from garmentimage.utils.seam import Seam
from garmentimage.utils.template import Template


def polygon_to_svg_file(
    polygon: Union[sg.Polygon, Dict[str, sg.Polygon]],
    filepath: Optional[str] = None,
    color: str = "#E3AFBA",
    maxx: Optional[float] = None,
    maxy: Optional[float] = None,
    minx: Optional[float] = None,
    miny: Optional[float] = None,
) -> str:
    """
    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    filepath : str
    color : str
    """
    if isinstance(polygon, dict):
        # Collect all polygons
        polygons = list(polygon.values())

        # Compute global bounding box
        all_bounds = [p.bounds for p in polygons]
        minx = min(b[0] for b in all_bounds) if minx is None else minx
        miny = min(b[1] for b in all_bounds) if miny is None else miny
        maxx = max(b[2] for b in all_bounds) if maxx is None else maxx
        maxy = max(b[3] for b in all_bounds) if maxy is None else maxy

        margin = 10
        width = (maxx - minx) + 2 * margin
        height = (maxy - miny) + 2 * margin

        polygon_elements = []
        for key, poly in polygon.items():
            points = []
            for x, y in poly.exterior.coords:
                sx = (x - minx) + margin
                sy = (maxy - y) + margin
                points.append(f"{sx},{sy}")
            points_str = " ".join(points)

            # Create a polygon element
            polygon_elements.append(
                f'<polygon points="{points_str}" fill="{color}" stroke="black" stroke-width="1" />'
            )

        # Construct final SVG content
        svg_content = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
{"".join(polygon_elements)}
</svg>"""

    else:
        # bbox
        if maxx is None or maxy is None or minx is None or miny is None:
            minx, miny, maxx, maxy = polygon.bounds

        # add margin to bbox
        margin = 10
        width = (maxx - minx) + 2 * margin
        height = (maxy - miny) + 2 * margin

        # polygon.exterior.coords to SVG points
        points = []
        for x, y in polygon.exterior.coords:
            sx = (x - minx) + margin
            sy = (maxy - y) + margin
            points.append(f"{sx},{sy}")

        points_str = " ".join(points)

        # SVG content
        svg_content = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <polygon points="{points_str}" fill="{color}" stroke="black" stroke-width="1" />
            </svg>"""

    if filepath is not None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_content)

    return svg_content


def convert_spec_json_to_polygon_and_save(
    spec_json: Dict[str, Any], output_file_path: str
) -> None:
    with open(output_file_path + "_specification.json", "w") as f:
        json.dump(spec_json, f, indent=4)
    panel_to_polygon = NeuralTailorConverter.spec_json_to_panel_to_polygon(spec_json)
    front_panel_to_polygon = {
        panel_name: polygon
        for panel_name, polygon in panel_to_polygon.items()
        if NeuralTailorConverter.judge_front(panel_name)
    }
    back_panel_to_polygon = {
        panel_name: polygon
        for panel_name, polygon in panel_to_polygon.items()
        if not NeuralTailorConverter.judge_front(panel_name)
    }
    polygon_to_svg_file(
        front_panel_to_polygon,
        output_file_path.replace(".svg", ".png").replace(".png", "_pattern_front.svg"),
    )
    polygon_to_svg_file(
        back_panel_to_polygon,
        output_file_path.replace(".svg", ".png").replace(".png", "_pattern_back.svg"),
    )


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
