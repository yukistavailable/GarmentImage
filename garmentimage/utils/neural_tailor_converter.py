import json
import os
import queue
import warnings
from collections import defaultdict
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
from dotenv import load_dotenv
from scipy.spatial.distance import cdist

from garmentimage.utils.piece import Piece
from garmentimage.utils.seam import Seam
from garmentimage.utils.template import Template
from garmentimage.utils.utils import (
    EdgeInfoType,
    Number,
    PanelToEdgeInfoType,
    curvature_3d_quadratic_bezier,
)
from garmentimage.utils.vertex2d import Vertex2D

load_dotenv(override=True)
GARMENT_IMAGE_RESOLUTION = int(os.getenv("GARMENT_IMAGE_RESOLUTION", 16))
print(f"GARMENT_IMAGE_RESOLUTION: {GARMENT_IMAGE_RESOLUTION}")
TEMPLATE_W = int(os.getenv("TEMPLATE_W", 512))


class NeuralTailorConverter:
    rotation_matrix_z_90: np.ndarray = np.array(
        [
            [
                np.cos(np.radians(90)),
                -np.sin(np.radians(90)),
                0,
                0,
            ],
            [
                np.sin(np.radians(90)),
                np.cos(np.radians(90)),
                0,
                0,
            ],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    rotation_matrix_z_270: np.ndarray = np.array(
        [
            [
                np.cos(np.radians(270)),
                -np.sin(np.radians(270)),
                0,
                0,
            ],
            [
                np.sin(np.radians(270)),
                np.cos(np.radians(270)),
                0,
                0,
            ],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    @staticmethod
    def construct_stitches_from_panel_to_edge_info(
        panel_to_edge_info: PanelToEdgeInfoType,
    ) -> List[List[Dict[str, Union[int, str]]]]:
        # construct stitches
        new_stitches_hash = set()
        new_stitches = []
        for panel_id in panel_to_edge_info:
            for i, edge_info in enumerate(panel_to_edge_info[panel_id]):
                if (
                    edge_info.get("stitch_info")
                    and edge_info["stitch_info"]["stitch_type"] != Seam.BOUNDARY
                ):
                    target_edge_index = edge_info["stitch_info"].get(
                        "target_edge_index"
                    )
                    target_edge_panel = edge_info["stitch_info"].get(
                        "target_edge_panel"
                    )
                    if target_edge_index is None or target_edge_panel is None:
                        continue
                    if panel_id == target_edge_panel:
                        stitch_hash = (
                            f"{panel_id}_{i}_{target_edge_panel}_{target_edge_index}"
                            if i < target_edge_index
                            else f"{target_edge_panel}_{target_edge_index}_{panel_id}_{i}"
                        )
                    else:
                        stitch_hash = (
                            f"{panel_id}_{i}_{target_edge_panel}_{target_edge_index}"
                            if panel_id < target_edge_panel
                            else f"{target_edge_panel}_{target_edge_index}_{panel_id}_{i}"
                        )
                    if stitch_hash in new_stitches_hash:
                        continue
                    new_stitches_hash.add(stitch_hash)
                    stitch = [
                        {
                            "edge": edge_info["edge_index"],
                            "panel": panel_id,
                        },
                        {
                            "edge": target_edge_index,
                            "panel": target_edge_panel,
                        },
                    ]
                    new_stitches.append(stitch)
        return new_stitches

    @staticmethod
    def convert_to_panel_to_edge_info(
        template: Template,
        garment_type: Optional[str] = None,
        strict_garment_type: bool = True,
        is_front: bool = True,
        predefined_scale: float = 2.75,
        min_x: float = TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        max_x: float = TEMPLATE_W - TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        min_y: float = TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        max_y: float = TEMPLATE_W - TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
    ) -> PanelToEdgeInfoType:
        named_meshes = Template.assign_piece_or_mesh_names(
            template.meshes,
            garment_type,
            strict_garment_type=strict_garment_type,
            is_front=is_front,
        )
        side_by_side_stitched_edges_pairs = template.side_by_side_stitched_edges_pairs

        panel_to_edge_info: PanelToEdgeInfoType = {}
        for i in range(len(named_meshes)):
            panel_name, mesh = named_meshes[i]
            sorted_boundary_edges = mesh.get_boundary_edges()
            edge_info: EdgeInfoType = []
            if is_front:
                for i, edge in enumerate(sorted_boundary_edges):
                    start = edge.start.to_np_four_dim()
                    end = edge.end.to_np_four_dim()
                    start[0] = (start[0] - (min_x + max_x) / 2) / predefined_scale
                    start[1] = (start[1] - (min_y + max_y) / 2) / predefined_scale
                    end[0] = (end[0] - (min_x + max_x) / 2) / predefined_scale
                    end[1] = (end[1] - (min_y + max_y) / 2) / predefined_scale
                    _edge_info = {
                        "global_edge_index": edge.global_index,
                        "start": start,
                        "end": end,
                        "start_index": i % len(sorted_boundary_edges),
                        "end_index": (i + 1) % len(sorted_boundary_edges),
                        "edge_index": i,
                        "panel_name": panel_name,
                    }
                    if edge.seam_type != Seam.NONE:
                        _edge_info["stitch_info"] = {
                            "stitch_type": edge.seam_type,
                        }

                    edge_info.append(_edge_info)
            else:
                for i, edge in enumerate(reversed(sorted_boundary_edges)):
                    start = edge.start.to_np_four_dim()
                    end = edge.end.to_np_four_dim()
                    start[0] = (start[0] - (min_x + max_x) / 2) / predefined_scale
                    start[1] = (start[1] - (min_y + max_y) / 2) / predefined_scale
                    end[0] = (end[0] - (min_x + max_x) / 2) / predefined_scale
                    end[1] = (end[1] - (min_y + max_y) / 2) / predefined_scale
                    _edge_info = {
                        "global_edge_index": edge.global_index,
                        "end": start,
                        "start": end,
                        "start_index": i % len(sorted_boundary_edges),
                        "end_index": (i + 1) % len(sorted_boundary_edges),
                        "edge_index": i,
                        "panel_name": panel_name,
                    }
                    if edge.seam_type != Seam.NONE:
                        _edge_info["stitch_info"] = {
                            "stitch_type": edge.seam_type,
                        }

                    edge_info.append(_edge_info)
            panel_to_edge_info[panel_name] = edge_info

        # add SIDE_BY_SIDE stitches
        used_edges = set()
        for edges_pair in side_by_side_stitched_edges_pairs:
            edge_1_global_index, edge_2_global_index = edges_pair
            if edge_1_global_index in used_edges or edge_2_global_index in used_edges:
                continue
            used_edges.add(edge_1_global_index)
            used_edges.add(edge_2_global_index)

            panel_name_1: Optional[str] = None
            panel_name_2: Optional[str] = None
            edge_index_1: Optional[int] = None
            edge_index_2: Optional[int] = None
            edge_info_1: Optional[Dict] = None
            edge_info_2: Optional[Dict] = None
            for panel_name in panel_to_edge_info.keys():
                for i, edge_info in enumerate(panel_to_edge_info[panel_name]):
                    if edge_info["global_edge_index"] is None:
                        continue
                    if (
                        panel_name_1 is None
                        and edge_info["global_edge_index"] == edge_1_global_index
                    ):
                        panel_name_1 = edge_info["panel_name"]
                        edge_index_1 = edge_info["edge_index"]
                        edge_info_1 = edge_info
                        break
                    elif edge_info["global_edge_index"] == edge_2_global_index:
                        panel_name_2 = edge_info["panel_name"]
                        edge_index_2 = edge_info["edge_index"]
                        edge_info_2 = edge_info
                        break
                if panel_name_1 is not None and panel_name_2 is not None:
                    break
            if panel_name_1 is None or panel_name_2 is None:
                continue
            edge_info_1["stitch_info"]["target_edge_panel"] = panel_name_2
            edge_info_1["stitch_info"]["target_edge_index"] = edge_index_2
            edge_info_2["stitch_info"]["target_edge_panel"] = panel_name_1
            edge_info_2["stitch_info"]["target_edge_index"] = edge_index_1

        return panel_to_edge_info

    @staticmethod
    def add_front_to_back_stitch_info(
        panel_to_edge_info: PanelToEdgeInfoType,
        template_front: Template,
        template_back: Template,
    ):
        front_to_back_stitched_edges_pairs: List[Tuple[int, int]] = (
            template_front.front_to_back_stitched_edges_pairs
        )
        used_edges_front = set()
        used_edges_back = set()
        for edges_pair in front_to_back_stitched_edges_pairs:
            edge_global_index_front, edge_global_index_back = edges_pair
            if (
                edge_global_index_front in used_edges_front
                or edge_global_index_back in used_edges_back
            ):
                continue
            used_edges_front.add(edge_global_index_front)
            used_edges_back.add(edge_global_index_back)

            panel_name_front: Optional[str] = None
            panel_name_back: Optional[str] = None
            edge_index_front: Optional[int] = None
            edge_index_back: Optional[int] = None
            edge_info_front: Optional[Dict] = None
            edge_info_back: Optional[Dict] = None
            for panel_name in panel_to_edge_info.keys():
                is_front = NeuralTailorConverter.judge_front(panel_name)
                for i, edge_info in enumerate(panel_to_edge_info[panel_name]):
                    if edge_info["global_edge_index"] is None:
                        continue
                    if (
                        is_front
                        and panel_name_front is None
                        and edge_info["global_edge_index"] == edge_global_index_front
                    ):
                        panel_name_front = edge_info["panel_name"]
                        edge_index_front = edge_info["edge_index"]
                        edge_info_front = edge_info
                        break
                    elif (
                        not is_front
                        and panel_name_back is None
                        and edge_info["global_edge_index"] == edge_global_index_back
                    ):
                        panel_name_back = edge_info["panel_name"]
                        edge_index_back = edge_info["edge_index"]
                        edge_info_back = edge_info
                        break
                if panel_name_front is not None and panel_name_back is not None:
                    break
            if panel_name_front is None or panel_name_back is None:
                continue
            edge_info_front["stitch_info"]["target_edge_panel"] = panel_name_back
            edge_info_front["stitch_info"]["target_edge_index"] = edge_index_back
            edge_info_back["stitch_info"]["target_edge_panel"] = panel_name_front
            edge_info_back["stitch_info"]["target_edge_index"] = edge_index_front

        return panel_to_edge_info

    @staticmethod
    def convert_panel_to_edge_info_to_panel_to_polygon(
        panel_to_edge_info: PanelToEdgeInfoType, bezier_subdivision: int = 10
    ) -> Dict[str, sg.Polygon]:
        panel_to_polygon: Dict[str, sg.Polygon] = {}
        for panel_name in panel_to_edge_info.keys():
            edge_infos = panel_to_edge_info[panel_name]
            seams: List[Seam] = []
            for edge_info in edge_infos:
                seam: Seam = NeuralTailorConverter.construct_seam_from_edge_info(
                    edge_info, bezier_subdivision=bezier_subdivision
                )
                seams.append(seam)
            polygon = Seam.to_sg_polygon(seams)
            panel_to_polygon[panel_name] = polygon
        return panel_to_polygon

    @staticmethod
    def spec_json_to_panel_to_polygon(
        specification: Union[str, Dict[str, Any]],
    ) -> Dict[str, sg.Polygon]:
        if isinstance(specification, str):
            with open(specification, "r") as f:
                data = json.load(f)
        else:
            data = specification

        panels: Dict = data["pattern"]["panels"]
        front_panels = {
            k: v for k, v in panels.items() if NeuralTailorConverter.judge_front(k)
        }
        back_panels = {
            k: v for k, v in panels.items() if not NeuralTailorConverter.judge_front(k)
        }
        front_panel_to_edge_info = NeuralTailorConverter.construct_panel_to_edge_info(
            front_panels
        )
        back_panel_to_edge_info = NeuralTailorConverter.construct_panel_to_edge_info(
            back_panels
        )
        panel_to_edge_info = {**front_panel_to_edge_info, **back_panel_to_edge_info}
        panel_to_polygon = (
            NeuralTailorConverter.convert_panel_to_edge_info_to_panel_to_polygon(
                panel_to_edge_info
            )
        )

        return panel_to_polygon

    @staticmethod
    def construct_specification_json_from_panel_to_edge_info(
        panel_to_edge_info: PanelToEdgeInfoType,
        panel_id_to_translation: Dict[str, List[float]],
        panel_id_to_rotation: Dict[str, List[float]],
        properties: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        parameter_order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        new_data = {}
        new_data["pattern"] = {}
        if properties is not None:
            new_data["properties"] = properties
        if parameters is not None:
            new_data["parameters"] = parameters
        if parameter_order is not None:
            new_data["parameter_order"] = parameter_order
        stitches = NeuralTailorConverter.construct_stitches_from_panel_to_edge_info(
            panel_to_edge_info=panel_to_edge_info
        )
        new_data["pattern"]["stitches"] = stitches

        new_panels = {}
        new_data["pattern"]["panels"] = new_panels
        for panel_id, edge_infos in panel_to_edge_info.items():
            panel = {}
            try:
                translation = panel_id_to_translation[panel_id]
                rotation = panel_id_to_rotation[panel_id]
            except BaseException:
                # check merged panels
                for _panel_id in panel_id_to_translation.keys():
                    if panel_id.startswith(_panel_id):
                        translation = panel_id_to_translation[_panel_id]
                        rotation = panel_id_to_rotation[_panel_id]
                        break

            panel["translation"] = translation
            panel["rotation"] = rotation

            translation_matrix = NeuralTailorConverter.calculate_translation_matrix(
                translation, do_reverse=False
            )
            rotation_matrix_x, rotation_matrix_y, rotation_matrix_z = (
                NeuralTailorConverter.calculate_rotation_matrices(
                    rotation, do_reverse=False
                )
            )
            affine_matrix: np.ndarray = np.dot(rotation_matrix_y, rotation_matrix_x)
            affine_matrix = np.dot(rotation_matrix_z, affine_matrix)
            affine_matrix = np.dot(translation_matrix, affine_matrix)
            affine_matrix_inv = np.linalg.inv(affine_matrix)

            edges = []
            panel["edges"] = edges
            index_to_vertex = {}
            for edge_info in edge_infos:
                edge = {}
                edge["endpoints"] = [edge_info["start_index"], edge_info["end_index"]]
                if "curvature" in edge_info:
                    edge["curvature"] = edge_info["curvature"].tolist()
                edges.append(edge)
                if edge_info["start_index"] not in index_to_vertex:
                    index_to_vertex[edge_info["start_index"]] = edge_info["start"]
                if edge_info["end_index"] not in index_to_vertex:
                    index_to_vertex[edge_info["end_index"]] = edge_info["end"]
            if len(index_to_vertex) == 0:
                panel["vertices"] = []
            else:
                vertices = np.array(
                    [index_to_vertex[i] for i in range(len(index_to_vertex))]
                )
                vertices = np.dot(affine_matrix_inv, vertices.T).T[:, :2]
                panel["vertices"] = vertices.tolist()

            new_panels[panel_id] = panel
        new_data["properties"] = {
            "curvature_coords": "relative",
            "normalize_panel_translation": False,
            "units_in_meter": 100,
            "normalized_edge_loops": True,
        }
        new_data["parameters"] = {}
        return new_data

    @staticmethod
    def construct_panel_to_edge_info_from_json_path_with_stitches(
        spec_json_path: str,
    ) -> PanelToEdgeInfoType:
        with open(spec_json_path, "r") as f:
            data = json.load(f)
        panels: Dict = data["pattern"]["panels"]
        stitches: List[List[Dict[str, Union[int, str]]]] = data["pattern"].get(
            "stitches"
        )
        front_panels = {
            k: v for k, v in panels.items() if NeuralTailorConverter.judge_front(k)
        }
        front_panel_to_edge_info = NeuralTailorConverter.construct_panel_to_edge_info(
            front_panels,
        )
        NeuralTailorConverter.add_stitches_to_edge_info(
            front_panel_to_edge_info, stitches
        )
        back_panels = {
            k: v for k, v in panels.items() if not NeuralTailorConverter.judge_front(k)
        }
        back_panel_to_edge_info = NeuralTailorConverter.construct_panel_to_edge_info(
            back_panels,
        )
        NeuralTailorConverter.add_stitches_to_edge_info(
            back_panel_to_edge_info, stitches
        )
        panel_to_edge_info = front_panel_to_edge_info | back_panel_to_edge_info
        return panel_to_edge_info

    @staticmethod
    def construct_panel_to_edge_info_and_do_preprocess(
        panels: Dict,
        parameters: Optional[Dict] = None,
        parameter_order: Optional[List[str]] = None,
        stitches: Optional[List[List[Dict[str, Union[int, str]]]]] = None,
        align_stitches: bool = False,
        force_alignment: bool = False,
        y_axis_alignment: bool = False,
        dart_alignment: bool = False,
        skirt_alignment: bool = False,
        separate_top: bool = False,
        merged_panels: Optional[Union[List[str], List[List[str]]]] = None,
        removed_panels: Optional[List[str]] = None,
        add_uv: bool = False,
        non_alignment_panels: Optional[List[str]] = None,
        only_horizontally_flat_edge: bool = False,
    ) -> PanelToEdgeInfoType:
        panel_to_edge_info = NeuralTailorConverter.construct_panel_to_edge_info(
            panels,
            parameters=parameters,
            parameter_order=parameter_order,
        )
        if stitches is not None:
            NeuralTailorConverter.add_stitches_to_edge_info(
                panel_to_edge_info, stitches
            )
            if align_stitches:
                NeuralTailorConverter.apply_stitch_alignment(
                    panel_to_edge_info,
                    force_alignment=force_alignment,
                    y_axis_alignment=y_axis_alignment,
                    dart_alignment=dart_alignment,
                    skirt_alignment=skirt_alignment,
                    non_target_panel_names=non_alignment_panels,
                    # align_with_uv=add_uv,
                )
                if merged_panels is not None:
                    if isinstance(merged_panels[0], str):
                        NeuralTailorConverter.merge_edge_info(
                            panel_to_edge_info, merged_panels
                        )
                    else:
                        assert isinstance(merged_panels[0], list)
                        NeuralTailorConverter.merge_edge_info_v2(
                            panel_to_edge_info,
                            merged_panels,
                            only_horizontally_flat_edge=only_horizontally_flat_edge,
                        )
            if removed_panels is not None:
                NeuralTailorConverter.remove_panel_from_edge_info(
                    panel_to_edge_info=panel_to_edge_info,
                    removing_panel_names=removed_panels,
                )
            if separate_top:
                NeuralTailorConverter.separate_top_front_panel(panel_to_edge_info)
                NeuralTailorConverter.separate_top_back_panel(panel_to_edge_info)
            if add_uv:
                NeuralTailorConverter.add_uv_to_edge_info(panel_to_edge_info)
        return panel_to_edge_info

    @staticmethod
    def construct_panel_to_edge_info(
        panels: Dict,
        parameters: Optional[Dict] = None,
        parameter_order: Optional[List[str]] = None,
        removed_panels: Optional[List[str]] = None,
    ) -> PanelToEdgeInfoType:
        panel_to_edge_info: Dict[
            str, List[Dict[str, Union[Vertex2D, np.ndarray, Dict]]]
        ] = {}
        for panel_name in panels.keys():  # type: ignore # type: str
            panel: Dict = panels[panel_name]
            # Extract the relevant information from the JSON data
            vertices: np.ndarray = np.array(panel["vertices"])

            if len(vertices) == 0:
                continue

            # homogenize the vertices in 3D space
            # see https://pdwslmr.netlify.app/posts/3d-prog/affine-transformation/
            vertices = np.hstack(
                (
                    vertices,
                    np.zeros((vertices.shape[0], 1)),
                    np.ones((vertices.shape[0], 1)),
                )
            )

            edges: List[Dict[str, Union[List[int]], List[float]]] = panel["edges"]
            translation: List[float] = panel["translation"]
            rotation: List[float] = panel["rotation"]
            for rotation_element in rotation:
                if rotation_element % 180 != 0:
                    warnings.warn("rotation is not a multiple of 180", UserWarning)

            # Apply translation and rotation to the vertices
            translation_matrix = NeuralTailorConverter.calculate_translation_matrix(
                translation
            )
            rotation_matrix_x, rotation_matrix_y, rotation_matrix_z = (
                NeuralTailorConverter.calculate_rotation_matrices(rotation)
            )

            # TODO: check if the order of the matrices is correct
            affine_matrix: np.ndarray = np.dot(rotation_matrix_y, rotation_matrix_x)
            affine_matrix = np.dot(rotation_matrix_z, affine_matrix)
            affine_matrix = np.dot(translation_matrix, affine_matrix)
            vertices = np.dot(affine_matrix, vertices.T).T
            if panel_name == "pant_front_right" or panel_name == "pant_back_left":
                # flip the x-axis
                vertices[:, 0] = -vertices[:, 0]

            # Construct edge_info
            edge_info: List[Dict[str, np.ndarray]] = []
            for i, edge in enumerate(edges):
                endpoints = edge["endpoints"]
                vertex1: np.ndarray = vertices[endpoints[0]]
                vertex2: np.ndarray = vertices[endpoints[1]]
                is_garmentcodedata = False
                if "curvature" in edge:
                    curvature: Union[List[float], dict] = edge["curvature"]
                    # curvature: np.ndarray = np.array(edge["curvature"])
                    if isinstance(curvature, list):
                        curvature = np.array(curvature)
                        if (
                            panel_name == "pant_front_right"
                            or panel_name == "pant_back_left"
                        ):
                            # flip the x-axis
                            curvature[1] = -curvature[1]
                        control_point = NeuralTailorConverter.calculate_control_point(
                            vertex1, vertex2, curvature
                        )
                    else:
                        is_garmentcodedata = True
                        curvature_type = curvature.get("type")
                        assert curvature_type is not None, "curvature_type is not found"
                        assert curvature_type in ["quadratic", "cubic", "circle"]
                        if curvature_type == "quadratic":
                            curvature = np.array(curvature["params"][0])
                            # print(curvature)
                            control_point = (
                                NeuralTailorConverter.calculate_control_point(
                                    vertex1, vertex2, curvature, is_garmentcodedata=True
                                )
                            )
                        elif curvature_type == "cubic":
                            curvature_1 = np.array(curvature["params"][0])
                            curvature_2 = np.array(curvature["params"][1])
                            # TODO: address the case whre the curvature is cubic
                            curvature = (curvature_1 + curvature_2) / 2
                            control_point = (
                                NeuralTailorConverter.calculate_control_point(
                                    vertex1, vertex2, curvature, is_garmentcodedata=True
                                )
                            )
                        elif curvature_type == "circle":
                            # TODO: address the case where the curvature is a circle
                            # raise NotImplementedError
                            edge_info.append(
                                {
                                    "start": vertices[endpoints[0]],
                                    "end": vertices[endpoints[1]],
                                    "stitch_info": {"stitch_type": Seam.BOUNDARY},
                                    "rotation": rotation,
                                    "start_index": endpoints[0],
                                    "end_index": endpoints[1],
                                    "edge_index": i,
                                    "panel_name": panel_name,
                                }
                            )
                            continue

                    edge_info.append(
                        {
                            "start": vertices[endpoints[0]],
                            "end": vertices[endpoints[1]],
                            "control_point": control_point,
                            "curvature": curvature,
                            "stitch_info": {"stitch_type": Seam.BOUNDARY},
                            "rotation": rotation,
                            "is_garmentcodedata": is_garmentcodedata,
                            "start_index": endpoints[0],
                            "end_index": endpoints[1],
                            "edge_index": i,
                            "panel_name": panel_name,
                        }
                    )
                else:
                    edge_info.append(
                        {
                            "start": vertices[endpoints[0]],
                            "end": vertices[endpoints[1]],
                            "stitch_info": {"stitch_type": Seam.BOUNDARY},
                            "rotation": rotation,
                            "start_index": endpoints[0],
                            "end_index": endpoints[1],
                            "edge_index": i,
                            "panel_name": panel_name,
                        }
                    )

            panel_to_edge_info[panel_name] = edge_info

        if parameters is not None:
            NeuralTailorConverter.parametric_manipulate(
                panel_to_edge_info,
                parameters=parameters,
                parameter_order=parameter_order,
            )

        if removed_panels is not None:
            NeuralTailorConverter.remove_panel_from_edge_info(
                panel_to_edge_info=panel_to_edge_info,
                removing_panel_names=removed_panels,
            )

        return panel_to_edge_info

    @staticmethod
    def calculate_translation_matrix(
        translation: List[float], do_reverse: bool = False
    ) -> np.ndarray:
        if do_reverse:
            translation_matrix: np.ndarray = np.array(
                [
                    [1, 0, 0, -translation[0]],
                    [0, 1, 0, -translation[1]],
                    [0, 0, 1, -translation[2]],
                    [0, 0, 0, 1],
                ]
            )
        else:
            translation_matrix: np.ndarray = np.array(
                [
                    [1, 0, 0, translation[0]],
                    [0, 1, 0, translation[1]],
                    [0, 0, 1, translation[2]],
                    [0, 0, 0, 1],
                ]
            )
        return translation_matrix

    @staticmethod
    def calculate_rotation_matrices(
        rotation: List[float], do_reverse: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if do_reverse:
            rotation_matrix_x: np.ndarray = np.array(
                [
                    [1, 0, 0, 0],
                    [
                        0,
                        np.cos(np.radians(rotation[0])),
                        np.sin(np.radians(rotation[0])),
                        0,
                    ],
                    [
                        0,
                        -np.sin(np.radians(rotation[0])),
                        np.cos(np.radians(rotation[0])),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )
            rotation_matrix_y: np.ndarray = np.array(
                [
                    [
                        np.cos(np.radians(rotation[1])),
                        0,
                        np.sin(np.radians(rotation[1])),
                        0,
                    ],
                    [0, 1, 0, 0],
                    [
                        -np.sin(np.radians(rotation[1])),
                        0,
                        np.cos(np.radians(rotation[1])),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )
            rotation_matrix_z: np.ndarray = np.array(
                [
                    [
                        np.cos(np.radians(rotation[2])),
                        np.sin(np.radians(rotation[2])),
                        0,
                        0,
                    ],
                    [
                        -np.sin(np.radians(rotation[2])),
                        np.cos(np.radians(rotation[2])),
                        0,
                        0,
                    ],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        else:
            rotation_matrix_x: np.ndarray = np.array(
                [
                    [1, 0, 0, 0],
                    [
                        0,
                        np.cos(np.radians(rotation[0])),
                        -np.sin(np.radians(rotation[0])),
                        0,
                    ],
                    [
                        0,
                        np.sin(np.radians(rotation[0])),
                        np.cos(np.radians(rotation[0])),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )
            rotation_matrix_y: np.ndarray = np.array(
                [
                    [
                        np.cos(np.radians(rotation[1])),
                        0,
                        -np.sin(np.radians(rotation[1])),
                        0,
                    ],
                    [0, 1, 0, 0],
                    [
                        np.sin(np.radians(rotation[1])),
                        0,
                        np.cos(np.radians(rotation[1])),
                        0,
                    ],
                    [0, 0, 0, 1],
                ]
            )
            rotation_matrix_z: np.ndarray = np.array(
                [
                    [
                        np.cos(np.radians(rotation[2])),
                        -np.sin(np.radians(rotation[2])),
                        0,
                        0,
                    ],
                    [
                        np.sin(np.radians(rotation[2])),
                        np.cos(np.radians(rotation[2])),
                        0,
                        0,
                    ],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
        return rotation_matrix_x, rotation_matrix_y, rotation_matrix_z

    @staticmethod
    def calculate_control_point(
        vertex1: np.ndarray,
        vertex2: np.ndarray,
        curvature: np.ndarray,
        is_garmentcodedata: bool = False,
    ) -> np.ndarray:
        # see the official implementation: https://github.com/maria-korosteleva/Garment-Pattern-Generator/blob/9b5702605271c087b99472704334ec154e6d07e4/packages/pattern/core.py#L388
        edge_vector = vertex2 - vertex1
        if vertex1[2] > 0 or is_garmentcodedata:
            rotated_edge_vector = np.dot(
                NeuralTailorConverter.rotation_matrix_z_90, edge_vector
            )
        else:
            rotated_edge_vector = np.dot(
                NeuralTailorConverter.rotation_matrix_z_270, edge_vector
            )
        cx = curvature[0] * edge_vector
        cy = curvature[1] * rotated_edge_vector
        control_point = vertex1 + cx + cy
        return control_point

    @staticmethod
    def edge_info_to_sampled_points(
        edge_info: EdgeInfoType, num_points: int
    ) -> Dict[int, np.ndarray]:
        edge_index_to_sampled_points: Dict[int, np.ndarray] = {}
        for _edge_info in edge_info:
            edge_index = _edge_info["edge_index"]
            start = _edge_info.get("start")
            end = _edge_info.get("end")
            curvature = _edge_info.get("curvature")

            t = np.linspace(0, 1, num_points - 1)
            _sampled_points = np.zeros((num_points, 2))
            _sampled_points[0] = start[:2]
            _sampled_points[-1] = end[:2]
            if curvature is not None:
                control_point = NeuralTailorConverter.calculate_control_point(
                    start, end, curvature, is_garmentcodedata=False
                )
                x: np.ndarray = (
                    (1 - t) ** 2 * start[0]
                    + 2 * (1 - t) * t * control_point[0]
                    + t**2 * end[0]
                )
                y: np.ndarray = (
                    (1 - t) ** 2 * start[1]
                    + 2 * (1 - t) * t * control_point[1]
                    + t**2 * end[1]
                )
                _sampled_points[1:-1, 0] = x[1:]
                _sampled_points[1:-1, 1] = y[1:]
            else:
                # interpolate between start and end
                x = (1 - t) * start[0] + t * end[0]
                y = (1 - t) * start[1] + t * end[1]
                _sampled_points[1:-1, 0] = x[1:]
                _sampled_points[1:-1, 1] = y[1:]
            sampled_points = np.array(_sampled_points)
            edge_index_to_sampled_points[edge_index] = sampled_points
        return edge_index_to_sampled_points

    @staticmethod
    def calculate_nearest_edge_index(
        edge_sampled_points: np.ndarray,
        edge_index_to_sampled_points: Dict[int, np.ndarray],
    ) -> int:
        min_dist = float("inf")
        nearest_edge_index = None
        for edge_index, sampled_points in edge_index_to_sampled_points.items():
            # calculate Hausdorff distance between edge_sampled_points and sampled_points
            dists = cdist(edge_sampled_points, sampled_points)
            dist = np.min(dists)
            h_AB = np.max(np.min(dists, axis=1))
            h_BA = np.max(np.min(dists, axis=0))
            dist = max(h_AB, h_BA)

            if dist < min_dist:
                min_dist = dist
                nearest_edge_index = edge_index
        assert nearest_edge_index is not None
        return nearest_edge_index

    @staticmethod
    def set_random_parameter_values(parameters: Dict, skip_parameters: List[str] = []):
        for parameter_key in parameters.keys():
            if parameter_key in skip_parameters:
                continue
            parameter: Dict = parameters[parameter_key]
            parameter_type: str = parameter["type"]
            parameter_range: Union[List[Number, Number], List[List[Number, Number]]] = (
                parameter["range"]
            )
            assert len(parameter_range) == 2, (
                f"parameter_range is not correct, {parameter_range}"
            )
            if parameter_type == "curve":
                # avoid the case where the parameter_value is too big
                parameter_value = np.random.uniform(
                    max(parameter_range[0], 0.90), min(parameter_range[1], 1.10)
                )
            elif parameter_type == "length":
                # avoid the case where the parameter_value is too big
                parameter_value = np.random.uniform(
                    max(parameter_range[0], 0.85), min(parameter_range[1], 1.15)
                )
            else:
                parameter_value = np.random.uniform(
                    parameter_range[0], parameter_range[1]
                )
            parameter["value"] = parameter_value

    @staticmethod
    def parametric_manipulate(
        panel_to_edge_info: PanelToEdgeInfoType,
        parameters: Dict[str, Dict[str, Any]],
        parameter_order: Optional[List[str]] = None,
    ):
        parameter_keys = (
            list(parameters.keys()) if parameter_order is None else parameter_order
        )
        for parameter_key in parameter_keys:
            parameter: Dict = parameters[parameter_key]
            parameter_type: str = parameter["type"]
            parameter_range: Union[List[Number, Number], List[List[Number, Number]]] = (
                parameter["range"]
            )
            parameter_value: Union[Number, Number] = parameter["value"]
            parameter_influence: List[Dict] = parameter["influence"]

            # check the validity of the parameters
            # assert parameter_type in ["length"], f"parameter_type is not supported, {parameter_type}"
            assert len(parameter_range) == 2, (
                f"parameter_range is not correct, {parameter_range}"
            )
            assert parameter_range[0] <= parameter_value <= parameter_range[1], (
                f"parameter_value is out of range, {parameter_value}"
            )

            for influenced_panel in parameter_influence:
                panel_name: str = influenced_panel["panel"]
                if panel_name not in panel_to_edge_info:
                    continue
                is_front = NeuralTailorConverter.judge_front(panel_name)
                influenced_edge_list: List[Dict[str, Union[str, int]]] = (
                    influenced_panel["edge_list"]
                )
                for influenced_edge in influenced_edge_list:
                    if parameter_type == "curve":
                        # see the official implementation: https://github.com/maria-korosteleva/Garment-Pattern-Generator/blob/9b5702605271c087b99472704334ec154e6d07e4/packages/pattern/core.py#L756
                        if isinstance(parameter_value, Number):
                            # if the parameter_value is a scalar, it is the y value of the control point
                            # see https://github.com/maria-korosteleva/Garment-Pattern-Generator/blob/9b5702605271c087b99472704334ec154e6d07e4/docs/template_spec_with_comments.json#L294
                            parameter_value = [1, parameter_value]
                        edge_info = panel_to_edge_info[panel_name][influenced_edge]
                        assert "curvature" in edge_info, "curvature is not in edge_info"
                        new_cx = edge_info["curvature"][0] * parameter_value[0]
                        new_cy = edge_info["curvature"][1] * parameter_value[1]
                        edge_info["curvature"] = np.array([new_cx, new_cy])
                    elif parameter_type in ["length", "additive_length"]:
                        # elif parameter_type in ["additive_length"]:
                        direction: str = influenced_edge.get("direction")
                        along: List[float] = influenced_edge.get("along")
                        edge_indices: Union[int, List[int]] = influenced_edge["id"]
                        if isinstance(edge_indices, int):
                            edge_indices = [edge_indices]
                        if direction in ["start", "end", "both"]:
                            if direction == "start":
                                base_point = panel_to_edge_info[panel_name][
                                    edge_indices[-1]
                                ]["end"]
                                target_point = panel_to_edge_info[panel_name][
                                    edge_indices[0]
                                ]["start"]
                            elif direction == "end":
                                base_point = panel_to_edge_info[panel_name][
                                    edge_indices[0]
                                ]["start"]
                                target_point = panel_to_edge_info[panel_name][
                                    edge_indices[-1]
                                ]["end"]
                            else:
                                assert direction == "both", (
                                    f"direction is not supported, {direction}"
                                )
                                assert parameter_type == "length", (
                                    f"parameter_type is not supported, {parameter_type}"
                                )
                                base_point = (
                                    panel_to_edge_info[panel_name][edge_indices[0]][
                                        "start"
                                    ]
                                    + panel_to_edge_info[panel_name][edge_indices[-1]][
                                        "end"
                                    ]
                                ) / 2
                                target_point = panel_to_edge_info[panel_name][
                                    edge_indices[0]
                                ]["start"]
                            target_base_vector = target_point - base_point
                            target_base_vector_norm = np.linalg.norm(target_base_vector)
                            target_base_vector_normed = (
                                target_base_vector / target_base_vector_norm
                            )
                            for i, edge_index in enumerate(edge_indices):
                                edge_info = panel_to_edge_info[panel_name][edge_index]
                                start = edge_info["start"]
                                end = edge_info["end"]
                                rotation = edge_info["rotation"]
                                rotation_matrix_x: np.ndarray = np.array(
                                    [
                                        [1, 0, 0, 0],
                                        [
                                            0,
                                            np.cos(np.radians(rotation[0])),
                                            -np.sin(np.radians(rotation[0])),
                                            0,
                                        ],
                                        [
                                            0,
                                            np.sin(np.radians(rotation[0])),
                                            np.cos(np.radians(rotation[0])),
                                            0,
                                        ],
                                        [0, 0, 0, 1],
                                    ]
                                )
                                rotation_matrix_y: np.ndarray = np.array(
                                    [
                                        [
                                            np.cos(np.radians(rotation[1])),
                                            0,
                                            -np.sin(np.radians(rotation[1])),
                                            0,
                                        ],
                                        [0, 1, 0, 0],
                                        [
                                            np.sin(np.radians(rotation[1])),
                                            0,
                                            np.cos(np.radians(rotation[1])),
                                            0,
                                        ],
                                        [0, 0, 0, 1],
                                    ]
                                )
                                rotation_matrix_z: np.ndarray = np.array(
                                    [
                                        [
                                            np.cos(np.radians(rotation[2])),
                                            -np.sin(np.radians(rotation[2])),
                                            0,
                                            0,
                                        ],
                                        [
                                            np.sin(np.radians(rotation[2])),
                                            np.cos(np.radians(rotation[2])),
                                            0,
                                            0,
                                        ],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1],
                                    ]
                                )
                                rotation_matrix: np.ndarray = np.dot(
                                    rotation_matrix_z,
                                    np.dot(rotation_matrix_y, rotation_matrix_x),
                                )

                                end_start_vector = end - start
                                sign: float = (
                                    1
                                    if end_start_vector.dot(target_base_vector) > 0
                                    else -1
                                )
                                if parameter_key == "leg_wideness" and is_front:
                                    sign = -sign
                                along_np: Optional[np.ndarray] = (
                                    np.array([*along, 0, 0])
                                    if along is not None
                                    else None
                                )
                                along_np = (
                                    along_np
                                    if along_np is not None
                                    else target_base_vector_normed
                                )
                                if parameter_type == "length":
                                    start += (
                                        (start - base_point).dot(along_np)
                                        * (parameter_value - 1)
                                    ) * along_np
                                    if i == len(edge_indices) - 1:
                                        end += (
                                            (end - base_point).dot(along_np)
                                            * (parameter_value - 1)
                                        ) * along_np
                                else:
                                    if parameter_key == "sllength":
                                        sign = -sign
                                    along_np = np.dot(rotation_matrix, along_np)
                                    assert parameter_type == "additive_length", (
                                        f"parameter_type is not supported, {parameter_type}"
                                    )
                                    start += (
                                        parameter_value
                                        * (-sign)
                                        * (
                                            np.abs((start - base_point).dot(along_np))
                                            / np.abs(target_base_vector.dot(along_np))
                                        )
                                        * along_np
                                    )
                                    if i == len(edge_indices) - 1:
                                        end += (
                                            parameter_value
                                            * sign
                                            * (
                                                np.abs((end - base_point).dot(along_np))
                                                / np.abs(
                                                    target_base_vector.dot(along_np)
                                                )
                                            )
                                            * along_np
                                        )

    @staticmethod
    def parallel_to_y_axis_edge_info(
        edge_info: Dict[str, Union[np.ndarray, Dict]],
    ) -> bool:
        start: np.ndarray = edge_info.get("start")
        end: np.ndarray = edge_info.get("end")
        return start[0] == end[0]

    @staticmethod
    def parallel_to_x_axis_edge_info(
        edge_info: Dict[str, Union[np.ndarray, Dict]],
    ) -> bool:
        start: np.ndarray = edge_info.get("start")
        end: np.ndarray = edge_info.get("end")
        return start[1] == end[1]

    @staticmethod
    def add_uv_to_edge_info(
        panel_to_edge_info: PanelToEdgeInfoType,
    ):
        for panel_name in panel_to_edge_info.keys():
            # if panel_name not in ["top_front"]:
            #     continue
            edge_info = panel_to_edge_info[panel_name]
            uv_vertices = np.zeros((len(edge_info), 4))
            NeuralTailorConverter.initialize_uv(edge_info, uv_vertices=uv_vertices)
            NeuralTailorConverter.calculate_uv(
                edge_info,
                uv_vertices=uv_vertices,
                target_stitch_types=[Seam.SIDE_BY_SIDE],
                iterations=1,
            )
            NeuralTailorConverter.calculate_uv(
                edge_info,
                uv_vertices=uv_vertices,
                target_stitch_types=[Seam.NONE, Seam.BOUNDARY, Seam.FRONT_TO_BACK],
                iterations=10,
            )

    @staticmethod
    def calculate_uv(
        edge_info: List[Dict[str, Union[np.ndarray, Dict]]],
        uv_vertices: np.ndarray,
        target_stitch_types: List[str],
        iterations: int = 10,
    ):
        for j in range(iterations):
            for i in range(len(edge_info)):
                edge = edge_info[i % len(edge_info)]
                stitch_info = edge.get("stitch_info")
                stitch_type = (
                    stitch_info.get("stitch_type") if stitch_info is not None else None
                )
                if stitch_type in target_stitch_types:
                    start: np.ndarray = edge.get("start")
                    end: np.ndarray = edge.get("end")
                    start_uv: np.ndarray = edge.get("start_uv")
                    end_uv: np.ndarray = edge.get("end_uv")
                    start_point: np.ndarray = (
                        start
                        if (
                            start_uv is None
                            or NeuralTailorConverter.is_zero_vector(start_uv)
                        )
                        else start_uv
                    )
                    end_point: np.ndarray = (
                        end
                        if (
                            end_uv is None
                            or NeuralTailorConverter.is_zero_vector(end_uv)
                        )
                        else end_uv
                    )

                    # TODO: modify the following code to handle the case where the following assumption is not satisfied
                    # I assume that two consecutive edges are not both side_by_side; otherwise, the uv_vertices will be overwritten
                    if stitch_type == Seam.SIDE_BY_SIDE:
                        if (
                            start_point[0] == end_point[0]
                            or start_point[1] == end_point[1]
                        ):
                            uv_vertices[i % len(edge_info)][0] = start_point[0]
                            uv_vertices[i % len(edge_info)][1] = start_point[1]
                            uv_vertices[i % len(edge_info)][2] = start_point[2]
                            uv_vertices[i % len(edge_info)][3] = start_point[3]
                            uv_vertices[(i + 1) % len(edge_info)][0] = end_point[0]
                            uv_vertices[(i + 1) % len(edge_info)][1] = end_point[1]
                            uv_vertices[(i + 1) % len(edge_info)][2] = end_point[2]
                            uv_vertices[(i + 1) % len(edge_info)][3] = end_point[3]
                        else:
                            # decide the axis to be parallel to
                            x_diff = np.abs(end_point[0] - start_point[0])
                            y_diff = np.abs(end_point[1] - start_point[1])
                            if x_diff < y_diff:
                                middle_x = (start_point[0] + end_point[0]) / 2
                                uv_vertices[i % len(edge_info)][0] = middle_x
                                uv_vertices[i % len(edge_info)][1] = start_point[1]
                                uv_vertices[i % len(edge_info)][2] = start_point[2]
                                uv_vertices[i % len(edge_info)][3] = start_point[3]
                                uv_vertices[(i + 1) % len(edge_info)][0] = middle_x
                                uv_vertices[(i + 1) % len(edge_info)][1] = end_point[1]
                                uv_vertices[(i + 1) % len(edge_info)][2] = end_point[2]
                                uv_vertices[(i + 1) % len(edge_info)][3] = end_point[3]
                            else:
                                middle_y = (start_point[1] + end_point[1]) / 2
                                uv_vertices[i % len(edge_info)][0] = start_point[0]
                                uv_vertices[i % len(edge_info)][1] = middle_y
                                uv_vertices[i % len(edge_info)][2] = start_point[2]
                                uv_vertices[i % len(edge_info)][3] = start_point[3]
                                uv_vertices[(i + 1) % len(edge_info)][0] = end_point[0]
                                uv_vertices[(i + 1) % len(edge_info)][1] = middle_y
                                uv_vertices[(i + 1) % len(edge_info)][2] = end_point[2]
                                uv_vertices[(i + 1) % len(edge_info)][3] = end_point[3]
                    else:
                        if (
                            start_point[0] == end_point[0]
                            or start_point[1] == end_point[1]
                        ):
                            uv_vertices[i % len(edge_info)][0] = start_point[0]
                            uv_vertices[i % len(edge_info)][1] = start_point[1]
                            uv_vertices[i % len(edge_info)][2] = start_point[2]
                            uv_vertices[i % len(edge_info)][3] = start_point[3]
                            uv_vertices[(i + 1) % len(edge_info)][0] = end_point[0]
                            uv_vertices[(i + 1) % len(edge_info)][1] = end_point[1]
                            uv_vertices[(i + 1) % len(edge_info)][2] = end_point[2]
                            uv_vertices[(i + 1) % len(edge_info)][3] = end_point[3]
                        else:
                            prev_edge = edge_info[(i - 1) % len(edge_info)]
                            next_edge = edge_info[(i + 1) % len(edge_info)]
                            start_point_fixed = False
                            end_point_fixed = False
                            if (
                                prev_edge.get("stitch_info") is not None
                                and prev_edge.get("stitch_info").get("stitch_type")
                                == Seam.SIDE_BY_SIDE
                            ):
                                start_point_fixed = True
                            if (
                                next_edge.get("stitch_info") is not None
                                and next_edge.get("stitch_info").get("stitch_type")
                                == Seam.SIDE_BY_SIDE
                            ):
                                end_point_fixed = True
                            if start_point_fixed and end_point_fixed:
                                warnings.warn(
                                    "both start and end points are fixed", UserWarning
                                )
                                continue

                            # decide the axis to be parallel to
                            x_diff = np.abs(end_point[0] - start_point[0])
                            y_diff = np.abs(end_point[1] - start_point[1])
                            if x_diff < y_diff:
                                if start_point_fixed:
                                    uv_vertices[(i + 1) % len(edge_info)][0] = (
                                        start_point[0]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][1] = (
                                        end_point[1]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][2] = (
                                        end_point[2]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][3] = (
                                        end_point[3]
                                    )
                                elif end_point_fixed:
                                    uv_vertices[i % len(edge_info)][0] = end_point[0]
                                    uv_vertices[i % len(edge_info)][1] = start_point[1]
                                    uv_vertices[i % len(edge_info)][2] = start_point[2]
                                    uv_vertices[i % len(edge_info)][3] = start_point[3]
                                else:
                                    middle_x = (start_point[0] + end_point[0]) / 2
                                    uv_vertices[i % len(edge_info)][0] = middle_x
                                    uv_vertices[i % len(edge_info)][1] = start_point[1]
                                    uv_vertices[i % len(edge_info)][2] = start_point[2]
                                    uv_vertices[i % len(edge_info)][3] = start_point[3]
                                    uv_vertices[(i + 1) % len(edge_info)][0] = middle_x
                                    uv_vertices[(i + 1) % len(edge_info)][1] = (
                                        end_point[1]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][2] = (
                                        end_point[2]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][3] = (
                                        end_point[3]
                                    )
                            else:
                                if start_point_fixed:
                                    uv_vertices[(i + 1) % len(edge_info)][0] = (
                                        end_point[0]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][1] = (
                                        start_point[1]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][2] = (
                                        end_point[2]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][3] = (
                                        end_point[3]
                                    )
                                elif end_point_fixed:
                                    uv_vertices[i % len(edge_info)][0] = start_point[0]
                                    uv_vertices[i % len(edge_info)][1] = end_point[1]
                                    uv_vertices[i % len(edge_info)][2] = start_point[2]
                                    uv_vertices[i % len(edge_info)][3] = start_point[3]
                                else:
                                    middle_y = (start_point[1] + end_point[1]) / 2
                                    uv_vertices[i % len(edge_info)][0] = start_point[0]
                                    uv_vertices[i % len(edge_info)][1] = middle_y
                                    uv_vertices[i % len(edge_info)][2] = start_point[2]
                                    uv_vertices[i % len(edge_info)][3] = start_point[3]
                                    uv_vertices[(i + 1) % len(edge_info)][0] = (
                                        end_point[0]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][1] = middle_y
                                    uv_vertices[(i + 1) % len(edge_info)][2] = (
                                        end_point[2]
                                    )
                                    uv_vertices[(i + 1) % len(edge_info)][3] = (
                                        end_point[3]
                                    )

    @staticmethod
    def initialize_uv(
        edge_info: List[Dict[str, Union[np.ndarray, Dict]]],
        uv_vertices: Optional[np.ndarray] = None,
    ):
        if uv_vertices is None:
            uv_vertices = np.zeros((len(edge_info), 2))
        for i in range(len(edge_info)):
            edge = edge_info[i]
            edge["start_uv"] = uv_vertices[i]
            edge["end_uv"] = uv_vertices[(i + 1) % len(edge_info)]

    @staticmethod
    def align_two_panel_to_edge_infos_with_centroid(
        panel_to_edge_info_1: PanelToEdgeInfoType,
        panel_to_edge_info_2: PanelToEdgeInfoType,
    ):
        for panel_name in panel_to_edge_info_1.keys():
            edge_info_1 = panel_to_edge_info_1[panel_name]
            edge_info_2 = panel_to_edge_info_2.get(panel_name)
            if edge_info_2 is None:
                continue
            NeuralTailorConverter.align_two_edge_infos_with_centroid(
                edge_info_1, edge_info_2
            )

    @staticmethod
    def align_two_edge_infos_with_centroid(
        edge_info_1: EdgeInfoType, edge_info_2: EdgeInfoType
    ):
        edge_info_1_centroid = NeuralTailorConverter.calculate_edge_info_centroid(
            edge_info_1
        )
        edge_info_2_centroid = NeuralTailorConverter.calculate_edge_info_centroid(
            edge_info_2
        )
        transformation = edge_info_1_centroid - edge_info_2_centroid
        transformation_matrix = np.array(
            [
                [1, 0, 0, transformation[0]],
                [0, 1, 0, transformation[1]],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        NeuralTailorConverter.apply_affine_matrix_edge_info(
            transformation_matrix, edge_info_2
        )

    @staticmethod
    def find_edge_by_start_vertex_index(
        edge_info: EdgeInfoType, start_index: int
    ) -> Optional[Dict[str, Union[np.ndarray, Dict]]]:
        for edge in edge_info:
            if edge["start_index"] == start_index:
                return edge
        return None

    @staticmethod
    def find_edge_by_end_vertex_index(
        edge_info: EdgeInfoType, end_index: int
    ) -> Optional[Dict[str, Union[np.ndarray, Dict]]]:
        for edge in edge_info:
            if edge["end_index"] == end_index:
                return edge
        return None

    @staticmethod
    def get_corner_vertex_indices(edge_info: EdgeInfoType) -> List[int]:
        """
        Return points which have different type seams
        """
        corner_vertex_indices = []
        for i in range(len(edge_info)):
            edge_start = NeuralTailorConverter.find_edge_by_start_vertex_index(
                edge_info, i
            )
            edge_end = NeuralTailorConverter.find_edge_by_end_vertex_index(edge_info, i)
            assert edge_start is not None and edge_end is not None
            if (
                edge_start["stitch_info"]["stitch_type"]
                != edge_end["stitch_info"]["stitch_type"]
            ):
                corner_vertex_indices.append(i)
        return corner_vertex_indices

    # @staticmethod
    def get_corner_vertex_indices(edge_info: EdgeInfoType) -> List[int]:
        """
        Return points which have different type seams
        """
        last_edge_info = edge_info[-1]
        last_edge_stitch_type = last_edge_info["stitch_info"]["stitch_type"]
        prev_seam_type: int = last_edge_stitch_type
        current_seam_type: Optional[int] = None
        corner_vertex_indices: List[int] = []
        for _edge_info in edge_info:
            current_seam_type = _edge_info["stitch_info"]["stitch_type"]
            if prev_seam_type != current_seam_type:
                corner_vertex_indices.append(_edge_info["start_index"])
            prev_seam_type = current_seam_type
        return corner_vertex_indices

    @staticmethod
    def is_zero_vector(vector: np.ndarray) -> bool:
        return np.all(vector == 0)

    @staticmethod
    def parallel_to_y_axis_edge_info_uv(
        edge_info: Dict[str, Union[np.ndarray, Dict]],
    ) -> bool:
        start_uv: np.ndarray = edge_info.get("start_uv")
        end_uv: np.ndarray = edge_info.get("end_uv")
        if start_uv is None or end_uv is None:
            return False
        return start_uv[0] == end_uv[0]

    @staticmethod
    def parallel_to_x_axis_edge_info_uv(
        edge_info: Dict[str, Union[np.ndarray, Dict]],
    ) -> bool:
        start_uv: np.ndarray = edge_info.get("start_uv")
        end_uv: np.ndarray = edge_info.get("end_uv")
        if start_uv is None or end_uv is None:
            return False
        return start_uv[1] == end_uv[1]

    @staticmethod
    def add_stitches_to_edge_info(
        panel_to_edge_info: PanelToEdgeInfoType,
        stitches: List[List[Dict[str, Union[int, str]]]],
    ):
        for stitch in stitches:
            # assert len(stitch) == 2, "stitch is not a pair"
            if len(stitch) != 2:
                warnings.warn("stitch is not a pair", UserWarning)
                continue
            edge_1 = stitch[0]
            edge_2 = stitch[1]
            edge_1_panel = panel_to_edge_info.get(edge_1["panel"])
            edge_2_panel = panel_to_edge_info.get(edge_2["panel"])
            is_front_edge_1_panel = NeuralTailorConverter.judge_front(edge_1["panel"])
            is_front_edge_2_panel = NeuralTailorConverter.judge_front(edge_2["panel"])
            try:
                edge_1_points: Optional[Dict[str, np.ndarray]] = (
                    edge_1_panel[edge_1["edge"]] if edge_1_panel is not None else None
                )
                edge_2_points: Optional[Dict[str, np.ndarray]] = (
                    edge_2_panel[edge_2["edge"]] if edge_2_panel is not None else None
                )
            except IndexError:
                warnings.warn("edge index is out of range", UserWarning)
                continue
            if edge_1_points is not None:
                stitch_type = (
                    Seam.FRONT_TO_BACK
                    if is_front_edge_1_panel != is_front_edge_2_panel
                    else Seam.SIDE_BY_SIDE
                )
                stitch_info = {
                    "stitch_type": stitch_type,
                    "target_edge_panel": edge_2["panel"],
                    "target_edge_index": edge_2["edge"],
                }
                edge_1_points["stitch_info"] = stitch_info
            if edge_2_points is not None:
                stitch_type = (
                    Seam.FRONT_TO_BACK
                    if is_front_edge_1_panel != is_front_edge_2_panel
                    else Seam.SIDE_BY_SIDE
                )
                stitch_info = {
                    "stitch_type": stitch_type,
                    "target_edge_panel": edge_1["panel"],
                    "target_edge_index": edge_1["edge"],
                }
                edge_2_points["stitch_info"] = stitch_info

    @staticmethod
    def apply_front_to_back_stitch_alignment(
        panel_to_edge_info: PanelToEdgeInfoType,
    ):
        for panel_name in panel_to_edge_info:
            edge_info = panel_to_edge_info[panel_name]
            for i, edge in enumerate(edge_info):
                prev_edge = edge_info[(i - 1) % len(edge_info)]
                next_edge = edge_info[(i + 1) % len(edge_info)]
                stitch_info = edge.get("stitch_info")
                if stitch_info.get("stitch_type") == Seam.BOUNDARY:
                    continue
                elif stitch_info.get("stitch_type") == Seam.SIDE_BY_SIDE:
                    continue
                elif stitch_info.get("stitch_type") == Seam.FRONT_TO_BACK:
                    # align back side with front side
                    if NeuralTailorConverter.judge_front(panel_name):
                        continue

                    target_panel_name = stitch_info["target_edge_panel"]
                    target_edge_index = stitch_info["target_edge_index"]
                    assert target_panel_name != panel_name, (
                        f"target_panel {target_panel_name} and base panel {panel_name} should not be the same for front_to_back stitch"
                    )
                    target_edge = panel_to_edge_info[target_panel_name][
                        target_edge_index
                    ]
                    start = edge.get("start")
                    end = edge.get("end")
                    target_start = target_edge.get("start")
                    target_end = target_edge.get("end")
                    if np.linalg.norm(start - target_start) < np.linalg.norm(
                        start - target_end
                    ):
                        if np.any(start != target_start) or np.any(end != target_end):
                            original_z = edge["start"][2]
                            target_start_copy = np.copy(target_start)
                            target_start_copy[2] = original_z
                            target_end_copy = np.copy(target_end)
                            target_end_copy[2] = original_z
                            edge["start"] = target_start_copy
                            edge["end"] = target_end_copy
                            prev_edge["end"] = target_start_copy
                            next_edge["start"] = target_end_copy
                    else:
                        if np.any(start != target_end) or np.any(end != target_start):
                            original_z = edge["start"][2]
                            target_start_copy = np.copy(target_start)
                            target_start_copy[2] = original_z
                            target_end_copy = np.copy(target_end)
                            target_end_copy[2] = original_z
                            edge["start"] = target_end_copy
                            edge["end"] = target_start_copy
                            prev_edge["end"] = target_end_copy
                            next_edge["start"] = target_start_copy

    @staticmethod
    def apply_waistband_alignment(panel_to_edge_info: PanelToEdgeInfoType):
        for panel_name in panel_to_edge_info:
            edge_info = panel_to_edge_info[panel_name]
            for i, edge in enumerate(edge_info):
                stitch_info = edge.get("stitch_info")
                if stitch_info.get("stitch_type") == Seam.BOUNDARY:
                    continue
                elif stitch_info.get("stitch_type") == Seam.FRONT_TO_BACK:
                    continue
                elif stitch_info.get("stitch_type") == Seam.SIDE_BY_SIDE:
                    target_panel_name = stitch_info["target_edge_panel"]
                    target_edge_index = stitch_info["target_edge_index"]
                    target_edge_info = panel_to_edge_info[target_panel_name][
                        target_edge_index
                    ]
                    start = edge.get("start")
                    end = edge.get("end")
                    target_start = target_edge_info.get("start")
                    target_end = target_edge_info.get("end")
                    if not target_panel_name.startswith("wb"):
                        continue
                    if panel_name == "left_ftorso":
                        target_start[:] = copy(end)
                    if panel_name == "right_ftorso":
                        target_end[:] = copy(start)
                    if panel_name == "left_btorso":
                        target_end[:] = copy(start)
                    if panel_name == "right_btorso":
                        target_start[:] = copy(end)
                    if panel_name == "skirt_front":
                        target_start[:] = copy(end)
                        target_end[:] = copy(start)
                    if panel_name == "skirt_back":
                        target_start[:] = copy(end)
                        target_end[:] = copy(start)

    @staticmethod
    def apply_stitch_alignment(
        panel_to_edge_info: PanelToEdgeInfoType,
        force_alignment: bool = False,
        align_with_uv: bool = False,
        y_axis_alignment: bool = False,
        dart_alignment: bool = False,
        skirt_alignment: bool = False,
        non_target_panel_names: Optional[List[str]] = None,
    ):
        panel_to_is_marked: Dict[str, bool] = defaultdict(bool)
        is_marked_flag = False
        for panel_name in panel_to_edge_info.keys():
            # TODO: メインとするpanelが名前にtopを持つことが常に成り立つかどうかを確かめる
            # if NeuralTailorConverter.judge_top(panel_name):
            if skirt_alignment:
                if panel_name in ["skirt_front", "skirt_back"]:
                    panel_to_is_marked[panel_name] = True
                    is_marked_flag = True
            elif "top" in panel_name:
                panel_to_is_marked[panel_name] = True
                is_marked_flag = True
        if not is_marked_flag:
            # the case where the main panel is not found.
            # We assume that garmetncodedata garments do not have a main panel
            # In this case, we mark the waist panel as the main panel
            for panel_name in panel_to_edge_info.keys():
                if "wb" in panel_name:
                    panel_to_is_marked[panel_name] = True
                    is_marked_flag = True

        if non_target_panel_names is not None:
            for panel_name in non_target_panel_names:
                panel_to_is_marked[panel_name] = True

        for panel_name in panel_to_edge_info:
            if panel_to_is_marked[panel_name]:
                continue
            edge_info = panel_to_edge_info[panel_name]
            for i, edge in enumerate(edge_info):
                stitch_info = edge.get("stitch_info")
                if stitch_info.get("stitch_type") == Seam.BOUNDARY:
                    continue
                # TODO: we have to consider the case where the stitch_type is FRONT_TO_BACK
                elif stitch_info.get("stitch_type") == Seam.FRONT_TO_BACK:
                    continue
                elif stitch_info.get("stitch_type") == Seam.SIDE_BY_SIDE:
                    target_panel_name = stitch_info["target_edge_panel"]
                    target_edge_index = stitch_info["target_edge_index"]
                    target_edge_info = panel_to_edge_info[target_panel_name][
                        target_edge_index
                    ]
                    if panel_to_is_marked[target_panel_name]:
                        if align_with_uv:
                            start = edge.get("start_uv")
                            end = edge.get("end_uv")
                            target_start = target_edge_info.get("start_uv")
                            target_end = target_edge_info.get("end_uv")
                        else:
                            start = edge.get("start")
                            end = edge.get("end")
                            target_start = target_edge_info.get("start")
                            target_end = target_edge_info.get("end")
                        translation_1 = target_end - start
                        translation_2 = target_start - end
                        # assert np.all(translation_1 == translation_2), f"translation is not the same, {translation_1} != {translation_2}"
                        if not np.all(translation_1 == translation_2):
                            # warnings.warn(
                            #     f"translation is not the same, {translation_1} != {translation_2}, {np.linalg.norm(translation_1 - translation_2)}",
                            #     UserWarning,
                            # )
                            pass
                        translation = (
                            translation_1 if start[1] < end[1] else translation_2
                        )

                        # translate all the vertices of the panel
                        if y_axis_alignment:
                            translation_matrix: np.ndarray = np.array(
                                [
                                    [1, 0, 0, 0],
                                    [0, 1, 0, translation_1[1]],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                ]
                            )
                        else:
                            translation_matrix: np.ndarray = np.array(
                                [
                                    [1, 0, 0, translation[0]],
                                    [0, 1, 0, translation[1]],
                                    [0, 0, 1, translation[2]],
                                    [0, 0, 0, 1],
                                ]
                            )
                        NeuralTailorConverter.apply_affine_matrix_edge_info(
                            translation_matrix, edge_info
                        )
                        # TODO: Make sure that the following assumption is correct
                        # We assume that panels have one side_by_side stitch at most except for the main panel
                        break
            panel_to_is_marked[panel_name] = True

        if dart_alignment:
            for panel_name in panel_to_edge_info:
                edge_info = panel_to_edge_info[panel_name]
                for i, edge in enumerate(edge_info):
                    prev_edge = edge_info[(i - 1) % len(edge_info)]
                    next_edge = edge_info[(i + 1) % len(edge_info)]
                    stitch_info = edge.get("stitch_info")
                    if stitch_info.get("stitch_type") == Seam.BOUNDARY:
                        continue
                    elif stitch_info.get("stitch_type") == Seam.FRONT_TO_BACK:
                        continue
                    elif stitch_info.get("stitch_type") == Seam.SIDE_BY_SIDE:
                        target_panel_name = stitch_info["target_edge_panel"]
                        target_edge_index = stitch_info["target_edge_index"]
                        if target_panel_name == panel_name:
                            target_edge = edge_info[target_edge_index]
                            start = edge.get("start")
                            end = edge.get("end")
                            target_start = target_edge.get("start")
                            target_end = target_edge.get("end")
                            if np.all(end == target_start):
                                # this is a dart
                                # assert (i + 1 % len(edge_info)) == target_edge_index
                                if np.any(start != target_end):
                                    if np.linalg.norm(start) > np.linalg.norm(
                                        target_end
                                    ):
                                        edge["start"] = target_end
                                        prev_edge["end"] = target_end
                                    else:
                                        next_edge_of_target_edge = edge_info[
                                            (target_edge_index + 1) % len(edge_info)
                                        ]
                                        target_edge["end"] = start
                                        next_edge_of_target_edge["start"] = start

        if force_alignment:
            panel_to_is_marked_force_alignment: Dict[str, bool] = defaultdict(bool)
            for panel_name in panel_to_edge_info.keys():
                # TODO: メインとするpanelが名前にtopを持つことが常に成り立つかどうかを確かめる
                if "top" in panel_name:
                    panel_to_is_marked_force_alignment[panel_name] = True
            if non_target_panel_names is not None:
                for panel_name in non_target_panel_names:
                    panel_to_is_marked_force_alignment[panel_name] = True
            for panel_name in panel_to_edge_info:
                if panel_to_is_marked_force_alignment[panel_name]:
                    continue
                edge_info = panel_to_edge_info[panel_name]
                for i, edge in enumerate(edge_info):
                    prev_edge = edge_info[(i - 1) % len(edge_info)]
                    next_edge = edge_info[(i + 1) % len(edge_info)]
                    stitch_info = edge.get("stitch_info")
                    if stitch_info.get("stitch_type") == Seam.BOUNDARY:
                        continue
                    # TODO: we have to consider the case where the stitch_type is FRONT_TO_BACK
                    elif stitch_info.get("stitch_type") == Seam.FRONT_TO_BACK:
                        continue
                    elif stitch_info.get("stitch_type") == Seam.SIDE_BY_SIDE:
                        target_panel_name = stitch_info["target_edge_panel"]
                        target_edge_index = stitch_info["target_edge_index"]
                        target_edge_info = panel_to_edge_info[target_panel_name][
                            target_edge_index
                        ]
                        if panel_to_is_marked_force_alignment[target_panel_name]:
                            if align_with_uv:
                                start = edge.get("start_uv")
                                end = edge.get("end_uv")
                                target_start = target_edge_info.get("start_uv")
                                target_end = target_edge_info.get("end_uv")
                                if np.any(start != target_end):
                                    edge["start_uv"] = target_end
                                if np.any(end != target_start):
                                    edge["end_uv"] = target_start

                            else:
                                start = edge.get("start")
                                end = edge.get("end")
                                target_start = target_edge_info.get("start")
                                target_end = target_edge_info.get("end")
                                if np.any(start != target_end):
                                    # not edge["start"] = target_end because edge["start"] does not change the reference object
                                    # edge["start"][:] = copy(target_end)
                                    edge["start"] = target_end
                                    prev_edge["end"] = target_end

                                if np.any(end != target_start):
                                    # edge["end"][:] = copy(target_start)
                                    edge["end"] = target_start
                                    next_edge["start"] = target_start
                            break
                            # TODO: Make sure that the following assumption is correct
                            # We assume that panels have one side_by_side stitch at most except for the main panel
                panel_to_is_marked_force_alignment[panel_name] = True

        if skirt_alignment:
            panel_to_is_marked_skirt_alignment: Dict[str, bool] = defaultdict(bool)
            for panel_name in panel_to_edge_info.keys():
                if panel_name not in ["skirt_front", "skirt_back"]:
                    panel_to_is_marked_skirt_alignment[panel_name] = True
            if non_target_panel_names is not None:
                for panel_name in non_target_panel_names:
                    panel_to_is_marked_skirt_alignment[panel_name] = True
            for panel_name in panel_to_edge_info:
                if panel_to_is_marked_skirt_alignment[panel_name]:
                    continue
                edge_info = panel_to_edge_info[panel_name]
                for i, edge in enumerate(edge_info):
                    prev_edge = edge_info[(i - 1) % len(edge_info)]
                    next_edge = edge_info[(i + 1) % len(edge_info)]
                    stitch_info = edge.get("stitch_info")
                    if stitch_info.get("stitch_type") == Seam.BOUNDARY:
                        continue
                    # TODO: we have to consider the case where the stitch_type is FRONT_TO_BACK
                    elif stitch_info.get("stitch_type") == Seam.FRONT_TO_BACK:
                        continue
                    elif stitch_info.get("stitch_type") == Seam.SIDE_BY_SIDE:
                        target_panel_name = stitch_info["target_edge_panel"]
                        target_edge_index = stitch_info["target_edge_index"]
                        target_edge_info = panel_to_edge_info[target_panel_name][
                            target_edge_index
                        ]
                        if panel_to_is_marked_skirt_alignment[target_panel_name]:
                            start = edge.get("start")
                            end = edge.get("end")
                            target_start = target_edge_info.get("start")
                            target_end = target_edge_info.get("end")
                            if np.any(start != target_end):
                                # not edge["start"] = target_end because edge["start"] does not change the reference object
                                # edge["start"][:] = copy(target_end)
                                edge["start"] = target_end
                                prev_edge["end"] = target_end

                            if np.any(end != target_start):
                                # edge["end"][:] = copy(target_start)
                                edge["end"] = target_start
                                next_edge["start"] = target_start

    @staticmethod
    def sort_edge_list(
        edge_list: List[Dict[str, Union[np.ndarray, Dict]]],
    ) -> List[Dict[str, Union[np.ndarray, Dict]]]:
        """
        Sort the edge list based on the position of the start point of the edge.
        The implementation is based on the assumption that the start and end points of the connected edges are the same.
        Make sure that the panel is force aligned before sorting if you merge the edges.
        """
        edge_queue = queue.Queue()
        for edge in edge_list:
            edge_queue.put(edge)
        sorted_edge_list = []
        count = 0
        while not edge_queue.empty():
            count += 1
            if count > 1000:
                raise Exception("The edge queue is not empty.")
            edge = edge_queue.get()
            if len(sorted_edge_list) == 0:
                sorted_edge_list.append(edge)
                continue
            prev_edge = sorted_edge_list[-1]
            if np.allclose(edge["start"], prev_edge["end"]):
                sorted_edge_list.append(edge)
            else:
                edge_queue.put(edge)
        return sorted_edge_list

    @staticmethod
    def merge_edge_info(
        edge_info: PanelToEdgeInfoType,
        panel_names: List[str],
    ):
        """
        Merge multiple closed edges into one edge.
        """
        if len(panel_names) == 0 or len(panel_names) == 1:
            warnings.warn(
                "The number of panels to merge should not be 0 or 1.", UserWarning
            )
            return
        merged_panel_names = set()
        merged_edge_list = []
        all_panel_names = list(edge_info.keys())
        for panel_name in all_panel_names:
            edge_list = edge_info[panel_name]
            if panel_name not in panel_names:
                continue
            edge_info.pop(panel_name)
            for edge in edge_list:
                stitch_info = edge.get("stitch_info")
                assert stitch_info is not None, (
                    "Stitch info should be provided for merging"
                )
                stitch_type = stitch_info.get("stitch_type")
                target_edge_panel = stitch_info.get("target_edge_panel")
                if (
                    stitch_type == Seam.SIDE_BY_SIDE
                    and target_edge_panel in panel_names
                ):
                    continue
                else:
                    merged_edge_list.append(edge)
                    merged_panel_names.add(panel_name)
        if len(merged_panel_names) > 0:
            merged_edge_name = "_".join(sorted(list(merged_panel_names)))
            sorted_merged_edge_list = NeuralTailorConverter.sort_edge_list(
                merged_edge_list
            )
            edge_info[merged_edge_name] = sorted_merged_edge_list

    @staticmethod
    def merge_edge_info_v2(
        panel_to_edge_info: PanelToEdgeInfoType,
        merged_panels_pairs: List[List[str]],
        only_horizontally_flat_edge: bool = False,
    ):
        """
        Merge multiple closed edges into one edge.
        """
        edge_index_map: Dict[
            str, Dict[int, Dict[str, Union[int, str]]]
        ] = {}  # panel_name -> before_edge_index -> {panel_name: str, edge_index: int}
        for merged_panel_names in merged_panels_pairs:
            merged_edge_list = []
            assert len(merged_panel_names) == 2, (
                "The number of panels to merge should be 2."
            )
            # TODO: Address more than two panels
            base_panel_edge: Optional[Dict[str, Union[np.ndarray, Dict]]] = None
            connected_panel_edge: Optional[Dict[str, Union[np.ndarray, Dict]]] = None
            base_panel_vertex_num: Optional[int] = None
            connected_panel_vertex_map: Optional[Dict[int, int]] = None
            for i, panel_name in enumerate(merged_panel_names):
                other_panel_name = merged_panel_names[(i + 1) % 2]
                if panel_name not in panel_to_edge_info:
                    continue
                edge_list = panel_to_edge_info[panel_name]
                panel_to_edge_info.pop(panel_name)
                if i == 0:
                    base_panel_vertex_num = len(edge_list)
                for edge in edge_list:
                    stitch_info = edge.get("stitch_info")
                    assert stitch_info is not None, (
                        "Stitch info should be provided for merging"
                    )
                    stitch_type = stitch_info.get("stitch_type")
                    target_edge_panel = stitch_info.get("target_edge_panel")
                    if (
                        stitch_type == Seam.SIDE_BY_SIDE
                        and target_edge_panel == other_panel_name
                    ):
                        if not only_horizontally_flat_edge or (
                            only_horizontally_flat_edge
                            and np.abs(edge["start"][1] - edge["end"][1]) < 1
                        ):
                            if i == 0:
                                base_panel_edge = edge
                            else:
                                connected_panel_edge = edge
                            continue
                    merged_edge_list.append(edge)

            if base_panel_edge is None or connected_panel_edge is None:
                continue
            # mapping
            base_panel_start_index = base_panel_edge["start_index"]
            base_panel_end_index = base_panel_edge["end_index"]
            connected_panel_start_index = connected_panel_edge["start_index"]
            connected_panel_end_index = connected_panel_edge["end_index"]
            connected_panel_vertex_map = {
                connected_panel_start_index: base_panel_end_index,
                connected_panel_end_index: base_panel_start_index,
            }
            connected_panel_indices = set()
            for i, edge in enumerate(merged_edge_list[base_panel_vertex_num - 1 :]):
                edge_start_index = edge["start_index"]
                edge_end_index = edge["end_index"]
                if edge_start_index not in connected_panel_vertex_map:
                    connected_panel_indices.add(edge_start_index)
                if edge_end_index not in connected_panel_vertex_map:
                    connected_panel_indices.add(edge_end_index)
            connected_panel_indices_sorted = sorted(list(connected_panel_indices))
            for i, connected_panel_index in enumerate(connected_panel_indices_sorted):
                connected_panel_vertex_map[connected_panel_index] = (
                    base_panel_vertex_num + i
                )

            for i, edge in enumerate(merged_edge_list[base_panel_vertex_num - 1 :]):
                start_index = edge["start_index"]
                end_index = edge["end_index"]
                edge["start_index"] = connected_panel_vertex_map[start_index]
                edge["end_index"] = connected_panel_vertex_map[end_index]

            merged_edge_name = "_".join(sorted(list(merged_panel_names)))
            sorted_merged_edge_list = NeuralTailorConverter.sort_edge_list(
                merged_edge_list
            )
            panel_to_edge_info[merged_edge_name] = sorted_merged_edge_list

            # mapping edge index
            for i, edge in enumerate(sorted_merged_edge_list):
                old_panel_name = edge["panel_name"]
                old_edge_index = edge["edge_index"]
                new_edge_index = i
                if old_panel_name not in edge_index_map:
                    edge_index_map[old_panel_name] = {}
                edge_index_map[old_panel_name][old_edge_index] = {
                    "panel_name": merged_edge_name,
                    "edge_index": new_edge_index,
                }

        # apply edge index mapping
        for panel_name in panel_to_edge_info:
            for edge_info in panel_to_edge_info[panel_name]:
                current_panel_name = edge_info["panel_name"]
                current_edge_index = edge_info["edge_index"]
                if (
                    current_panel_name in edge_index_map
                    and current_edge_index in edge_index_map[current_panel_name]
                ):
                    edge_info["panel_name"] = edge_index_map[current_panel_name][
                        current_edge_index
                    ]["panel_name"]
                    edge_info["edge_index"] = edge_index_map[current_panel_name][
                        current_edge_index
                    ]["edge_index"]
                stitch_info = edge_info.get("stitch_info")
                if stitch_info is not None:
                    target_edge_panel = stitch_info.get("target_edge_panel")
                    target_edge_index = stitch_info.get("target_edge_index")
                    if (
                        target_edge_panel
                        and target_edge_index
                        and target_edge_panel in edge_index_map
                        and target_edge_index in edge_index_map[target_edge_panel]
                    ):
                        stitch_info["target_edge_panel"] = edge_index_map[
                            target_edge_panel
                        ][target_edge_index]["panel_name"]
                        stitch_info["target_edge_index"] = edge_index_map[
                            target_edge_panel
                        ][target_edge_index]["edge_index"]

    @staticmethod
    def separate_top_back_panel(panel_to_edge_info: PanelToEdgeInfoType):
        top_back_panel_name = "top_back"
        # top_back_panel_name = "up_back"
        top_back_panel = panel_to_edge_info[top_back_panel_name]
        # assume the top_front has V-neck
        x_center = 0.0
        for edge_info in top_back_panel:
            x_center += edge_info["start"][0]
        x_center /= len(top_back_panel)

        # assume U-neck
        top_back_u_neck_edge_left_index: Optional[int] = None
        top_back_u_neck_edge_right_index: Optional[int] = None
        top_back_u_neck_edge_left_vertex: Optional[np.ndarray] = None
        top_back_u_neck_edge_right_vertex: Optional[np.ndarray] = None
        top_back_u_neck_control_point: Optional[np.ndarray] = None
        top_back_u_neck_curvature: Optional[np.ndarray] = None
        top_back_u_neck_left_curvature: Optional[np.ndarray] = None
        top_back_u_neck_right_curvature: Optional[np.ndarray] = None
        for edge_info in top_back_panel:
            if edge_info["start"][0] < x_center and edge_info["end"][0] > x_center:
                left_index = edge_info["start_index"]
                right_index = edge_info["end_index"]
                top_back_u_neck_edge_left_index = left_index
                top_back_u_neck_edge_right_index = right_index
                top_back_u_neck_edge_left_vertex = edge_info["start"]
                top_back_u_neck_edge_right_vertex = edge_info["end"]
                top_back_u_neck_control_point = edge_info.get("control_point")
                assert top_back_u_neck_control_point is not None
                top_back_u_neck_curvature = edge_info.get("curvature")

        # sample t = 0.5 of the control point based on top_back_u_neck_control_point, top_back_u_neck_edge_left_vertex, top_back_u_neck_edge_right_vertex
        # separate the quadratic bezier curve into two quadratic bezier curves
        t = 0.5
        new_vertex_1 = (
            (t**2) * top_back_u_neck_edge_left_vertex
            + 2 * t * (1 - t) * top_back_u_neck_control_point
            + ((1 - t) ** 2) * top_back_u_neck_edge_right_vertex
        )
        control_point_left = (
            t * top_back_u_neck_edge_left_vertex
            + (1 - t) * top_back_u_neck_control_point
        )
        control_point_right = (
            t * top_back_u_neck_control_point
            + (1 - t) * top_back_u_neck_edge_right_vertex
        )
        top_back_u_neck_left_curvature = curvature_3d_quadratic_bezier(
            top_back_u_neck_edge_left_vertex[:3],
            control_point_left[:3],
            new_vertex_1[:3],
            0.5,
        )
        top_back_u_neck_right_curvature = curvature_3d_quadratic_bezier(
            top_back_u_neck_edge_right_vertex[:3],
            control_point_right[:3],
            new_vertex_1[:3],
            0.5,
        )

        top_back_bottom_edge_left_index: Optional[int] = None
        top_back_bottom_edge_right_index: Optional[int] = None
        top_back_bottom_edge_left_vertex: Optional[np.ndarray] = None
        top_back_bottom_edge_right_vertex: Optional[np.ndarray] = None
        for edge_info in top_back_panel:
            if edge_info["start"][0] > x_center and edge_info["end"][0] < x_center:
                left_index = edge_info["end_index"]
                right_index = edge_info["start_index"]
                top_back_bottom_edge_left_index = left_index
                top_back_bottom_edge_right_index = right_index
                top_back_bottom_edge_left_vertex = edge_info["end"]
                top_back_bottom_edge_right_vertex = edge_info["start"]

        new_vertex_2 = (
            top_back_bottom_edge_left_vertex + top_back_bottom_edge_right_vertex
        ) / 2

        new_vertex_1_index = len(top_back_panel)
        new_vertex_2_index = new_vertex_1_index + 1
        new_edge_index_left_1 = len(top_back_panel)
        new_edge_index_left_2 = new_edge_index_left_1 + 1
        new_edge_index_right_1 = new_edge_index_left_2 + 1
        new_edge_index_right_2 = new_edge_index_right_1 + 1

        left_top_back_panel: EdgeInfoType = []
        right_top_back_panel: EdgeInfoType = []
        left_top_back_edge_info_1 = {
            "start": top_back_u_neck_edge_left_vertex.copy(),
            "start_index": top_back_u_neck_edge_left_index,
            "end": new_vertex_1.copy(),
            "end_index": new_vertex_1_index,
            "stitch_info": {
                "stitch_type": Seam.BOUNDARY,
            },
            # "curvature": top_back_u_neck_curvature,
            "curvature": np.array([0.5, top_back_u_neck_left_curvature]),
            "control_point": control_point_left,
        }
        left_top_back_edge_info_2 = {
            "start": new_vertex_1.copy(),
            "start_index": new_vertex_1_index,
            "end": new_vertex_2.copy(),
            "end_index": new_vertex_2_index,
            "edge_index": new_edge_index_left_1,
            "stitch_info": {
                "stitch_type": Seam.SIDE_BY_SIDE,
                "target_edge_panel": top_back_panel_name,
                "target_edge_index": new_edge_index_right_1,
            },
        }
        left_top_back_edge_info_3 = {
            "start": new_vertex_2.copy(),
            "start_index": new_vertex_2_index,
            "end": top_back_bottom_edge_left_vertex.copy(),
            "end_index": top_back_bottom_edge_left_index,
            "stitch_info": {
                "stitch_type": Seam.BOUNDARY,
            },
        }
        left_top_back_panel.append(left_top_back_edge_info_1)
        left_top_back_panel.append(left_top_back_edge_info_2)
        left_top_back_panel.append(left_top_back_edge_info_3)

        right_top_back_edge_info_1 = {
            "start": top_back_bottom_edge_right_vertex.copy(),
            "start_index": top_back_bottom_edge_right_index,
            "end": new_vertex_2.copy(),
            "end_index": new_vertex_2_index,
            "stitch_info": {
                "stitch_type": Seam.BOUNDARY,
            },
        }
        right_top_back_edge_info_2 = {
            "start": new_vertex_2.copy(),
            "start_index": new_vertex_2_index,
            "end": new_vertex_1.copy(),
            "end_index": new_vertex_1_index,
            "edge_index": new_edge_index_right_1,
            "stitch_info": {
                "stitch_type": Seam.SIDE_BY_SIDE,
                "target_edge_panel": top_back_panel_name,
                "target_edge_index": new_edge_index_left_1,
            },
        }
        right_top_back_edge_info_3 = {
            "start": new_vertex_1.copy(),
            "start_index": new_vertex_1_index,
            "end": top_back_u_neck_edge_right_vertex.copy(),
            "end_index": top_back_u_neck_edge_right_index,
            "stitch_info": {
                "stitch_type": Seam.BOUNDARY,
            },
            # "curvature": top_back_u_neck_curvature,
            "curvature": np.array([0.5, top_back_u_neck_right_curvature]),
            "control_point": control_point_right,
        }
        right_top_back_panel.append(right_top_back_edge_info_1)
        right_top_back_panel.append(right_top_back_edge_info_2)
        right_top_back_panel.append(right_top_back_edge_info_3)

        # construct left_top_back_panel and right_top_back_panel
        prev_end_vertex_index = top_back_bottom_edge_left_index
        while True:
            complete_flag = False
            for edge_info in top_back_panel:
                if edge_info["start_index"] == prev_end_vertex_index:
                    left_top_back_panel.append(edge_info)
                    if edge_info["end_index"] == top_back_u_neck_edge_left_index:
                        complete_flag = True
                        break
                    prev_end_vertex_index = edge_info["end_index"]
            if complete_flag:
                break
        prev_end_vertex_index = top_back_u_neck_edge_right_index
        while True:
            complete_flag = False
            for edge_info in top_back_panel:
                if edge_info["start_index"] == prev_end_vertex_index:
                    right_top_back_panel.append(edge_info)
                    if edge_info["end_index"] == top_back_bottom_edge_right_index:
                        complete_flag = True
                        break
                    prev_end_vertex_index = edge_info["end_index"]
            if complete_flag:
                break

        # reassign edge_index
        edge_index_map: Dict[
            str, Dict[int, Dict[str, Union[int, str]]]
        ] = {}  # panel_name -> before_edge_index -> {panel_name: str, edge_index: int}

        # Note that left is right, right is left from the perspective of the person wearing the clothes
        edge_index_map[top_back_panel_name] = {}
        for i, edge_info in enumerate(left_top_back_panel):
            prev_edge_index = edge_info.get("edge_index")
            if prev_edge_index is not None:
                edge_index_map[top_back_panel_name][prev_edge_index] = {
                    "panel_name": "top_back_right",
                    "edge_index": i,
                }
            # update panel name
            edge_info["panel_name"] = "top_back_right"
            # update edge_index
            edge_info["edge_index"] = i
            # update start_index and end_index
            edge_info["start_index"] = i
            edge_info["end_index"] = i + 1
        for i, edge_info in enumerate(right_top_back_panel):
            prev_edge_index = edge_info.get("edge_index")
            if prev_edge_index is not None:
                edge_index_map[top_back_panel_name][prev_edge_index] = {
                    "panel_name": "top_back_left",
                    "edge_index": i,
                }
            # update panel name
            edge_info["panel_name"] = "top_back_left"
            # update edge_index
            edge_info["edge_index"] = i
            # update start_index and end_index
            edge_info["start_index"] = i
            edge_info["end_index"] = i + 1

        panel_to_edge_info.pop(top_back_panel_name)
        panel_to_edge_info["top_back_right"] = left_top_back_panel
        panel_to_edge_info["top_back_left"] = right_top_back_panel
        for panel_name, edge_infos in panel_to_edge_info.items():
            for edge_info in edge_infos:
                stitch_info = edge_info.get("stitch_info")
                if stitch_info is not None:
                    target_edge_panel = stitch_info.get("target_edge_panel")
                    target_edge_index = stitch_info.get("target_edge_index")
                    if target_edge_panel is not None and target_edge_index is not None:
                        target_edge_panel_map = edge_index_map.get(target_edge_panel)
                        if target_edge_panel_map is not None:
                            target_edge_index = target_edge_panel_map.get(
                                target_edge_index
                            )
                            if target_edge_index is not None:
                                stitch_info["target_edge_panel"] = target_edge_index[
                                    "panel_name"
                                ]
                                stitch_info["target_edge_index"] = target_edge_index[
                                    "edge_index"
                                ]

        # remove top_back panel from the panel_to_edge_info completely
        NeuralTailorConverter.remove_panel_from_edge_info(
            panel_to_edge_info, [top_back_panel_name]
        )

    @staticmethod
    def separate_top_front_panel(panel_to_edge_info: PanelToEdgeInfoType):
        top_front_panel_name = "top_front"
        # top_front_panel_name = "up_front"
        top_front_panel = panel_to_edge_info[top_front_panel_name]
        # assume the top_front has V-neck
        x_center = 0.0
        for edge_info in top_front_panel:
            x_center += edge_info["start"][0]
        x_center /= len(top_front_panel)

        v_neck_bottom_vertex_index: Optional[int] = None
        v_neck_bottom_vertex: Optional[np.ndarray] = None
        min_distance = 1e9
        for edge_info in top_front_panel:
            if np.abs(edge_info["start"][0] - x_center) < min_distance:
                min_distance = np.abs(edge_info["start"][0] - x_center)
                v_neck_bottom_vertex_index = edge_info["start_index"]
                v_neck_bottom_vertex = edge_info["start"]

        top_front_bottom_edge_left_index: Optional[int] = None
        top_front_bottom_edge_right_index: Optional[int] = None
        top_front_bottom_edge_left_vertex: Optional[np.ndarray] = None
        top_front_bottom_edge_right_vertex: Optional[np.ndarray] = None
        for edge_info in top_front_panel:
            if (
                (edge_info["start"][0] < x_center and edge_info["end"][0] > x_center)
                or (edge_info["start"][0] > x_center and edge_info["end"][0] < x_center)
                and edge_info["start_index"] != v_neck_bottom_vertex_index
                and edge_info["end_index"] != v_neck_bottom_vertex_index
            ):
                right_index = (
                    edge_info["start_index"]
                    if edge_info["start"][0] > x_center
                    else edge_info["end_index"]
                )
                left_index = (
                    edge_info["start_index"]
                    if edge_info["start"][0] < x_center
                    else edge_info["end_index"]
                )
                top_front_bottom_edge_left_index = left_index
                top_front_bottom_edge_right_index = right_index
                top_front_bottom_edge_left_vertex = (
                    edge_info["start"]
                    if edge_info["start"][0] < x_center
                    else edge_info["end"]
                )
                top_front_bottom_edge_right_vertex = (
                    edge_info["start"]
                    if edge_info["start"][0] > x_center
                    else edge_info["end"]
                )

        # new_vertex_index = (len(top_front_panel) + 1) // 2
        new_vertex_index = len(top_front_panel)
        new_edge_index_left = len(top_front_panel)
        new_edge_index_right = new_edge_index_left + 1
        new_vertex = (
            top_front_bottom_edge_left_vertex + top_front_bottom_edge_right_vertex
        ) / 2

        left_top_front_panel: EdgeInfoType = []
        right_top_front_panel: EdgeInfoType = []
        left_top_front_edge_info_1 = {
            "start": top_front_bottom_edge_left_vertex.copy(),
            "start_index": top_front_bottom_edge_left_index,
            "end": new_vertex.copy(),
            "end_index": new_vertex_index,
            "stitch_info": {
                "stitch_type": Seam.BOUNDARY,
            },
        }
        left_top_front_edge_info_2 = {
            "start": new_vertex.copy(),
            "start_index": new_vertex_index,
            "end": v_neck_bottom_vertex.copy(),
            "end_index": v_neck_bottom_vertex_index,
            "edge_index": new_edge_index_left,
            "stitch_info": {
                "stitch_type": Seam.SIDE_BY_SIDE,
                "target_edge_panel": top_front_panel_name,
                "target_edge_index": new_edge_index_right,
            },
        }
        left_top_front_panel.append(left_top_front_edge_info_1)
        left_top_front_panel.append(left_top_front_edge_info_2)

        right_top_front_edge_info_1 = {
            "start": v_neck_bottom_vertex.copy(),
            "start_index": v_neck_bottom_vertex_index,
            "end": new_vertex.copy(),
            "end_index": new_vertex_index,
            "edge_index": new_edge_index_right,
            "stitch_info": {
                "stitch_type": Seam.SIDE_BY_SIDE,
                "target_edge_panel": top_front_panel_name,
                "target_edge_index": new_edge_index_left,
            },
        }
        right_top_front_edge_info_2 = {
            "start": new_vertex.copy(),
            "start_index": new_vertex_index,
            "end": top_front_bottom_edge_right_vertex.copy(),
            "end_index": top_front_bottom_edge_right_index,
            "stitch_info": {
                "stitch_type": Seam.BOUNDARY,
            },
        }
        right_top_front_panel.append(right_top_front_edge_info_1)
        right_top_front_panel.append(right_top_front_edge_info_2)

        # construct left_top_front_panel
        prev_end_vertex_index = v_neck_bottom_vertex_index
        while True:
            complete_flag = False
            for edge_info in top_front_panel:
                if edge_info["start_index"] == prev_end_vertex_index:
                    left_top_front_panel.append(edge_info)
                    if edge_info["end_index"] == top_front_bottom_edge_left_index:
                        complete_flag = True
                        break
                    prev_end_vertex_index = edge_info["end_index"]

            if complete_flag:
                break

        # construct right_top_front_panel
        prev_end_vertex_index = top_front_bottom_edge_right_index
        while True:
            complete_flag = False
            for edge_info in top_front_panel:
                if edge_info["start_index"] == prev_end_vertex_index:
                    right_top_front_panel.append(edge_info)
                    if edge_info["end_index"] == v_neck_bottom_vertex_index:
                        complete_flag = True
                        break
                    prev_end_vertex_index = edge_info["end_index"]

            if complete_flag:
                break

        # reassign edge_index
        edge_index_map: Dict[
            str, Dict[int, Dict[str, Union[int, str]]]
        ] = {}  # panel_name -> before_edge_index -> {panel_name: str, edge_index: int}

        # Note that left is right, right is left from the perspective of the person wearing the clothes
        edge_index_map[top_front_panel_name] = {}
        for i, edge_info in enumerate(left_top_front_panel):
            prev_edge_index = edge_info.get("edge_index")
            if prev_edge_index is not None:
                edge_index_map[top_front_panel_name][prev_edge_index] = {
                    "panel_name": "top_front_right",
                    "edge_index": i,
                }
            # update panel name
            edge_info["panel_name"] = "top_front_right"
            # update edge_index
            edge_info["edge_index"] = i
            # update start_index and end_index
            edge_info["start_index"] = i
            edge_info["end_index"] = i + 1

        for i, edge_info in enumerate(right_top_front_panel):
            prev_edge_index = edge_info.get("edge_index")
            if prev_edge_index is not None:
                edge_index_map[top_front_panel_name][prev_edge_index] = {
                    "panel_name": "top_front_left",
                    "edge_index": i,
                }
            # update panel name
            edge_info["panel_name"] = "top_front_left"
            # update edge_index
            edge_info["edge_index"] = i
            # update start_index and end_index
            edge_info["start_index"] = i
            edge_info["end_index"] = i + 1

        panel_to_edge_info.pop(top_front_panel_name)
        panel_to_edge_info["top_front_right"] = left_top_front_panel
        panel_to_edge_info["top_front_left"] = right_top_front_panel

        for panel_name, edge_infos in panel_to_edge_info.items():
            for edge_info in edge_infos:
                stitch_info = edge_info.get("stitch_info")
                if stitch_info is not None:
                    target_edge_panel = stitch_info.get("target_edge_panel")
                    target_edge_index = stitch_info.get("target_edge_index")
                    if target_edge_panel is not None and target_edge_index is not None:
                        target_edge_panel_map = edge_index_map.get(target_edge_panel)
                        if target_edge_panel_map is not None:
                            target_edge_index = target_edge_panel_map.get(
                                target_edge_index
                            )
                            if target_edge_index is not None:
                                stitch_info["target_edge_panel"] = target_edge_index[
                                    "panel_name"
                                ]
                                stitch_info["target_edge_index"] = target_edge_index[
                                    "edge_index"
                                ]

        # remove top_front panel from the panel_to_edge_info completely
        NeuralTailorConverter.remove_panel_from_edge_info(
            panel_to_edge_info, [top_front_panel_name]
        )

    @staticmethod
    def remove_panel_from_edge_info(
        panel_to_edge_info: PanelToEdgeInfoType, removing_panel_names: List[str]
    ):
        """
        Remove the specified panels from the edge info
        """
        for panel_name in list(panel_to_edge_info.keys()):
            if panel_name in removing_panel_names:
                panel_to_edge_info.pop(panel_name)
            else:
                for edge_info in panel_to_edge_info[panel_name]:
                    stitch_info = edge_info.get("stitch_info")
                    target_edge_panel = stitch_info.get("target_edge_panel")
                    if target_edge_panel in removing_panel_names:
                        stitch_info["stitch_type"] = Seam.BOUNDARY
                        edge_info["stitch_info"] = {
                            "stitch_type": Seam.BOUNDARY,
                        }

    @staticmethod
    def check_intersection_of_two_line_segments(
        a_start: np.ndarray, a_end: np.ndarray, b_start: np.ndarray, b_end: np.ndarray
    ) -> bool:
        """
        Check whether two line segments intersect or not
        """
        # Check if the two line segments are parallel
        a = a_end - a_start
        s = np.cross(b_start - a_start, a)
        t = np.cross(b_end - a_start, a)
        if s * t >= 0:
            return False

        b = b_end - b_start
        s = np.cross(a_start - b_start, b)
        t = np.cross(a_end - b_start, b)
        if s * t >= 0:
            return False

        return True

    @staticmethod
    def check_intersection(
        panel_to_edge_info: PanelToEdgeInfoType,
    ) -> bool:
        """
        Check whether there is an intersection between the edges within the same panel
        """
        for panel_name in panel_to_edge_info:
            edge_list = panel_to_edge_info[panel_name]
            for i, edge_info_1 in enumerate(edge_list):
                for j, edge_info_2 in enumerate(edge_list):
                    if i == j:
                        continue
                    a_start = edge_info_1["start"][:2]
                    a_end = edge_info_1["end"][:2]
                    b_start = edge_info_2["start"][:2]
                    b_end = edge_info_2["end"][:2]
                    if (
                        np.allclose(a_start, b_start)
                        or np.allclose(a_start, b_end)
                        or np.allclose(a_end, b_start)
                        or np.allclose(a_end, b_end)
                    ):
                        continue
                    if NeuralTailorConverter.check_intersection_of_two_line_segments(
                        a_start, a_end, b_start, b_end
                    ):
                        return True
        return False

    @staticmethod
    def judge_top(panel_name: str) -> bool:
        # judge is the panel is top or not
        if "top" in panel_name:
            return True
        if panel_name == "left_ftorso":
            return True
        if panel_name == "left_btorso":
            return True
        if panel_name == "right_ftorso":
            return True
        if panel_name == "right_btorso":
            return True
        return False

    @staticmethod
    def judge_front(panel_name: str) -> bool:
        if "front" in panel_name:
            return True
        if "lf" in panel_name:
            return True
        if "rf" in panel_name:
            return True
        if "left_f" in panel_name:
            return True
        if "right_f" in panel_name:
            return True
        if panel_name.startswith("left_f"):
            return True
        if panel_name.startswith("right_f"):
            return True
        return False

    @staticmethod
    def calculate_edge_info_centroid(edge_info: EdgeInfoType) -> np.ndarray:
        # TODO: consider that there are curved edges
        vertex_matrix = NeuralTailorConverter.construct_np_matrix_from_edge_info(
            edge_info
        )
        # convert to sg.Polygon
        polygon = sg.Polygon(vertex_matrix[:, :2])
        return np.array(polygon.centroid.coords[0])

    # @staticmethod
    # def construct_np_matrix_from_edge_info(
    #     edge_info: List[Dict[str, np.ndarray]],
    # ) -> np.ndarray:
    #     np_matrix = np.zeros((len(edge_info), 4))
    #     for i in range(len(edge_info)):
    #         edge_start = NeuralTailorConverter.find_edge_by_start_vertex_index(i)
    #         np_matrix[i] = edge_start["start"]
    #     return np_matrix

    @staticmethod
    def construct_np_matrix_from_edge_info(
        edge_info: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        np_matrix = np.zeros((len(edge_info), 4))
        for i, edge in enumerate(edge_info):
            np_matrix[i][:] = edge["start"]
        return np_matrix

    @staticmethod
    def apply_affine_matrix_edge_info(
        affine_matrix: np.ndarray, edge_info: List[Dict[str, np.ndarray]]
    ):
        np_matrix: np.ndarray = (
            NeuralTailorConverter.construct_np_matrix_from_edge_info(edge_info)
        )
        np_matrix = np.dot(affine_matrix, np_matrix.T).T
        for i, edge in enumerate(edge_info):
            # start = edge.get("start")
            # end = edge.get("end")
            # start = np.dot(affine_matrix, start)
            # end = np.dot(affine_matrix, end)
            # not edge["start"] = target_end because edge["start"] does not change the reference object
            # edge["start"][:] = start
            # edge["end"][:] = end
            # edge["start"] = start
            # edge["end"] = end
            edge["start"] = np_matrix[i]
            edge["end"] = np_matrix[(i + 1) % len(edge_info)]

    @staticmethod
    def determine_scale_of_panel_edge_info(
        panel_to_edge_info: PanelToEdgeInfoType,
        min_x: float = TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        max_x: float = TEMPLATE_W - TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        min_y: float = TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        max_y: float = TEMPLATE_W - TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
    ) -> Tuple[float, float, float, float, float, bool]:
        # max_x and max_y should not be TEMPLATE_W because the number of the edges saved in a GarmentImage file is GARMENT_IMAGE_RESOLUTION * GARMENT_IMAGE_RESOLUTION, not (GARMENT_IMAGE_RESOLUTION + 1) * (GARMENT_IMAGE_RESOLUTION + 1
        # x軸方向、y軸方向それぞれに保存できる辺の数はGARMENT_IMAGE_RESOLUTIONである (Faceの数がGARMENT_IMAGE_RESOLUTIONであるため)が、max_x, max_yをTEMPLATE_Wに設定すると、GARMENT_IMAGE_RESOLUTION + 1の数の辺を保存することを前提としているため、エラーが発生する
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        is_garmentcodedata = False
        for panel_name in panel_to_edge_info.keys():
            edge_info = panel_to_edge_info[panel_name]
            for i, edge in enumerate(edge_info):
                start = edge.get("start")
                end = edge.get("end")
                control_point = start
                tmp_is_garmentcodedata = edge.get("is_garmentcodedata", False)
                if tmp_is_garmentcodedata:
                    is_garmentcodedata = True
                # curvature = edge.get("curvature")
                # if curvature is not None:
                #     control_point = NeuralTailorConverter.calculate_control_point(
                #         start, end, curvature
                #     )
                x_min = min(x_min, start[0], end[0], control_point[0])
                x_max = max(x_max, start[0], end[0], control_point[0])
                y_min = min(y_min, start[1], end[1], control_point[1])
                y_max = max(y_max, start[1], end[1], control_point[1])
        x_scale = (max_x - min_x) / (x_max - x_min)
        y_scale = (max_y - min_y) / (y_max - y_min)
        scale = min(x_scale, y_scale)
        return scale, x_max, x_min, y_max, y_min, is_garmentcodedata

    @staticmethod
    def resize_panel_edge_info(
        panel_to_edge_info: PanelToEdgeInfoType,
        scale: float,
        x_max: float,
        x_min: float,
        y_max: float,
        y_min: float,
        is_garmentcodedata: bool = False,
        min_x: float = TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        max_x: float = TEMPLATE_W - TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        min_y: float = TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
        max_y: float = TEMPLATE_W - TEMPLATE_W / GARMENT_IMAGE_RESOLUTION,
    ):
        # print("x_max + x_min", (x_max + x_min)/2)
        # print((min_x + max_x) / 2)
        translate_to_origin = np.array(
            [
                # [1, 0, 0, -(x_max + x_min) / 2],
                [1, 0, 0, 0],
                [0, 1, 0, -(y_max + y_min) / 2 if is_garmentcodedata else 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        scale_matrix = np.array(
            [
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        translate_to_new_center = np.array(
            [
                [1, 0, 0, (min_x + max_x) / 2],
                [0, 1, 0, (min_y + max_y) / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        affine_matrix = translate_to_new_center @ scale_matrix @ translate_to_origin

        for panel_name in panel_to_edge_info.keys():
            edge_info = panel_to_edge_info[panel_name]
            NeuralTailorConverter.apply_affine_matrix_edge_info(
                affine_matrix, edge_info
            )

    @staticmethod
    def construct_seam_from_edge_info(
        edge_info: Dict[str, Union[np.ndarray, Dict]],
        bezier_subdivision: int = 10,
    ) -> Seam:
        start = edge_info.get("start")
        end = edge_info.get("end")
        start_uv = edge_info.get("start_uv")
        end_uv = edge_info.get("end_uv")
        curvature = edge_info.get("curvature")
        is_garmentcodedata = edge_info.get("is_garmentcodedata", False)
        stitch_info = edge_info.get("stitch_info")
        points: List[Vertex2D] = []
        if curvature is not None:
            control_point = NeuralTailorConverter.calculate_control_point(
                start, end, curvature, is_garmentcodedata=is_garmentcodedata
            )
            t = np.linspace(0, 1, bezier_subdivision)
            x: np.ndarray = (
                (1 - t) ** 2 * start[0]
                + 2 * (1 - t) * t * control_point[0]
                + t**2 * end[0]
            )
            y: np.ndarray = (
                (1 - t) ** 2 * start[1]
                + 2 * (1 - t) * t * control_point[1]
                + t**2 * end[1]
            )
            for _x, _y in zip(
                x[1:-1], y[1:-1]
            ):  # TODO: this should be x[1:], y[1:]. I'll fix it after the submission
                points.append(Vertex2D(_x, _y))

        seam: Seam = Seam(points)
        seam.start = Vertex2D(start[0], start[1])
        seam.end = Vertex2D(end[0], end[1])
        if start_uv is not None:
            seam.start.uv = Vertex2D(start_uv[0], start_uv[1])
        else:
            seam.start.uv = Vertex2D(start[0], start[1])
        if end_uv is not None:
            seam.end.uv = Vertex2D(end_uv[0], end_uv[1])
        else:
            seam.end.uv = Vertex2D(end[0], end[1])
        seam.type = stitch_info.get("stitch_type") if stitch_info is not None else None
        seam.set_stroke()
        # print(seam.stroke)
        return seam

    @staticmethod
    def construct_pieces_from_panel_edge_info(
        panel_to_edge_info: PanelToEdgeInfoType,
        is_back: bool = False,
    ) -> List[Piece]:
        pieces: List[Piece] = []
        for panel_name in panel_to_edge_info.keys():
            edge_infos = panel_to_edge_info[panel_name]
            seams: List[Seam] = []
            for edge_info in edge_infos:
                seam: Seam = NeuralTailorConverter.construct_seam_from_edge_info(
                    edge_info
                )
                seams.append(seam)
            piece: Piece = Piece(seams)
            piece.reversed = is_back
            if piece.reversed:
                piece.layer = 1
            else:
                piece.layer = 0
            piece.update_network()
            pieces.append(piece)
        return pieces

    @staticmethod
    def load_spec_json(
        file_name: str,
        parameterize: bool = False,
        random_parameterize: bool = False,
        align_stitches: bool = True,
        force_alignment: bool = False,
        y_axis_alignment: bool = False,
        wb_alignment: bool = False,
        dart_alignment: bool = False,
        skirt_alignment: bool = False,
        front_to_back_alignment: bool = False,
        merged_panels: Optional[List[str]] = None,
        removed_panels: Optional[List[str]] = None,
        add_uv: bool = False,
        predefined_scale: Optional[int] = None,
    ) -> List[Piece]:
        file_ext: str = os.path.splitext(file_name)[-1]
        assert file_ext == ".json", f"The file extension must be '.json', {file_name}"
        with open(file_name, "r") as f:
            data: Dict[str, Any] = json.load(f)
        panels: Dict = data["pattern"]["panels"]
        parameters: Optional[Dict] = None
        parameter_order: Optional[List[str]] = None
        if parameterize or random_parameterize:
            parameters = data.get("parameters")
            parameter_order = data.get("parameter_order")
            if random_parameterize:
                NeuralTailorConverter.set_random_parameter_values(parameters)
        stitches: List[List[Dict[str, Union[int, str]]]] = data["pattern"]["stitches"]

        # front panel
        front_panels: Dict = {
            k: v for k, v in panels.items() if NeuralTailorConverter.judge_front(k)
        }
        # back panel
        back_panels: Dict = {
            k: v for k, v in panels.items() if not NeuralTailorConverter.judge_front(k)
        }

        front_panel_to_edge_info = NeuralTailorConverter.construct_panel_to_edge_info(
            front_panels, parameters=parameters, parameter_order=parameter_order
        )
        NeuralTailorConverter.add_stitches_to_edge_info(
            front_panel_to_edge_info, stitches
        )
        # TODO: apply_stich_alignment should be done using uv
        if align_stitches:
            NeuralTailorConverter.apply_stitch_alignment(
                front_panel_to_edge_info,
                force_alignment=force_alignment,
                y_axis_alignment=y_axis_alignment,
                dart_alignment=dart_alignment,
                skirt_alignment=skirt_alignment,
            )

        back_panel_to_edge_info = NeuralTailorConverter.construct_panel_to_edge_info(
            back_panels, parameters=parameters, parameter_order=parameter_order
        )
        NeuralTailorConverter.add_stitches_to_edge_info(
            back_panel_to_edge_info, stitches
        )
        if removed_panels is not None:
            NeuralTailorConverter.remove_panel_from_edge_info(
                front_panel_to_edge_info, removed_panels
            )
            NeuralTailorConverter.remove_panel_from_edge_info(
                back_panel_to_edge_info, removed_panels
            )
        if align_stitches:
            NeuralTailorConverter.apply_stitch_alignment(
                back_panel_to_edge_info,
                force_alignment=force_alignment,
                y_axis_alignment=y_axis_alignment,
                dart_alignment=dart_alignment,
            )

        # front_panel_to_edge_info + back_panel_to_edge_info
        both_panel_to_edge_info = {
            **front_panel_to_edge_info,
            **back_panel_to_edge_info,
        }
        if merged_panels is not None:
            NeuralTailorConverter.merge_edge_info_v2(
                both_panel_to_edge_info, merged_panels
            )
            front_panel_to_edge_info = {
                panel_name: v
                for panel_name, v in both_panel_to_edge_info.items()
                if NeuralTailorConverter.judge_front(panel_name)
            }
            back_panel_to_edge_info = {
                panel_name: v
                for panel_name, v in both_panel_to_edge_info.items()
                if not NeuralTailorConverter.judge_front(panel_name)
            }

        if front_to_back_alignment:
            NeuralTailorConverter.apply_front_to_back_stitch_alignment(
                both_panel_to_edge_info
            )

        if wb_alignment:
            NeuralTailorConverter.apply_waistband_alignment(both_panel_to_edge_info)

        scale, x_max, x_min, y_max, y_min, is_garmentcodedata = (
            NeuralTailorConverter.determine_scale_of_panel_edge_info(
                panel_to_edge_info=both_panel_to_edge_info,
            )
        )
        if predefined_scale is not None:
            assert scale > predefined_scale, (
                f"scale: {scale}, predefined_scale: {predefined_scale}"
            )
            scale = predefined_scale

        NeuralTailorConverter.resize_panel_edge_info(
            front_panel_to_edge_info,
            scale=scale,
            x_max=x_max,
            x_min=x_min,
            y_max=y_max,
            y_min=y_min,
            is_garmentcodedata=is_garmentcodedata,
        )
        NeuralTailorConverter.resize_panel_edge_info(
            back_panel_to_edge_info,
            scale=scale,
            x_max=x_max,
            x_min=x_min,
            y_max=y_max,
            y_min=y_min,
            is_garmentcodedata=is_garmentcodedata,
        )
        if add_uv:
            NeuralTailorConverter.add_uv_to_edge_info(front_panel_to_edge_info)
            NeuralTailorConverter.add_uv_to_edge_info(back_panel_to_edge_info)

        front_pieces: List[Piece] = (
            NeuralTailorConverter.construct_pieces_from_panel_edge_info(
                front_panel_to_edge_info, is_back=False
            )
        )

        back_pieces: List[Piece] = (
            NeuralTailorConverter.construct_pieces_from_panel_edge_info(
                back_panel_to_edge_info, is_back=True
            )
        )
        return front_pieces + back_pieces

    @staticmethod
    def visualize_panels(
        panels: Dict,
        parameters: Optional[Dict] = None,
        parameter_order: Optional[List[str]] = None,
        stitches: Optional[List[List[Dict[str, Union[int, str]]]]] = None,
        align_stitches: bool = False,
        force_alignment: bool = False,
        y_axis_alignment: bool = False,
        wb_alignment: bool = False,
        dart_alignment: bool = False,
        merged_panels: Optional[List[str]] = None,
        add_uv: bool = False,
        bezier_subdivision: int = 10,
        output_file_path: Optional[str] = None,
    ):
        panel_to_edge_info = (
            NeuralTailorConverter.construct_panel_to_edge_info_and_do_preprocess(
                panels=panels,
                parameters=parameters,
                parameter_order=parameter_order,
                stitches=stitches,
                align_stitches=align_stitches,
                force_alignment=force_alignment,
                y_axis_alignment=y_axis_alignment,
                wb_alignment=wb_alignment,
                dart_alignment=dart_alignment,
                merged_panels=merged_panels,
                add_uv=add_uv,
            )
        )

        NeuralTailorConverter.visualize_panel_to_edge_info(
            panel_to_edge_info,
            add_uv=add_uv,
            bezier_subdivision=bezier_subdivision,
            output_file_path=output_file_path,
        )

    @staticmethod
    def visualize_panel_to_edge_info(
        panel_to_edge_info: Dict[str, List[Dict[str, np.ndarray]]],
        add_uv: bool = False,
        bezier_subdivision: int = 10,
        output_file_path: Optional[str] = None,
    ):
        # Plot the edges
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        for panel_name in panel_to_edge_info.keys():
            edge_info = panel_to_edge_info[panel_name]
            for i, edge in enumerate(edge_info):
                start = edge.get("start")
                end = edge.get("end")
                start_uv = edge.get("start_uv")
                end_uv = edge.get("end_uv")
                curvature = edge.get("curvature")
                control_point = edge.get("control_point")
                is_garmentcodedata = edge.get("is_garmentcodedata", False)
                stitch_info = edge.get("stitch_info")
                assert start is not None, "start is None"
                assert end is not None, "end is None"
                ax.scatter(start[0], start[1], c="b", marker="o")
                ax.scatter(end[0], end[1], c="b", marker="o")
                if add_uv and start_uv is not None:
                    ax.scatter(start_uv[0], start_uv[1], c="green", marker="+", s=300)
                if add_uv and end_uv is not None:
                    ax.scatter(end_uv[0], end_uv[1], c="green", marker="+", s=300)
                curvature = edge.get("curvature")
                if curvature is not None and control_point is None:
                    control_point = NeuralTailorConverter.calculate_control_point(
                        start, end, curvature, is_garmentcodedata=is_garmentcodedata
                    )
                if control_point is None:
                    x_min = min(x_min, start[0], end[0])
                    x_max = max(x_max, start[0], end[0])
                    y_min = min(y_min, start[1], end[1])
                    y_max = max(y_max, start[1], end[1])
                else:
                    x_min = min(x_min, start[0], end[0], control_point[0])
                    x_max = max(x_max, start[0], end[0], control_point[0])
                    y_min = min(y_min, start[1], end[1], control_point[1])
                    y_max = max(y_max, start[1], end[1], control_point[1])
                if curvature is not None:
                    control_point = NeuralTailorConverter.calculate_control_point(
                        start, end, curvature, is_garmentcodedata=is_garmentcodedata
                    )
                    t = np.linspace(0, 1, bezier_subdivision)
                    x: np.ndarray = (
                        (1 - t) ** 2 * start[0]
                        + 2 * (1 - t) * t * control_point[0]
                        + t**2 * end[0]
                    )
                    y: np.ndarray = (
                        (1 - t) ** 2 * start[1]
                        + 2 * (1 - t) * t * control_point[1]
                        + t**2 * end[1]
                    )
                    ax.plot(
                        x,
                        y,
                        "r",
                        linestyle=Seam.boundary_types_to_linestyle[
                            stitch_info.get("stitch_type")
                            if stitch_info is not None
                            else Seam.BOUNDARY
                        ],
                    )
                    ax.scatter(control_point[0], control_point[1], c="b", marker="x")
                else:
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        "r",
                        linestyle=Seam.boundary_types_to_linestyle[
                            stitch_info.get("stitch_type")
                            if stitch_info is not None
                            else Seam.BOUNDARY
                        ],
                    )

        # Set the axis limits and labels
        ax.set_xlim(x_min - 5, x_max + 5)
        ax.set_ylim(y_min - 5, y_max + 5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Garment Visualization")

        # plt.savefig("media/dress_0XAVEH5G53/front_visualization_stitch_aligned.png")
        if output_file_path is not None:
            plt.savefig(output_file_path)
        else:
            plt.show()
        plt.close()
