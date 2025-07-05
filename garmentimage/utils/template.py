from __future__ import annotations

import os
from copy import copy
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from garmentimage.utils.draw_panel import DrawPanel
from garmentimage.utils.edge2d import Edge2D
from garmentimage.utils.face import Face2D
from garmentimage.utils.line2d import Line2D
from garmentimage.utils.mesh import Mesh2D
from garmentimage.utils.piece import Piece
from garmentimage.utils.seam import Seam
from garmentimage.utils.utils import GARMENT_IMAGE_RESOLUTION, TEMPLATE_W
from garmentimage.utils.vertex2d import Vector2, Vertex2D


class Template(Mesh2D):
    W: int = TEMPLATE_W
    N: int = GARMENT_IMAGE_RESOLUTION
    X0: int = 0
    Y0: int = 0

    class Direction(Enum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3
        NONE = 4

        @staticmethod
        def is_opposite(d1: "Template.Direction", d2: "Template.Direction") -> bool:
            return (
                (d1 == Template.Direction.LEFT and d2 == Template.Direction.RIGHT)
                or (d1 == Template.Direction.RIGHT and d2 == Template.Direction.LEFT)
                or (d1 == Template.Direction.UP and d2 == Template.Direction.DOWN)
                or (d1 == Template.Direction.DOWN and d2 == Template.Direction.UP)
            )

    def __init__(self):
        super().__init__()
        self.name: str = ""
        self.grid_vertices: List[List[Optional[Vertex2D]]] = [
            [None for _ in range(self.N + 1)] for _ in range(self.N + 1)
        ]
        self.h_edges: List[List[Optional[Edge2D]]] = [
            [None for _ in range(self.N + 1)] for _ in range(self.N)
        ]
        self.v_edges: List[List[Optional[Edge2D]]] = [
            [None for _ in range(self.N)] for _ in range(self.N + 1)
        ]
        self.x_edges: List[List[Optional[Edge2D]]] = [
            [None for _ in range(self.N)] for _ in range(self.N)
        ]
        self.grid_faces: List[List[Optional[Face2D]]] = [
            [None for _ in range(self.N)] for _ in range(self.N)
        ]
        self.original_meshes: List[Mesh2D] = []
        self.meshes: List[Mesh2D] = []
        self.original_template_pieces: List[TemplatePiece] = []
        self.template_pieces: List[TemplatePiece] = []
        self.edge_global_index_to_piece_index_and_local_index: Optional[
            Dict[int, Tuple[int, int]]
        ] = None
        self.piece_index_and_local_index_to_edge_global_index: Optional[
            Dict[Tuple[int, int], int]
        ] = None
        # SIDE_BY_SIDE edges pairs
        self.stitched_edges_pair: List[
            Tuple[int, int]
        ] = []  # [(edge1_global_index, edge2_global_index), ...]
        self.side_by_side_stitched_edges_pairs = []  # [(edge_1_global_index, edge_2_global_index), ...]
        self.front_to_back_stitched_edges_pairs = []  # [(edge_front_global_index, edge_back_global_index), ...]

        # vertices
        for x in range(self.N + 1):
            for y in range(self.N + 1):
                self.grid_vertices[x][y] = Vertex2D(1.0 * x / self.N, 1.0 * y / self.N)
                self.grid_vertices[x][y].grid_xy = [x, y]
                self.grid_vertices[x][y].TX = Vector2(1.0 * self.W / self.N, 0)
                self.grid_vertices[x][y].TY = Vector2(0, 1.0 * self.W / self.N)

        # h_edges
        for x in range(self.N):
            for y in range(self.N + 1):
                v0: Vertex2D = self.grid_vertices[x][y]
                v1: Vertex2D = self.grid_vertices[x + 1][y]
                self.h_edges[x][y] = Edge2D(v0, v1)
                self.h_edges[x][y].direction = Edge2D.HORIZONTAL

        # v_edges
        for x in range(self.N + 1):
            for y in range(self.N):
                v0: Vertex2D = self.grid_vertices[x][y]
                v1: Vertex2D = self.grid_vertices[x][y + 1]
                self.v_edges[x][y] = Edge2D(v0, v1)
                self.v_edges[x][y].direction = Edge2D.VERTICAL

        # faces
        for x in range(self.N):
            for y in range(self.N):
                face: Face2D = Face2D(
                    self.h_edges[x][y],
                    self.v_edges[x + 1][y],
                    self.h_edges[x][y + 1],
                    self.v_edges[x][y],
                )
                self.faces.append(face)
                self.grid_faces[x][y] = face
                face.grid_x = x
                face.grid_y = y

        for x in range(self.N + 1):
            for y in range(self.N + 1):
                if self.grid_vertices[x][y] is not None:
                    self.grid_vertices[x][y].warp(
                        self.convert(self.grid_vertices[x][y])
                    )
                    self.vertices.append(self.grid_vertices[x][y])

        for i in range(self.N):
            for j in range(self.N + 1):
                if self.h_edges[i][j] is not None:
                    self.edges.append(self.h_edges[i][j])

        for i in range(self.N + 1):
            for j in range(self.N):
                if self.v_edges[i][j] is not None:
                    self.edges.append(self.v_edges[i][j])

    def symmetrize_faces(
        self, target_faces: Optional[List[Face2D]] = None, is_left_base: bool = False
    ) -> None:
        """
        Symmetrize the faces in the template by mirroring them about the vertical axis.
        """
        N = Template.N
        mid_point = N // 2

        for x in range(N):
            for y in range(N):
                if x < mid_point:
                    opposite_x = N - x - 1
                    if is_left_base:
                        if (
                            target_faces is None
                            or self.grid_faces[x][y] in target_faces
                        ):
                            self.mirror_faces(x, y, opposite_x, y)
                    else:
                        if (
                            target_faces is None
                            or self.grid_faces[opposite_x][y] in target_faces
                        ):
                            self.mirror_faces(opposite_x, y, x, y)

    def mirror_faces(self, x1: int, y1: int, x2: int, y2: int) -> None:
        face1 = self.grid_faces[x1][y1]
        face2 = self.grid_faces[x2][y2]
        face2.inside = face1.inside
        if x1 < x2:
            h_edge1_0, v_edge1_1, h_edge1_1, v_edge1_0 = face1.edges
            h_edge2_0, v_edge2_1, h_edge2_1, v_edge2_0 = face2.edges
            for edge1, edge2 in zip(
                [h_edge1_0, v_edge1_1, h_edge1_1, v_edge1_0],
                [h_edge2_0, v_edge2_0, h_edge2_1, v_edge2_1],
            ):
                edge2.align_seam_type(edge1)
            v1 = self.grid_vertices[x1][y1]
            if x1 != 0:
                v2_right = self.grid_vertices[x2 + 1][y2]
                v2_right.Y_EDGE_TYPE = v1.Y_EDGE_TYPE
            v2 = self.grid_vertices[x2][y2]
            v2.X_EDGE_TYPE = v1.X_EDGE_TYPE

        else:
            h_edge1_0, v_edge1_1, h_edge1_1, v_edge1_0 = face1.edges
            h_edge2_0, v_edge2_1, h_edge2_1, v_edge2_0 = face2.edges
            for edge2, edge1 in zip(
                [h_edge2_0, v_edge2_1, h_edge2_1, v_edge2_0],
                [h_edge1_0, v_edge1_0, h_edge1_1, v_edge1_1],
            ):
                edge2.align_seam_type(edge1)

            v1 = self.grid_vertices[x1][y1]
            v2 = self.grid_vertices[x2][y2]
            if x2 != 0:
                v1_right = self.grid_vertices[x1 + 1][y1]
                v2.Y_EDGE_TYPE = v1_right.Y_EDGE_TYPE
            v2.X_EDGE_TYPE = v1.X_EDGE_TYPE

    def reconstruct_pieces_from_faces(
        self,
        is_reversed: bool,
        reject_two_pieces: bool = False,
        desirable_piece_num: Optional[int] = None,
        n_tries: int = 5,
    ) -> None:
        """
        Reconstruct pieces from deformation-embedded template.faces
        Parameters
        ----------
        reject_two_pieces: bool
            If True, reject the case where the number of pieces is 2
            See https://github.com/yukistavailable/Dresscode/issues/190
        """

        count = 0
        max_piece_num = 0
        best_pieces = []
        while count < n_tries:
            pieces: List[Piece] = []
            boundary_grid_edges: Set[Edge2D] = self.get_boundary_grid_new_edges()
            original_boundary_grid_edges: Set[Edge2D] = copy(boundary_grid_edges)

            piece_index = 0
            while boundary_grid_edges:
                seams: List[Seam] = []
                try:
                    closed_path: List[Edge2D] = Template.find_closed_stroke(
                        boundary_grid_edges
                    )
                except Exception:
                    break
                # reject if the path is a single straight stroke
                if Edge2D.is_straight_stroke(closed_path):
                    count += 1
                    # if count == n_tries:
                    #     raise ValueError(
                    #         "Found a invalid piece. The boundary edges of a piece should not be a single straight stroke."
                    #     )
                    break

                for i in range(len(closed_path)):
                    seam: Seam = Seam(
                        [
                            closed_path[i].start,
                            closed_path[(i + 1) % len(closed_path)].start,
                        ]
                    )
                    seam.type = closed_path[i].seam_type
                    seams.append(seam)

                piece: Piece = Piece(seams)
                pieces.append(piece)
                piece_index += 1
                # TODO: consider is_reversed, current implementation of visualization has a bug when is_reversed is True, i don't know why
                # piece.reversed = is_reversed
            count += 1
            if desirable_piece_num is not None and len(pieces) == desirable_piece_num:
                best_pieces = [piece for piece in pieces]
                max_piece_num = len(pieces)
                break
            if len(pieces) > max_piece_num:
                max_piece_num = len(pieces)
                best_pieces = [piece for piece in pieces]

        if len(best_pieces) == 0:
            raise ValueError("Failed to reconstruct pieces from faces")
        else:
            print(
                f"Reconstructed {len(best_pieces)} pieces from faces, max_piece_num: {max_piece_num}"
            )
            for piece in best_pieces:
                template_piece: TemplatePiece = TemplatePiece(
                    _piece=piece, _template=self
                )

    @staticmethod
    def find_closed_stroke(edges: Set[Edge2D]) -> List[Edge2D]:
        """
        Return a closed path in the given edges
        The direction of the edges should be consistent
        """
        prev_edge: Edge2D = edges.pop()
        start_edge: Edge2D = prev_edge
        sorted_edges: List[Edge2D] = [prev_edge]
        next_edge: Optional[Edge2D] = None
        next_low_possible_edge: Optional[Edge2D] = None
        while True:
            next_edge = None
            next_low_possible_edge = None
            for edge in edges:
                if prev_edge.end == edge.start:
                    if edge.seam_type == Seam.SIDE_BY_SIDE:
                        # prevent backflow
                        if edge.end != prev_edge.start:
                            next_edge = edge
                            break
                        else:
                            # U-tern
                            next_low_possible_edge = edge
                    elif edge.seam_type == Seam.BOUNDARY:
                        # turn 90 degrees
                        if prev_edge.left_face == edge.left_face:
                            next_edge = edge
                        # prevent backflow
                        elif (
                            prev_edge.right_face == edge.left_face
                            and prev_edge.left_face == edge.right_face
                        ):
                            next_low_possible_edge = edge
                        else:
                            next_edge = edge
                    elif edge.seam_type == Seam.FRONT_TO_BACK:
                        # prioritize SIDE_BY_SIDE
                        if edge.seam_type == Seam.SIDE_BY_SIDE:
                            next_edge = edge
                        # prevent backflow
                        elif (
                            prev_edge.right_face == edge.left_face
                            and prev_edge.left_face == edge.right_face
                        ):
                            next_low_possible_edge = edge
                        else:
                            next_edge = edge
            if next_edge is None:
                if next_low_possible_edge is not None:
                    # print('next_edge is None next_low_possible_edge')
                    next_edge = next_low_possible_edge
                else:
                    raise Exception("next_edge is None")
            prev_edge = next_edge
            edges.remove(next_edge)
            sorted_edges.append(next_edge)
            if next_edge.end == start_edge.start:
                break
        return sorted_edges

    def get_closed_paths_in_grid(self) -> List[List[Edge2D]]:
        count = 0
        while count < 5:
            boundary_grid_new_edges: List[Edge2D] = self.get_boundary_grid_new_edges(
                treat_none_seam_as_boundary=False
            )
            closed_paths: List[List[Edge2D]] = []
            while boundary_grid_new_edges:
                closed_path: List[Edge2D] = Template.find_closed_stroke(
                    boundary_grid_new_edges
                )
                closed_paths.append(closed_path)
                if Edge2D.is_straight_stroke(closed_path):
                    count += 1
                    if count == 5:
                        raise ValueError(
                            "Found a invalid piece. The boundary edges of a piece should not be a single straight stroke."
                        )
                    break
            return closed_paths

    # TODO: Make the function faster?
    def get_boundary_grid_new_edges(
        self, treat_none_seam_as_boundary: bool = True
    ) -> Set[Edge2D]:
        boundary_grid_edges: Set[Edge2D] = set()
        # h_edges
        for x in range(self.N):
            for y in range(self.N + 1):
                edge: Edge2D = self.h_edges[x][y]
                if edge.seam_type is not None:
                    new_edge: Optional[Edge2D] = None
                    if edge.left_face is not None and edge.left_face.inside == 1:
                        if edge.seam_type == Seam.NONE:
                            if treat_none_seam_as_boundary and (
                                edge.right_face is not None
                                and edge.right_face.inside != 1
                            ):
                                edge.seam_type = Seam.BOUNDARY
                                new_edge: Edge2D = edge.shallow_duplicate()
                        else:
                            new_edge: Edge2D = edge.shallow_duplicate()
                    elif edge.right_face is not None and edge.right_face.inside == 1:
                        if edge.seam_type == Seam.NONE:
                            if treat_none_seam_as_boundary and (
                                edge.left_face is not None
                                and edge.left_face.inside != 1
                            ):
                                edge.seam_type = Seam.BOUNDARY
                                new_edge: Edge2D = edge.shallow_duplicate_reverse()
                        else:
                            new_edge: Edge2D = edge.shallow_duplicate_reverse()
                    if new_edge is not None:
                        boundary_grid_edges.add(new_edge)
                        if edge.seam_type == Seam.SIDE_BY_SIDE or (
                            edge.seam_type == Seam.FRONT_TO_BACK
                            and (
                                edge.left_face is not None
                                and edge.left_face.inside == 1
                            )
                            and (
                                edge.right_face is not None
                                and edge.right_face.inside == 1
                            )
                        ):
                            reversed_edge: Edge2D = edge.shallow_duplicate_reverse()
                            boundary_grid_edges.add(reversed_edge)
                        if edge.seam_type == Seam.BOUNDARY:
                            if edge.is_cut():
                                reversed_edge: Edge2D = edge.shallow_duplicate_reverse()
                                boundary_grid_edges.add(reversed_edge)

        # v_edges
        for x in range(self.N + 1):
            for y in range(self.N):
                edge: Edge2D = self.v_edges[x][y]
                if edge.seam_type is not None:
                    new_edge: Optional[Edge2D] = None
                    if edge.left_face is not None and edge.left_face.inside == 1:
                        if edge.seam_type == Seam.NONE:
                            if treat_none_seam_as_boundary and (
                                edge.right_face is not None
                                and edge.right_face.inside != 1
                            ):
                                edge.seam_type = Seam.BOUNDARY
                                new_edge: Edge2D = edge.shallow_duplicate()
                        else:
                            new_edge: Edge2D = edge.shallow_duplicate()
                    elif edge.right_face is not None and edge.right_face.inside == 1:
                        if edge.seam_type == Seam.NONE:
                            if treat_none_seam_as_boundary and (
                                edge.left_face is not None
                                and edge.left_face.inside != 1
                            ):
                                edge.seam_type = Seam.BOUNDARY
                                new_edge: Edge2D = edge.shallow_duplicate_reverse()
                        else:
                            new_edge: Edge2D = edge.shallow_duplicate_reverse()
                    if new_edge is not None:
                        boundary_grid_edges.add(new_edge)
                        if edge.seam_type == Seam.SIDE_BY_SIDE or (
                            edge.seam_type == Seam.FRONT_TO_BACK
                            and (
                                edge.left_face is not None
                                and edge.left_face.inside == 1
                            )
                            and (
                                edge.right_face is not None
                                and edge.right_face.inside == 1
                            )
                        ):
                            reversed_edge: Edge2D = edge.shallow_duplicate_reverse()
                            boundary_grid_edges.add(reversed_edge)
                        if edge.seam_type == Seam.BOUNDARY:
                            if edge.is_cut():
                                reversed_edge: Edge2D = edge.shallow_duplicate_reverse()
                                boundary_grid_edges.add(reversed_edge)

        return boundary_grid_edges

    def get_used_grid_vertices(self) -> Set[Vertex2D]:
        used_faces: List[Face2D] = self.get_used_grid_faces()
        vertices: Set[Vertex2D] = set()
        for face in used_faces:
            for vertex in face.get_vertices():
                vertices.add(vertex)
        return vertices

    def get_used_grid_faces(self) -> List[Face2D]:
        return [face for face in self.faces if face.inside > 0]

    def get_bbox(self) -> Tuple[int, int, int, int]:
        min_x: int = self.N
        max_x: int = 0
        min_y: int = self.N
        max_y: int = 0
        for x in range(self.N):
            for y in range(self.N):
                face: Face2D = self.grid_faces[x][y]
                if face.inside > 0:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        return min_x, max_x, min_y, max_y

    def add_template_piece(self, template_piece: TemplatePiece) -> None:
        self.template_pieces.append(template_piece)

    def clear_meshes(self) -> None:
        self.meshes.clear()

    @staticmethod
    def assign_piece_or_mesh_names(
        pieces: Union[List[Piece], List[Mesh2D]],
        garment_type: Optional[str],
        is_front: bool = True,
        strict_garment_type: bool = True,
    ) -> List[Tuple[str, Union[Piece, Mesh2D]]]:
        assert not strict_garment_type or garment_type in [
            None,
            "dress_sleeveless",
            "dress",
            "jumpsuit_sleeveless",
            "jumpsuit",
            "dress_sleeveless_centerseparated_skirtremoved",
            "dress_sleeveless_skirtremoved",
            "dress_centerseparated_skirtremoved",
            "dress_skirtremoved",
            "unmerged_dress",
            "unmerged_dress_sleeveless",
            "jumpsuit_centerseparated",
            "jumpsuit_sleeveless_centerseparated",
            "unmerged_dress_centerseparated",
            "one_genus_jumpsuit_sleeveless",
            "merged_jumpsuit_sleeveless",
        ]
        assert len(pieces) <= 6
        if len(pieces) == 1:
            assert not strict_garment_type or garment_type in [
                None,
                "dress_sleeveless",
                "dress_sleeveless_skirtremoved",
                "dress_skirtremoved",
                "merged_jumpsuit_sleeveless",
            ], f"garment_type: {garment_type}"
            if garment_type == "dress_sleeveless":
                if is_front:
                    return [("skirt_front_top_front", pieces[0])]
                else:
                    return [("skirt_back_top_back", pieces[0])]
            elif garment_type == "dress_sleeveless_skirtremoved":
                if is_front:
                    return [("top_front", pieces[0])]
                else:
                    return [("top_back", pieces[0])]
            elif garment_type == "dress_skirtremoved":
                if is_front:
                    return [("top_front", pieces[0])]
                else:
                    return [("top_back", pieces[0])]
            elif garment_type == "merged_jumpsuit_sleeveless":
                if is_front:
                    return [("top_front", pieces[0])]
                else:
                    return [("top_back", pieces[0])]
            elif garment_type is None:
                if is_front:
                    return [("top_front", pieces[0])]
                else:
                    return [("top_back", pieces[0])]

        elif len(pieces) == 2:
            if garment_type == "unmerged_dress_sleeveless":
                mean_ys = [piece.mean_y() for piece in pieces]
                top_idx = mean_ys.index(max(mean_ys))
                bottom_idx = mean_ys.index(min(mean_ys))
                top_piece = pieces[top_idx]
                bottom_piece = pieces[bottom_idx]
                if is_front:
                    return [
                        ("top_front", top_piece),
                        ("skirt_front", bottom_piece),
                    ]
                else:
                    return [
                        ("top_back", top_piece),
                        ("skirt_back", bottom_piece),
                    ]
            elif garment_type == "dress_sleeveless_centerseparated_skirtremoved":
                mean_xs = [piece.mean_x() for piece in pieces]
                left_idx = mean_xs.index(min(mean_xs))
                right_idx = mean_xs.index(max(mean_xs))
                left_piece = pieces[left_idx]
                right_piece = pieces[right_idx]
                if is_front:
                    return [
                        ("top_front_right", left_piece),
                        ("top_front_left", right_piece),
                    ]
                else:
                    return [
                        ("top_back_right", left_piece),
                        ("top_back_left", right_piece),
                    ]

            elif garment_type == "dress":
                # assume failure case of dress
                mean_xs = [piece.mean_x() for piece in pieces]
                left_idx = mean_xs.index(min(mean_xs))
                right_idx = mean_xs.index(max(mean_xs))
                left_piece = pieces[left_idx]
                right_piece = pieces[right_idx]
                left_piece_x_width = left_piece.max_x() - left_piece.min_x()
                right_piece_x_width = right_piece.max_x() - right_piece.min_x()
                if left_piece_x_width > right_piece_x_width:
                    body_piece = left_piece
                    left_piece = None
                else:
                    body_piece = right_piece
                    right_piece = None
                if is_front:
                    return [
                        ("skirt_front_top_front", body_piece),
                        ("rfsleeve", left_piece)
                        if left_piece is not None
                        else ("lfsleeve", right_piece),
                    ]
                else:
                    return [
                        ("skirt_back_top_back", body_piece),
                        ("rbsleeve", left_piece)
                        if left_piece is not None
                        else ("lbsleeve", right_piece),
                    ]

            elif garment_type == "jumpsuit_sleeveless":
                # assume failure case of jumpsuit_sleeveless
                # mean_ys = [piece.mean_y() for piece in pieces]
                # mean_xs = [piece.mean_x() for piece in pieces]
                # left_idx = mean_xs.index(min(mean_xs))
                # right_idx = mean_xs.index(max(mean_xs))
                # left_piece = pieces[left_idx]
                # right_piece = pieces[right_idx]
                # left_piece_x_width = left_piece.max_x() - left_piece.min_x()
                # right_piece_x_width = right_piece.max_x() - right_piece.min_x()
                print("len(pieces) == 2")
                raise NotImplementedError("len(pieces) == 2")
            elif garment_type == "dress_skirtremoved":
                mean_xs = [piece.mean_x() for piece in pieces]
                left_idx = mean_xs.index(min(mean_xs))
                right_idx = mean_xs.index(max(mean_xs))
                if is_front:
                    return [
                        ("top_front", pieces[left_idx]),
                        ("lfsleeve", pieces[right_idx]),
                    ]
                else:
                    return [
                        ("top_back", pieces[left_idx]),
                        ("lbsleeve", pieces[right_idx]),
                    ]
            elif garment_type is None:
                # assume the garment is center-separated shirt
                mean_xs = [piece.mean_x() for piece in pieces]
                left_idx = mean_xs.index(min(mean_xs))
                right_idx = mean_xs.index(max(mean_xs))
                left_piece = pieces[left_idx]
                right_piece = pieces[right_idx]
                if is_front:
                    return [
                        ("top_front_right", left_piece),
                        ("top_front_left", right_piece),
                    ]
                else:
                    return [
                        ("top_back_right", left_piece),
                        ("top_back_left", right_piece),
                    ]

                """
                # assume the garment is dress_sleeveless_unmerged
                mean_ys = [piece.mean_y() for piece in pieces]
                top_idx = mean_ys.index(min(mean_ys))
                bottom_idx = mean_ys.index(max(mean_ys))
                top_piece = pieces[top_idx]
                bottom_piece = pieces[bottom_idx]
                if is_front:
                    return [
                        ("top_front", top_piece),
                        ("skirt_front", bottom_piece),
                    ]
                else:
                    return [
                        ("top_back", top_piece),
                        ("skirt_back", bottom_piece),
                    ]
                """
            else:
                raise NotImplementedError("len(pieces) == 2")

        elif len(pieces) == 3:
            assert not strict_garment_type or garment_type in [
                None,
                "dress",
                "jumpsuit_sleeveless",
                "dress_skirtremoved",
                "dress_centerseparated_skirtremoved",
            ], f"garment_type: {garment_type}"
            garment_type = "dress" if garment_type is None else garment_type
            if garment_type == "dress_centerseparated_skirtremoved":
                mean_xs = [piece.mean_x() for piece in pieces]
                left_sleeve_idx = mean_xs.index(min(mean_xs))
                right_sleeve_idx = mean_xs.index(max(mean_xs))
                body_idx = 3 - left_sleeve_idx - right_sleeve_idx
                if is_front:
                    return [
                        ("top_front_left", pieces[body_idx]),
                        ("rfsleeve", pieces[left_sleeve_idx]),
                        ("lfsleeve", pieces[right_sleeve_idx]),
                    ]
                else:
                    return [
                        ("top_back_left", pieces[body_idx]),
                        ("rbsleeve", pieces[left_sleeve_idx]),
                        ("lbsleeve", pieces[right_sleeve_idx]),
                    ]

            elif garment_type == "dress_skirtremoved":
                mean_xs = [piece.mean_x() for piece in pieces]
                left_sleeve_idx = mean_xs.index(min(mean_xs))
                right_sleeve_idx = mean_xs.index(max(mean_xs))
                body_idx = 3 - left_sleeve_idx - right_sleeve_idx
                if is_front:
                    return [
                        ("top_front", pieces[body_idx]),
                        ("rfsleeve", pieces[left_sleeve_idx]),
                        ("lfsleeve", pieces[right_sleeve_idx]),
                    ]
                else:
                    return [
                        ("top_back", pieces[body_idx]),
                        ("rbsleeve", pieces[left_sleeve_idx]),
                        ("lbsleeve", pieces[right_sleeve_idx]),
                    ]

            elif garment_type == "dress" or garment_type is None:
                # body, left_sleeve, right_sleeve
                mean_xs = [piece.mean_x() for piece in pieces]
                left_sleeve_idx = mean_xs.index(min(mean_xs))
                right_sleeve_idx = mean_xs.index(max(mean_xs))
                if left_sleeve_idx == right_sleeve_idx:
                    print("WARNING: left_sleeve and right_sleeve are the same")
                    left_sleeve_idx = (left_sleeve_idx + 1) % 3
                body_idx = 3 - left_sleeve_idx - right_sleeve_idx
                if is_front:
                    return [
                        ("skirt_front_top_front", pieces[body_idx]),
                        ("rfsleeve", pieces[left_sleeve_idx]),
                        ("lfsleeve", pieces[right_sleeve_idx]),
                    ]
                else:
                    return [
                        ("skirt_back_top_back", pieces[body_idx]),
                        ("rbsleeve", pieces[left_sleeve_idx]),
                        ("lbsleeve", pieces[right_sleeve_idx]),
                    ]
            elif garment_type == "jumpsuit":
                # assume failure case of jumpsuit
                min_ys = [piece.min_y() for piece in pieces]
                num_piece_bottom_150 = sum([min_y < 150 for min_y in min_ys])
                if num_piece_bottom_150 == 1:
                    # assume pants are not separated, body, pants, sleeve
                    pants_idx = min_ys.index(min(min_ys))
                    pants_piece = pieces[pants_idx]
                    unused_pieces = [
                        pieces[i] for i in range(3) if i not in [pants_idx]
                    ]
                    min_xs = [piece.min_x() for piece in unused_pieces]
                    left_idx = min_xs.index(min(min_xs))
                    right_idx = min_xs.index(max(min_xs))
                    left_piece = unused_pieces[left_idx]
                    right_piece = unused_pieces[right_idx]
                    assert left_idx != right_idx, (
                        f"left_idx: {left_idx}, right_idx: {right_idx}"
                    )
                    left_width = (
                        unused_pieces[left_idx].max_x()
                        - unused_pieces[left_idx].min_x()
                    )
                    right_width = (
                        unused_pieces[right_idx].max_x()
                        - unused_pieces[right_idx].min_x()
                    )
                    if left_width > right_width:
                        body_piece = left_piece
                        left_piece = None
                    else:
                        body_piece = right_piece
                        right_piece = None
                    if is_front:
                        return [
                            ("lfsleeve", right_piece)
                            if right_piece is not None
                            else ("rfsleeve", left_piece),
                            ("up_front", body_piece),
                            ("Rfront", pants_piece),
                        ]
                    else:
                        return [
                            ("lbsleeve", right_piece)
                            if right_piece is not None
                            else ("rbsleeve", left_piece),
                            ("up_back", body_piece),
                            ("Rback", pants_piece),
                        ]
                elif num_piece_bottom_150 == 2:
                    # assume pants are separated, body, left_pant, right_pant
                    body_idx = min_ys.index(max(min_ys))
                    body_piece = pieces[body_idx]
                    unused_pieces = [pieces[i] for i in range(3) if i not in [body_idx]]
                    min_xs = [piece.min_x() for piece in unused_pieces]
                    left_pant_idx = min_xs.index(min(min_xs))
                    right_pant_idx = 1 - left_pant_idx
                    left_bottom_piece = unused_pieces[left_pant_idx]
                    right_bottom_piece = unused_pieces[right_pant_idx]

                    if is_front:
                        return [
                            ("up_front", body_piece),
                            ("Rfront", left_bottom_piece),
                            ("Lfront", right_bottom_piece),
                        ]
                    else:
                        return [
                            ("up_back", body_piece),
                            ("Rback", left_bottom_piece),
                            ("Lback", right_bottom_piece),
                        ]

                else:
                    raise ValueError("num_piece_bottom_150 != 1 or 2")

            else:
                # assume garment_type is jumpsuit_sleeveless
                # left bottom, right bottom, body
                max_ys = [piece.max_y() for piece in pieces]
                body_idx = max_ys.index(max(max_ys))
                left_body_idx = (body_idx + 1) % 3
                right_body_idx = (body_idx + 2) % 3
                left_body_min_x = pieces[left_body_idx].min_x()
                right_body_min_x = pieces[right_body_idx].min_x()
                if right_body_min_x < left_body_min_x:
                    left_body_idx, right_body_idx = right_body_idx, left_body_idx
                if is_front:
                    return [
                        ("Rfront", pieces[left_body_idx]),
                        ("Lfront", pieces[right_body_idx]),
                        ("up_front", pieces[body_idx]),
                    ]
                else:
                    return [
                        ("Rback", pieces[left_body_idx]),
                        ("Lback", pieces[right_body_idx]),
                        ("up_back", pieces[body_idx]),
                    ]
        elif len(pieces) == 4:
            if garment_type == "unmerged_dress":
                min_ys = [piece.min_y() for piece in pieces]
                skirt_index = min_ys.index(min(min_ys))
                skirt_piece = pieces[skirt_index]
                unused_pieces = [pieces[i] for i in range(4) if i not in [skirt_index]]
                mean_xs = [piece.mean_x() for piece in unused_pieces]
                left_sleeve_idx = mean_xs.index(min(mean_xs))
                right_sleeve_idx = mean_xs.index(max(mean_xs))
                top_idx = 3 - left_sleeve_idx - right_sleeve_idx
                left_sleeve_piece = unused_pieces[left_sleeve_idx]
                right_sleeve_piece = unused_pieces[right_sleeve_idx]
                top_piece = unused_pieces[top_idx]
                if is_front:
                    return [
                        ("skirt_front", skirt_piece),
                        ("rfsleeve", left_sleeve_piece),
                        ("lfsleeve", right_sleeve_piece),
                        ("top_front", top_piece),
                    ]
                else:
                    return [
                        ("skirt_back", skirt_piece),
                        ("rbsleeve", left_sleeve_piece),
                        ("lbsleeve", right_sleeve_piece),
                        ("top_back", top_piece),
                    ]

            if garment_type == "dress_centerseparated_skirtremoved":
                mean_xs = [piece.mean_x() for piece in pieces]
                left_sleeve_idx = mean_xs.index(min(mean_xs))
                right_sleeve_idx = mean_xs.index(max(mean_xs))
                left_sleeve_piece = pieces[left_sleeve_idx]
                right_sleeve_piece = pieces[right_sleeve_idx]
                unused_pieces = [
                    pieces[i]
                    for i in range(4)
                    if i not in [left_sleeve_idx, right_sleeve_idx]
                ]
                mean_xs = [piece.mean_x() for piece in unused_pieces]
                left_body_idx = mean_xs.index(min(mean_xs))
                right_body_idx = mean_xs.index(max(mean_xs))
                left_body_piece = unused_pieces[left_body_idx]
                right_body_piece = unused_pieces[right_body_idx]
                if is_front:
                    return [
                        ("top_front_left", right_body_piece),
                        ("top_front_right", left_body_piece),
                        ("rfsleeve", left_sleeve_piece),
                        ("lfsleeve", right_sleeve_piece),
                    ]
                else:
                    return [
                        ("top_back_left", right_body_piece),
                        ("top_back_right", left_body_piece),
                        ("rbsleeve", left_sleeve_piece),
                        ("lbsleeve", right_sleeve_piece),
                    ]
            if garment_type == "jumpsuit_sleeveless_centerseparated":
                mean_ys = [piece.mean_y() for piece in pieces]
                bottom_idx_1 = mean_ys.index(min(mean_ys))
                bottom_piece_1 = pieces[bottom_idx_1]
                unused_pieces = [pieces[i] for i in range(4) if i != bottom_idx_1]
                mean_ys = [piece.mean_y() for piece in unused_pieces]
                bottom_idx_2 = mean_ys.index(min(mean_ys))
                bottom_piece_2 = unused_pieces[bottom_idx_2]
                if bottom_piece_1.mean_x() < bottom_piece_2.mean_x():
                    left_bottom_piece = bottom_piece_1
                    right_bottom_piece = bottom_piece_2
                else:
                    left_bottom_piece = bottom_piece_2
                    right_bottom_piece = bottom_piece_1

                unused_pieces = [
                    unused_pieces[i] for i in range(3) if i != bottom_idx_2
                ]
                mean_x = [piece.mean_x() for piece in unused_pieces]
                left_top_idx = mean_x.index(min(mean_x))
                right_top_idx = mean_x.index(max(mean_x))
                left_top_piece = unused_pieces[left_top_idx]
                right_top_piece = unused_pieces[right_top_idx]
                if is_front:
                    return [
                        ("top_front_left", right_top_piece),
                        ("top_front_right", left_top_piece),
                        ("Rfront", left_bottom_piece),
                        ("Lfront", right_bottom_piece),
                    ]
                else:
                    return [
                        ("top_back_left", right_top_piece),
                        ("top_back_right", left_top_piece),
                        ("Rback", left_bottom_piece),
                        ("Lback", right_bottom_piece),
                    ]

            if garment_type == "one_genus_jumpsuit_sleeveless":
                max_ys = [piece.max_y() for piece in pieces]
                body_idx = max_ys.index(max(max_ys))
                body_piece = pieces[body_idx]

                unused_pieces = [pieces[i] for i in range(4) if i != body_idx]
                mean_ys = [piece.mean_y() for piece in unused_pieces]
                small_bottom_idx = mean_ys.index(min(mean_ys))
                small_bottom_piece = unused_pieces[small_bottom_idx]

                unused_pieces = [
                    unused_pieces[i] for i in range(3) if i != small_bottom_idx
                ]
                mean_xs = [piece.mean_x() for piece in unused_pieces]
                left_bottom_idx = mean_xs.index(min(mean_xs))
                right_bottom_idx = mean_xs.index(max(mean_xs))
                left_bottom_piece = unused_pieces[left_bottom_idx]
                right_bottom_piece = unused_pieces[right_bottom_idx]

                if is_front:
                    return [
                        ("Rfrontsmall", small_bottom_piece),
                        ("Rfront", left_bottom_piece),
                        ("Lfront", right_bottom_piece),
                        ("up_front", body_piece),
                    ]
                else:
                    return [
                        ("Rbacksmall", small_bottom_piece),
                        ("Rback", left_bottom_piece),
                        ("Lback", right_bottom_piece),
                        ("up_back", body_piece),
                    ]

            else:
                # assume failure case of jumpsuit
                mean_xs = [piece.mean_x() for piece in pieces]
                left_sleeve_idx = mean_xs.index(min(mean_xs))
                right_sleeve_idx = mean_xs.index(max(mean_xs))
                if left_sleeve_idx == right_sleeve_idx:
                    raise ValueError("left_sleeve_idx == right_sleeve_idx")
                left_sleeve_piece = pieces[left_sleeve_idx]
                right_sleeve_piece = pieces[right_sleeve_idx]

                unused_pieces = [
                    pieces[i]
                    for i in range(4)
                    if i not in [left_sleeve_idx, right_sleeve_idx]
                ]

                min_ys = [piece.min_y() for piece in unused_pieces]
                num_piece_bottom_150 = sum([min_y < 150 for min_y in min_ys])
                if num_piece_bottom_150 == 1:
                    # assume pants are not separated
                    max_ys = [piece.max_y() for piece in unused_pieces]
                    body_idx = max_ys.index(max(max_ys))
                    body_piece = unused_pieces[body_idx]
                    pants_idx = min_ys.index(min(min_ys))
                    pants_piece = unused_pieces[pants_idx]
                    if is_front:
                        return [
                            ("rfsleeve", left_sleeve_piece),
                            ("lfsleeve", right_sleeve_piece),
                            ("up_front", body_piece),
                            ("Rfront", pants_piece),
                        ]
                    else:
                        return [
                            ("rbsleeve", left_sleeve_piece),
                            ("lbsleeve", right_sleeve_piece),
                            ("up_back", body_piece),
                            ("Rback", pants_piece),
                        ]

                elif num_piece_bottom_150 == 2:
                    # assume pants are separated
                    min_xs = [piece.min_x() for piece in unused_pieces]
                    left_pant_idx = min_xs.index(min(min_xs))
                    right_pant_idx = 1 - left_pant_idx
                    left_bottom_piece = unused_pieces[left_pant_idx]
                    right_bottom_piece = unused_pieces[right_pant_idx]

                    # decide which is body among left_sleeve_piece and right_sleeve_piece
                    left_piece_x_width = (
                        left_sleeve_piece.max_x() - left_sleeve_piece.min_x()
                    )
                    right_piece_x_width = (
                        right_sleeve_piece.max_x() - right_sleeve_piece.min_x()
                    )
                    if left_piece_x_width > right_piece_x_width:
                        body_piece = left_sleeve_piece
                        left_sleeve_piece = None
                    else:
                        body_piece = right_sleeve_piece
                        right_sleeve_piece = None

                    if is_front:
                        return [
                            ("lfsleeve", right_sleeve_piece)
                            if right_sleeve_piece is not None
                            else ("rfsleeve", left_sleeve_piece),
                            ("up_front", body_piece),
                            ("Rfront", left_bottom_piece),
                            ("Lfront", right_bottom_piece),
                        ]
                    else:
                        return [
                            ("lbsleeve", right_sleeve_piece)
                            if right_sleeve_piece is not None
                            else ("rbsleeve", left_sleeve_piece),
                            ("up_back", body_piece),
                            ("Rback", left_bottom_piece),
                            ("Lback", right_bottom_piece),
                        ]
                else:
                    raise ValueError("num_piece_bottom_150 != 1 or 2")

        elif len(pieces) == 5:
            assert not strict_garment_type or garment_type in [
                None,
                "jumpsuit",
                "unmerged_dress_centerseparated",
            ], f"garment_type: {garment_type}"
            if garment_type == "unmerged_dress_centerseparated":
                mean_ys = [piece.mean_y() for piece in pieces]
                mean_xs = [piece.mean_x() for piece in pieces]
                skirt_idx = mean_ys.index(min(mean_ys))
                skirt_piece = pieces[skirt_idx]
                left_sleeve_idx = mean_xs.index(min(mean_xs))
                right_sleeve_idx = mean_xs.index(max(mean_xs))
                left_sleeve_piece = pieces[left_sleeve_idx]
                right_sleeve_piece = pieces[right_sleeve_idx]
                unused_pieces = [
                    pieces[i]
                    for i in range(5)
                    if i not in [skirt_idx, left_sleeve_idx, right_sleeve_idx]
                ]
                mean_xs = [piece.mean_x() for piece in unused_pieces]
                left_top_idx = mean_xs.index(min(mean_xs))
                right_top_idx = mean_xs.index(max(mean_xs))
                left_top_piece = unused_pieces[left_top_idx]
                right_top_piece = unused_pieces[right_top_idx]
                if is_front:
                    return [
                        ("rfsleeve", left_sleeve_piece),
                        ("lfsleeve", right_sleeve_piece),
                        ("top_front_left", right_top_piece),
                        ("top_front_right", left_top_piece),
                        ("skirt_front", skirt_piece),
                    ]
                else:
                    return [
                        ("rbsleeve", left_sleeve_piece),
                        ("lbsleeve", right_sleeve_piece),
                        ("top_back_left", right_top_piece),
                        ("top_back_right", left_top_piece),
                        ("skirt_back", skirt_piece),
                    ]

            elif garment_type is None or garment_type == "jumpsuit":
                # assume garment_type is jumpsuit
                min_xs = [piece.min_x() for piece in pieces]
                max_xs = [piece.max_x() for piece in pieces]
                left_sleeve_idx = min_xs.index(min(min_xs))
                right_sleeve_idx = max_xs.index(max(max_xs))
                if left_sleeve_idx == right_sleeve_idx:
                    raise ValueError("left_sleeve_idx == right_sleeve_idx")
                left_sleeve_piece = pieces[left_sleeve_idx]
                right_sleeve_piece = pieces[right_sleeve_idx]

                unused_pieces = [
                    pieces[i]
                    for i in range(5)
                    if i not in [left_sleeve_idx, right_sleeve_idx]
                ]
                max_ys = [piece.max_y() for piece in unused_pieces]
                body_idx = max_ys.index(max(max_ys))
                body_piece = unused_pieces[body_idx]

                unused_pieces.remove(unused_pieces[body_idx])
                min_xs = [piece.min_x() for piece in unused_pieces]
                left_bottom_idx = min_xs.index(min(min_xs))
                right_bottom_idx = 1 - left_bottom_idx
                if left_bottom_idx == right_bottom_idx:
                    raise ValueError("left_bottom_idx == right_bottom_idx")
                left_bottom_piece = unused_pieces[left_bottom_idx]
                right_bottom_piece = unused_pieces[right_bottom_idx]

                if is_front:
                    return [
                        ("rfsleeve", left_sleeve_piece),
                        ("lfsleeve", right_sleeve_piece),
                        ("up_front", body_piece),
                        ("Rfront", left_bottom_piece),
                        ("Lfront", right_bottom_piece),
                    ]
                else:
                    return [
                        ("rbsleeve", left_sleeve_piece),
                        ("lbsleeve", right_sleeve_piece),
                        ("up_back", body_piece),
                        ("Rback", left_bottom_piece),
                        ("Lback", right_bottom_piece),
                    ]
        elif len(pieces) == 6:
            assert not strict_garment_type or garment_type in [
                None,
                "jumpsuit_centerseparated",
            ], f"garment_type: {garment_type}"
            mean_xs = [piece.mean_x() for piece in pieces]
            left_sleeve_idx = mean_xs.index(min(mean_xs))
            right_sleeve_idx = mean_xs.index(max(mean_xs))
            if left_sleeve_idx == right_sleeve_idx:
                raise ValueError("left_sleeve_idx == right_sleeve_idx")
            left_sleeve_piece = pieces[left_sleeve_idx]
            right_sleeve_piece = pieces[right_sleeve_idx]

            unused_pieces = [
                pieces[i]
                for i in range(6)
                if i not in [left_sleeve_idx, right_sleeve_idx]
            ]
            mean_ys = [piece.mean_y() for piece in unused_pieces]
            body_idx_1 = mean_ys.index(max(mean_ys))
            body_idx_2 = mean_ys.index(
                max([mean_y for i, mean_y in enumerate(mean_ys) if i != body_idx_1])
            )
            body_piece_1 = unused_pieces[body_idx_1]
            body_piece_2 = unused_pieces[body_idx_2]
            if body_piece_1.mean_x() < body_piece_2.mean_x():
                left_top_piece = body_piece_1
                right_top_piece = body_piece_2
            else:
                left_top_piece = body_piece_2
                right_top_piece = body_piece_1

            unused_pieces = [
                unused_pieces[i] for i in range(4) if i not in [body_idx_1, body_idx_2]
            ]
            mean_xs = [piece.mean_x() for piece in unused_pieces]
            left_bottom_idx = mean_xs.index(min(mean_xs))
            right_bottom_idx = mean_xs.index(max(mean_xs))
            left_bottom_piece = unused_pieces[left_bottom_idx]
            right_bottom_piece = unused_pieces[right_bottom_idx]

            if is_front:
                return [
                    ("rfsleeve", left_sleeve_piece),
                    ("lfsleeve", right_sleeve_piece),
                    ("top_front_left", right_top_piece),
                    ("top_front_right", left_top_piece),
                    ("Rfront", left_bottom_piece),
                    ("Lfront", right_bottom_piece),
                ]
            else:
                return [
                    ("rbsleeve", left_sleeve_piece),
                    ("lbsleeve", right_sleeve_piece),
                    ("top_back_left", right_top_piece),
                    ("top_back_right", left_top_piece),
                    ("Rback", left_bottom_piece),
                    ("Lback", right_bottom_piece),
                ]

    def visualize_pieces(
        self,
        use_points: bool = True,
        show_markers: bool = True,
        output_file_path: Optional[str] = None,
    ) -> None:
        pieces: List[Piece] = [
            template_piece.piece for template_piece in self.template_pieces
        ]
        piece_to_constraints: Dict[Piece, Dict[Vertex2D, Vertex2D]] = {
            template_piece.piece: template_piece.original_constraints
            for template_piece in self.template_pieces
        }
        Piece.visualize_pieces(
            pieces,
            piece_to_constraints,
            use_points=use_points,
            show_markers=show_markers,
            output_file_path=output_file_path,
        )

    def visualize_meshes(
        self,
        output_file_path: Optional[str] = None,
        show_original_meshes: bool = False,
        axis_off: bool = True,
        show_constraints: bool = False,
        boundary_paint: bool = False,
        boundary_paint_dpi: Optional[int] = None,
        edge_color: Optional[str] = None,
        fixed_min_x: Optional[int] = None,
        fixed_max_x: Optional[int] = None,
        fixed_min_y: Optional[int] = None,
        fixed_max_y: Optional[int] = None,
        markersize: int = 5,
    ) -> None:
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
        if boundary_paint:
            if boundary_paint_dpi is None:
                boundary_paint_dpi = GARMENT_IMAGE_RESOLUTION
            fig, ax = plt.subplots(figsize=(1, 1), dpi=boundary_paint_dpi)
        else:
            fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        if axis_off:
            ax.axis("off")
        if show_original_meshes:
            meshes = self.original_meshes
        else:
            meshes = self.meshes

        constraint_marker = "X"

        # show the edges in the meshes
        for mesh in meshes:
            if boundary_paint:
                # Use the boundary of the mesh, not faces
                boundary_vertices: List[Vertex2D] = mesh.get_boundary_vertices()
                boundary_np_array: np.ndarray = np.array(
                    [[v.x, v.y] for v in boundary_vertices]
                )
                # Create a polygon from the vertices and add it to the plot
                polygon = plt.Polygon(
                    boundary_np_array,
                    closed=True,
                    fill=True,
                    edgecolor=edge_color,
                    facecolor="lightgreen",
                )
                ax.add_patch(polygon)
                min_x = min(min_x, boundary_np_array[:, 0].min())
                max_x = max(max_x, boundary_np_array[:, 0].max())
                min_y = min(min_y, boundary_np_array[:, 1].min())
                max_y = max(max_y, boundary_np_array[:, 1].max())
            else:
                for edge in mesh.edges:
                    if edge.seam_type is None:
                        linestyle = Seam.boundary_types_to_linestyle[Seam.NONE]
                        linecolor = Seam.boundary_types_to_color[Seam.NONE]
                    else:
                        linestyle = Seam.boundary_types_to_linestyle[edge.seam_type]
                        linecolor = Seam.boundary_types_to_color[edge.seam_type]
                    ax.plot(
                        [edge.start.x, edge.end.x],
                        [edge.start.y, edge.end.y],
                        marker=None,
                        linestyle=linestyle,
                        color=linecolor,
                        alpha=1.0,
                        markersize=markersize,
                    )
                    min_x = min(min_x, edge.start.x, edge.end.x)
                    max_x = max(max_x, edge.start.x, edge.end.x)
                    min_y = min(min_y, edge.start.y, edge.end.y)
                    max_y = max(max_y, edge.start.y, edge.end.y)

        min_x = fixed_min_x if fixed_min_x is not None else min_x
        max_x = fixed_max_x if fixed_max_x is not None else max_x
        min_y = fixed_min_y if fixed_min_y is not None else min_y
        max_y = fixed_max_y if fixed_max_y is not None else max_y

        # show the constraints
        if not boundary_paint and show_constraints:
            for template_piece in self.template_pieces:
                if show_original_meshes:
                    constraints: Dict[Vertex2D, Vertex2D] = (
                        template_piece.original_constraints
                    )
                else:
                    constraints: Dict[Vertex2D, Vertex2D] = template_piece.constraints

                for point in constraints.keys():
                    ax.plot(
                        point.x,
                        point.y,
                        marker=constraint_marker,
                        linestyle="-",
                        color=Seam.constraint_color,
                        alpha=1.0,
                        markersize=7,
                    )

        ax.set_xticks(
            range(int(min_x), int(max_x)),
            minor=True,
        )
        ax.set_yticks(
            range(int(min_y), int(max_y)),
            minor=True,
        )
        if axis_off:
            ax.axis("off")
        if output_file_path is not None:
            dir_path = os.path.dirname(output_file_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(output_file_path, bbox_inches="tight")
            fig.savefig(output_file_path.replace(".png", ".svg"), format="svg")
        else:
            plt.show()
        # close the figure to prevent memory leak
        plt.close(fig)

    def visualize_faces(
        self,
        output_file_path: Optional[str] = None,
        markersize: int = 5,
        axis_off: bool = True,
    ) -> None:
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")

        used_grid_vertices: Set[Vertex2D] = self.get_used_grid_vertices()

        for v in used_grid_vertices:
            top_edge: Edge2D = v.get_top_edge()
            right_edge: Edge2D = v.get_right_edge()
            # y_top = y + self.W / self.N
            # x_left = x - self.W / self.N
            # top_edge: Edge2D = Edge2D(Vertex2D(x, y), Vertex2D(x, y_top))
            # left_edge: Edge2D = Edge2D(Vertex2D(x, y), Vertex2D(x_left, y))
            for edge in [top_edge, right_edge]:
                if (
                    edge is not None
                    and edge.start in used_grid_vertices
                    and edge.end in used_grid_vertices
                ):
                    seam_type = edge.seam_type
                    if seam_type == Seam.NONE:
                        if edge.right_face is None or edge.left_face is None:
                            continue
                        if edge.right_face.inside <= 0 and edge.left_face.inside <= 0:
                            continue
                    # if seam_type is None:
                    #     seam_type = (
                    #         v.X_EDGE_TYPE if edge == right_edge else v.Y_EDGE_TYPE
                    #     )
                    if seam_type is not None and (
                        (edge.right_face is not None and edge.right_face.inside == 1)
                        or (edge.left_face is not None and edge.left_face.inside == 1)
                    ):
                        linestyle = Seam.boundary_types_to_linestyle[seam_type]
                        linecolor = Seam.boundary_types_to_color[seam_type]
                        ax.plot(
                            [edge.start.x, edge.end.x],
                            [edge.start.y, edge.end.y],
                            marker=None,
                            linestyle=linestyle,
                            color=linecolor,
                            alpha=1.0,
                            markersize=markersize,
                        )

        ax.set_xticks(
            range(0, self.W + 1),
            minor=True,
        )
        ax.set_yticks(
            range(0, self.W + 1),
            minor=True,
        )
        if axis_off:
            ax.axis("off")
        if output_file_path is not None:
            dir_path = os.path.dirname(output_file_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(output_file_path)
            fig.savefig(output_file_path.replace(".png", ".svg"), format="svg")
        else:
            plt.show()
        # close the figure to prevent memory leak
        plt.close(fig)

    def clear(self):
        for x in range(self.N):
            for y in range(self.N):
                face: Face2D = self.grid_faces[x][y]
                face.inside = 0
                face.H_EDGE0 = Vector2(self.W / self.N, 0)
                face.H_EDGE1 = Vector2(self.W / self.N, 0)
                face.V_EDGE0 = Vector2(0, self.W / self.N)
                face.V_EDGE1 = Vector2(0, self.W / self.N)
                v: Vertex2D = self.grid_vertices[x][y]
                v.X_EDGE_TYPE = Seam.NONE
                v.Y_EDGE_TYPE = Seam.NONE

    def adjacent_to_plus(self, x: int, y: int) -> bool:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                face: Face2D = self.get_grid_face(x + dx, y + dy)
                if face is not None and face.inside > 0:
                    return True
        return False

    def adjacent_to_minus(self, x: int, y: int) -> bool:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                face: Face2D = self.get_grid_face(x + dx, y + dy)
                if face is not None and face.inside <= 0:
                    return True
        return False

    def get_maximum_nearby(self, x: int, y: int) -> int:
        max: int = self.grid_faces[x][y].inside
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                face: Face2D = self.get_grid_face(x + dx, y + dy)
                if face is not None and face.inside <= 0:
                    max = max(max, face.inside - 1)
        return max

    def get_minimum_nearby(self, x: int, y: int) -> int:
        min: int = self.grid_faces[x][y].inside
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                face: Face2D = self.get_grid_face(x + dx, y + dy)
                if face is not None and face.inside > 0:
                    min = min(min, face.inside + 1)
        return min

    def get_grid_face(self, x: int, y: int) -> Face2D:
        if x < 0 or x >= self.N or y < 0 or y >= self.N:
            return None
        return self.grid_faces[x][y]

    def find_nearest_vertex(self, v: Vertex2D) -> Vertex2D:
        return super().find_nearest_vertex(v)

    @classmethod
    def get_grid_xy(cls, v: Vertex2D) -> List[int]:
        """
        Get the grid coordinates [x, y] of a given Vertex2D object.
        """
        D: int = cls.W // cls.N
        x: int = round(v.x / D)
        y: int = round(v.y / D)
        return [x, y]

    def convert(self, v: Vertex2D) -> Vertex2D:
        return Vertex2D(v.x * self.W + self.X0, v.y * self.W + self.Y0)

    @classmethod
    def convert_relative_to_absolute(cls, relative: Vertex2D) -> Vertex2D:
        return Vertex2D(
            round(cls.N * relative.x) * (cls.W / cls.N) + cls.X0,
            round(cls.N * relative.y) * (cls.W / cls.N) + cls.Y0,
        )

    @staticmethod
    def get_path_v2(
        start: Vertex2D,
        end: Vertex2D,
        seam_type: Optional[int] = None,
        template: Optional[Template] = None,
        prev_seam_type: Optional[int] = None,
        prev_direction: Optional[Template.Direction] = None,
        next_seam_type: Optional[int] = None,
        next_direction: Optional[Template.Direction] = None,
    ) -> List[Vertex2D]:
        """
        Assume start and end are on the template grids
        """
        x_diff = end.x - start.x
        y_diff = end.y - start.y
        x_direction = (
            Template.Direction.RIGHT if x_diff > 0 else Template.Direction.LEFT
        )
        if x_diff == 0:
            x_direction = Template.Direction.NONE
        y_direction = Template.Direction.UP if y_diff > 0 else Template.Direction.DOWN
        if y_diff == 0:
            y_direction = Template.Direction.NONE
        assert (
            x_direction is not Template.Direction.NONE
            or y_direction is not Template.Direction.NONE
        ), f"x_direction: {x_direction}, y_direction: {y_direction}"

        initial_direction: Optional[Template.Direction] = None
        final_direction: Optional[Template.Direction] = None
        current_vertex: Vertex2D = start
        paths: List[Vertex2D] = [current_vertex]

        if x_direction == Template.Direction.NONE:
            initial_direction = y_direction
        elif y_direction == Template.Direction.NONE:
            initial_direction = x_direction
        elif seam_type == Seam.SIDE_BY_SIDE and (
            x_direction == Template.Direction.LEFT
            and y_direction == Template.Direction.UP
        ):
            initial_direction = x_direction
        elif seam_type == Seam.SIDE_BY_SIDE and (
            x_direction == Template.Direction.RIGHT
            and y_direction == Template.Direction.UP
        ):
            initial_direction = x_direction
        elif seam_type == Seam.SIDE_BY_SIDE and (
            x_direction == Template.Direction.LEFT
            and y_direction == Template.Direction.DOWN
        ):
            initial_direction = y_direction
        elif seam_type == Seam.SIDE_BY_SIDE and (
            x_direction == Template.Direction.RIGHT
            and y_direction == Template.Direction.DOWN
        ):
            initial_direction = y_direction
        elif (
            x_direction == Template.Direction.LEFT
            and y_direction == Template.Direction.DOWN
        ):
            initial_direction = x_direction
        elif (
            x_direction == Template.Direction.RIGHT
            and y_direction == Template.Direction.DOWN
        ):
            initial_direction = y_direction
        elif (
            x_direction == Template.Direction.LEFT
            and y_direction == Template.Direction.UP
        ):
            initial_direction = y_direction
        elif (
            x_direction == Template.Direction.RIGHT
            and y_direction == Template.Direction.UP
        ):
            initial_direction = x_direction
        elif prev_direction == x_direction:
            initial_direction = x_direction
            final_direction = y_direction
        elif prev_direction == y_direction:
            initial_direction = y_direction
            final_direction = x_direction
        elif Template.Direction.is_opposite(prev_direction, x_direction):
            if (
                prev_seam_type is not None
                and seam_type is not None
                and (
                    prev_seam_type == Seam.SIDE_BY_SIDE
                    and seam_type == Seam.SIDE_BY_SIDE
                )
            ):
                initial_direction = x_direction
                final_direction = y_direction
            else:
                initial_direction = y_direction
                final_direction = x_direction
        elif Template.Direction.is_opposite(prev_direction, y_direction):
            if (
                prev_seam_type is not None
                and seam_type is not None
                and (
                    (
                        prev_seam_type == Seam.SIDE_BY_SIDE
                        and seam_type == Seam.SIDE_BY_SIDE
                    )
                    or (prev_seam_type == Seam.BOUNDARY and seam_type == Seam.BOUNDARY)
                )
            ):
                initial_direction = y_direction
                final_direction = x_direction
            else:
                initial_direction = x_direction
                final_direction = y_direction
        else:
            initial_direction = None
            final_direction = None

        if next_direction is not None:
            if Template.Direction.is_opposite(next_direction, final_direction):
                if not (
                    prev_seam_type is not None
                    and seam_type is not None
                    and (
                        prev_seam_type == Seam.SIDE_BY_SIDE
                        and seam_type == Seam.SIDE_BY_SIDE
                    )
                ):
                    initial_direction, final_direction = (
                        final_direction,
                        initial_direction,
                    )

        ordered_axes: List[str] = ["x", "y"]
        if (
            initial_direction is Template.Direction.LEFT
            or initial_direction is Template.Direction.RIGHT
        ):
            ordered_axes = ["x", "y"]
        elif (
            initial_direction is Template.Direction.UP
            or initial_direction is Template.Direction.DOWN
        ):
            ordered_axes = ["y", "x"]

        delta = int(Template.W / Template.N)
        for axis in ordered_axes:
            if axis == "x":
                while abs(x_diff) > 0:
                    if x_diff > 0:
                        next_vertex: Vertex2D = Vertex2D(
                            current_vertex.x + delta, current_vertex.y
                        )
                        x_diff -= delta
                    else:
                        next_vertex: Vertex2D = Vertex2D(
                            current_vertex.x - delta, current_vertex.y
                        )
                        x_diff += delta
                    paths.append(next_vertex)
                    current_vertex = next_vertex
            elif axis == "y":
                while abs(y_diff) > 0:
                    if y_diff > 0:
                        next_vertex: Vertex2D = Vertex2D(
                            current_vertex.x, current_vertex.y + delta
                        )
                        y_diff -= delta
                    else:
                        next_vertex: Vertex2D = Vertex2D(
                            current_vertex.x, current_vertex.y - delta
                        )
                        y_diff += delta
                    paths.append(next_vertex)
                    current_vertex = next_vertex
            else:
                raise ValueError(f"Invalid axis: {axis}")
        return paths

    # TODO: address not only the straight line but also the curved line
    @staticmethod
    def get_path(
        mesh: Mesh2D,
        _start: Vertex2D,
        _end: Vertex2D,
        boundary_only: bool,
        is_reversed: bool,
        seam_type: Optional[int] = None,
        debug: bool = False,
        template: Optional[Template] = None,
    ) -> List[Vertex2D]:
        """
        Calculate a path between two points on a 2D mesh
        """
        start: Vertex2D = mesh.find_nearest_vertex(_start)
        end: Vertex2D = mesh.find_nearest_vertex(_end)

        v0: List[int] = Template.get_grid_xy(start)
        v1: List[int] = Template.get_grid_xy(end)
        if debug:
            pass

        # By the following if statement, avoid the invalid bottomo grid edge. See https://drive.google.com/file/d/1VRy9lPBBPyDtkR0JkW7wk_Sj8y8BYvJR/view?usp=drive_link
        if v1[1] > v0[1]:
            return Template.get_path_main(
                mesh,
                start,
                end,
                boundary_only,
                is_reversed,
                seam_type=seam_type,
                template=template,
            )
        else:
            path: List[Vertex2D] = Template.get_path_main(
                mesh,
                end,
                start,
                boundary_only,
                is_reversed,
                seam_type=seam_type,
                template=template,
            )
            return list(reversed(path))

    @staticmethod
    def get_path_main(
        mesh: Mesh2D,
        start: Vertex2D,
        end: Vertex2D,
        boundary_only: bool,
        reversed: bool,
        seam_type: Optional[int] = None,
        template: Optional[Template] = None,
    ) -> List[Vertex2D]:
        """
        Compute a path within a mesh from a starting vertex (start) to an ending vertex (end)
        """
        path: List[Vertex2D] = []
        prev_vertex: Vertex2D = start
        path.append(start)
        while True:
            next_vertex: Vertex2D = Template.get_next_vertex(
                prev_vertex,
                start,
                end,
                boundary_only,
                reversed,
                seam_type=seam_type,
                template=template,
            )
            if Template.same_grid_position(next_vertex, end):
                path.append(next_vertex)
                break
            if next_vertex == prev_vertex:
                raise Exception("get_path_main() failed")
            path.append(next_vertex)
            prev_vertex = next_vertex
        return path

    @staticmethod
    def same_grid_position(a: Vertex2D, b: Vertex2D) -> bool:
        a_x = a.grid_xy[0]
        a_y = a.grid_xy[1]
        b_x = b.grid_xy[0]
        b_y = b.grid_xy[1]
        if a_x == a_y == 0:
            a_x, a_y = Template.get_grid_xy(a)
        if b_x == b_y == 0:
            b_x, b_y = Template.get_grid_xy(b)

        return a_x == b_x and a_y == b_y

    @staticmethod
    def get_next_vertex(
        prev_vertex: Vertex2D,
        start: Vertex2D,
        end: Vertex2D,
        boundary_only: bool,
        reversed: bool,
        seam_type: Optional[int] = None,
        template: Optional[Template] = None,
    ) -> Vertex2D:
        """
        Find the closest vertex from a given set of candidate vertices to a specified line
        """
        vec: Vector2 = Vector2(start, end)
        line: Line2D = Line2D(start, vec)
        vs: List[Vertex2D] = Template.get_candidate_vertices(
            prev_vertex, boundary_only, reversed, seam_type=seam_type, template=template
        )

        min_d: float = 100000
        closest: Vertex2D = prev_vertex
        for v in vs:
            _vec: Vector2 = Vector2(prev_vertex, v)
            # TODO: address not only the straight line but also the curved line
            if Vector2.cos(vec, _vec) > 0:
                d: float = line.distance(v)
                if d < min_d:
                    min_d = d
                    closest = v
                elif d == min_d:
                    # By the following if statement, avoid the invalid bottomo grid edge. See https://drive.google.com/file/d/1VRy9lPBBPyDtkR0JkW7wk_Sj8y8BYvJR/view?usp=drive_link
                    if prev_vertex.x == v.x and prev_vertex.y != v.y:
                        closest = v
        if prev_vertex == closest:
            raise Exception("get_next_vertex() failed")

        return closest

    @staticmethod
    def get_consider_next_next_vertex(
        candidate_v1: Vertex2D, candidate_v2: Vertex2D, start: Vertex2D, end: Vertex2D
    ) -> Optional[Vertex2D]:
        vec: Vector2 = Vector2(start, end)
        for candidate_v in [candidate_v1, candidate_v2]:
            vs: List[Vertex2D] = Template.get_candidate_vertices(
                candidate_v, boundary=True, reversed=False
            )
            for v in vs:
                _vec: Vector2 = Vector2(candidate_v, v)
                if Vector2.cos(vec, _vec) > 0:
                    return candidate_v
        return None

    @staticmethod
    def get_candidate_vertices(
        v: Vertex2D,
        boundary: bool,
        reversed: bool,
        seam_type: Optional[int] = None,
        template: Optional[Template] = None,
    ) -> List[Vertex2D]:
        """
        Retrieve a list of candidate vertices connected to a given vertex (v) in a mesh structure
        """
        vertices_high_priority: Set[Vertex2D] = set()
        vertices: Set[Vertex2D] = set()
        sub_vertices: Set[Vertex2D] = set()
        # print('get_candidate_vertices', v, v.edges)
        for edge in v.edges:
            flag = False
            if template is not None and seam_type is not None:
                v0 = v
                v1 = edge.get_the_other_vertex(v)
                x0, y0 = Template.get_grid_xy(v0)
                x1, y1 = Template.get_grid_xy(v1)
                x: int = min(x0, x1)
                y: int = min(y0, y1)
                edge_seam_type = (
                    template.grid_vertices[x][y].X_EDGE_TYPE
                    if y0 == y1
                    else template.grid_vertices[x][y].Y_EDGE_TYPE
                )
                if edge_seam_type == seam_type:
                    # vertices.add(v1)
                    sub_vertices.add(v1)
                flag = True

            if boundary:
                # TODO: the following if statement is not perfect. Kireme is treated as non-boundary, which is wrong
                if edge.seam_type is None:
                    if (edge.left_face is not None and edge.left_face.inside > 0) and (
                        edge.right_face is not None and edge.right_face.inside > 0
                    ):
                        continue
                else:
                    if edge.seam_type == Seam.NONE:
                        continue
                if template is not None and seam_type is not None and not flag:
                    continue
            else:
                # TODO: it's not perfect to solve https://github.com/yukistavailable/Dresscode/issues/137#issuecomment-2254089186
                if seam_type == Seam.FRONT_TO_BACK:
                    if (
                        (edge.left_face is not None and edge.left_face.inside > 0)
                        and (
                            not (
                                edge.right_face is not None
                                and edge.right_face.inside > 0
                            )
                            or edge.right_face is None
                        )
                        or (
                            (
                                not (
                                    edge.left_face is not None
                                    and edge.left_face.inside > 0
                                )
                                or edge.left_face is None
                            )
                            and (
                                edge.right_face is not None
                                and edge.right_face.inside > 0
                            )
                        )
                    ):
                        sub_vertices.add(edge.get_the_other_vertex(v))

            vertices.add(edge.get_the_other_vertex(v))
        vertices = list(vertices)
        sub_vertices = list(sub_vertices)
        result = sub_vertices + vertices
        return result

    def assign_global_index_to_edges(self) -> None:
        global_index: int = 0
        for mesh in self.meshes:
            for edge in mesh.edges:
                edge.global_index = global_index
                global_index += 1

    @staticmethod
    def correspond_side_by_side_stitched_edges(templte: Template):
        stitched_edges_pairs: List[Tuple[int, int]] = []
        all_edges = set([edge for mesh in templte.meshes for edge in mesh.edges])
        while all_edges:
            edge = all_edges.pop()
            if edge.seam_type == Seam.SIDE_BY_SIDE:
                template_edge = edge.template_edge
                for other_edge in all_edges:
                    if other_edge.seam_type == Seam.SIDE_BY_SIDE:
                        other_template_edge = other_edge.template_edge
                        if template_edge.same_position_directed(other_template_edge):
                            stitched_edges_pairs.append(
                                (edge.global_index, other_edge.global_index)
                            )
                            all_edges.remove(other_edge)
                            break
        return stitched_edges_pairs

    @staticmethod
    def correspond_front_to_back_stitched_edges(
        templte_front: Template, template_back: Template
    ):
        stitched_edges_pairs: List[Tuple[int, int]] = []
        for mesh_front in templte_front.meshes:
            for mesh_back in template_back.meshes:
                for edge_front in mesh_front.edges:
                    if edge_front.seam_type == Seam.FRONT_TO_BACK:
                        template_edge_front = edge_front.template_edge
                        for edge_back in mesh_back.edges:
                            if edge_back.seam_type == Seam.FRONT_TO_BACK:
                                template_edge_back = edge_back.template_edge
                                if template_edge_front.same_position_directed(
                                    template_edge_back
                                ):
                                    stitched_edges_pairs.append(
                                        (
                                            edge_front.global_index,
                                            edge_back.global_index,
                                        )
                                    )
                                    break
        return stitched_edges_pairs


class TemplatePiece:
    def __init__(
        self,
        _piece: Optional[Piece] = None,
        _template: Optional[Template] = None,
        update_corners: bool = True,
    ):
        self.template: Optional[Template] = _template
        self.template.add_template_piece(self)
        self.piece: Optional[Piece] = _piece
        self.original_constraints: Dict[Vertex2D, Vertex2D] = {}
        self.constraints: Dict[Vertex2D, Vertex2D] = {}
        if self.piece is not None:
            self.piece.template_piece = self
        self.linked: bool = True
        self.seam_to_points: Dict[Seam, List[Vertex2D]] = {}
        self.outer_loop: List[Vertex2D]
        self.outer_loop_boundary_types: List[int]
        if update_corners:
            self.update_corners()

    def add_constraints(self, constraints: Dict[Vertex2D, Vertex2D]):
        for key, value in constraints.items():
            self.constraints[key] = value

    def add_original_constraints(self, constraints: Dict[Vertex2D, Vertex2D]):
        for key, value in constraints.items():
            self.original_constraints[key] = value

    def duplicate(self, new_piece: Piece) -> TemplatePiece:
        new_templatepiece: TemplatePiece = TemplatePiece()
        new_templatepiece.piece = new_piece
        new_templatepiece.template = self.template
        new_piece.template_piece = new_templatepiece
        new_templatepiece.update_points()
        return new_templatepiece

    def drawPanel_to_templatePanel(self, v: Vertex2D) -> Vertex2D:
        CENTER_X: float = DrawPanel.get_width() / 2
        CENTER_Y: float = DrawPanel.get_height() / 2
        x: float = v.x - CENTER_X + TEMPLATE_W / 2
        y: float = v.y - CENTER_Y + TEMPLATE_W / 2
        return Vertex2D(x, y)

    def update_corners(self, seams: Optional[List[Seam]] = None):
        if seams is None:
            seams = self.piece.get_all_seams()
        for seam in seams:
            if seam.start.corner is None:
                seam.start.corner = Vertex2D(
                    self.template.find_nearest_vertex(
                        self.drawPanel_to_templatePanel(seam.start)
                    )
                )
            if seam.end.corner is None:
                seam.end.corner = self.template.find_nearest_vertex(
                    self.drawPanel_to_templatePanel(seam.end)
                )
        self.update_points()

    def update_points(
        self,
        faces: Optional[List[Face2D]] = None,
        boundary_only: bool = False,
        consider_seam_type: bool = False,
        update_corners: bool = False,
    ):
        """
        Update the points of seams and the outer loop for a piece of template-based design or structure
        This method plays a crucial role in dynamically updating the geometry of a template-based piece, especially after changes in the template or the piece itself. It recalculates the points along the seams of the piece and updates an outer loop that defines the boundary or outline of the piece.
        """
        mesh: Optional[Mesh2D] = (
            Mesh2D(faces, integrate_adjacent_face_edges=True)
            if faces is not None
            else None
        )
        for seam in self.piece.get_all_seams():
            if mesh is not None:
                if update_corners:
                    start = mesh.find_nearest_vertex_specified_seam_type(
                        seam.start.corner, seam.type
                    )
                    end = mesh.find_nearest_vertex_specified_seam_type(
                        seam.end.corner, seam.type
                    )
                    seam.start.corner = (
                        start if start is not None else seam.start.corner
                    )
                    seam.end.corner = end if end is not None else seam.end.corner
                v0: Vertex2D = seam.start.corner
                v1: Vertex2D = seam.end.corner

                seam_points: List[Vertex2D] = Template.get_path(
                    mesh,
                    v0,
                    v1,
                    boundary_only=boundary_only,
                    is_reversed=False,
                    seam_type=seam.type if consider_seam_type else None,
                )
                seam_points: List[Vertex2D] = Template.get_path_v2(
                    mesh,
                    v0,
                    v1,
                    seam_type=seam.type if consider_seam_type else None,
                )
            else:
                v0: Vertex2D = seam.start.corner
                v1: Vertex2D = seam.end.corner
                seam_points: List[Vertex2D] = Template.get_path(
                    self.template,
                    v0,
                    v1,
                    boundary_only=boundary_only,
                    is_reversed=False,
                    seam_type=seam.type if consider_seam_type else None,
                )
            self.seam_to_points[seam] = seam_points
        self.outer_loop = []
        self.outer_loop_boundary_types = []
        for seam in self.piece.seams:
            points: List[Vertex2D] = self.seam_to_points[seam]
            for i in range(len(points) - 1):
                self.outer_loop.append(points[i])
                self.outer_loop_boundary_types.append(seam.type)

    def encloses(self, v: Vertex2D, reversed: bool) -> bool:
        """
        Checks if v is within the bounds of the piece, taking into account whether the piece is reversed.
        """
        sign: int = -1 if reversed else 1
        total: float = 0
        for seam in self.piece.seams:
            seam_points: List[Vertex2D] = self.seam_to_points[seam]
            for i in range(len(seam_points) - 1):
                v0: Vertex2D = seam_points[i]
                v1: Vertex2D = seam_points[i + 1]
                vec0: Vector2 = Vector2(v, v0)
                vec1: Vector2 = Vector2(v, v1)
                total += Vector2.get_angle_signed_180(vec0, vec1)
        total *= sign
        return total > 180
