from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from garmentimage.utils.vector2 import Vector2

EdgeInfoType = List[Dict[str, Union[np.ndarray, Dict, int]]]
PanelToEdgeInfoType = Dict[str, EdgeInfoType]
Number = Union[int, float, np.floating]
GarmentImageType = Union[str, np.ndarray, torch.Tensor]



class Edge2D:
    HORIZONTAL = 0
    VERTICAL = 1

    # TODO: Handle the case where start and end are points, not Vertex2D objects
    def __init__(
        self,
        start: Optional[Vertex2D] = None,
        end: Optional[Vertex2D] = None,
    ):
        self.start: Optional[Vertex2D]
        self.end: Optional[Vertex2D]
        if start is None and end is None:
            self.start = None
            self.end = None
        else:
            assert isinstance(start, Vertex2D), type(start)
            assert isinstance(end, Vertex2D), type(end)
            self.start = start
            self.end = end
            self.start.edges.append(self)
            self.end.edges.append(self)

        self.global_index: Optional[int] = None  # index for index across all pieces
        self.index: Optional[int] = None
        self.left_face: Optional[Face2D] = None
        self.right_face: Optional[Face2D] = None
        self.template_edge: Optional[Edge2D] = None  # stitch
        self.seam_type: Optional[int] = None
        self.direction: Optional[int] = None

    def __str__(self):
        return f"Edge2D: ({Seam.boundary_int_to_name[self.seam_type]}) {self.start} -> {self.end}"

    def __repr__(self):
        return f"Edge2D: ({Seam.boundary_int_to_name[self.seam_type]}) {self.start} -> {self.end}"

    def find_stitched_edge(self, edges: List[Edge2D]) -> Optional[Edge2D]:
        for edge in edges:
            if edge is self:
                continue
            if edge.same_position_undirected(self):
                return edge
        return None

    def align_start_end_seam_type(self, edge: Edge2D):
        self.start.x = edge.start.x
        self.start.y = edge.start.y
        self.end.x = edge.end.x
        self.end.y = edge.end.y
        self.seam_type = edge.seam_type

    def align_seam_type(self, edge: Edge2D):
        self.seam_type = edge.seam_type

    def shallow_duplicate(self) -> Edge2D:
        edge: Edge2D = Edge2D(self.start, self.end)
        edge.index = self.index
        edge.left_face = self.left_face
        edge.right_face = self.right_face
        edge.template_edge = self.template_edge
        edge.seam_type = self.seam_type
        edge.direction = self.direction
        edge.global_index = self.global_index
        return edge

    def shallow_duplicate_reverse(self) -> Edge2D:
        edge: Edge2D = Edge2D(self.end, self.start)
        edge.index = self.index
        edge.left_face = self.right_face
        edge.right_face = self.left_face
        edge.template_edge = self.template_edge
        edge.seam_type = self.seam_type
        edge.direction = self.direction
        edge.global_index = self.global_index
        return edge

    def deep_duplicate(self) -> Edge2D:
        edge: Edge2D = Edge2D(self.start.duplicate(), self.end.duplicate())
        edge.index = self.index
        edge.left_face = self.left_face
        edge.right_face = self.right_face
        edge.template_edge = self.template_edge
        edge.seam_type = self.seam_type
        edge.direction = self.direction
        edge.global_index = self.global_index
        return edge

    def deep_duplicate_reverse(self) -> Edge2D:
        edge: Edge2D = Edge2D(self.end.duplicate(), self.start.duplicate())
        edge.index = self.index
        edge.left_face = self.right_face
        edge.right_face = self.left_face
        edge.template_edge = self.template_edge
        edge.seam_type = self.seam_type
        edge.direction = self.direction
        edge.global_index = self.global_index
        return edge

    def get_the_other_vertex(self, v: Vertex2D) -> Vertex2D:
        if self.start == v:
            return self.end
        elif self.end == v:
            return self.start
        else:
            return None

    def get_opposite_face(self, face: Face2D) -> Optional[Face2D]:
        if self.left_face == face:
            return self.right_face
        elif self.right_face == face:
            return self.left_face
        else:
            return None

    def cross(self, e: Edge2D) -> bool:
        return self._cross(
            self.start.x,
            self.start.y,
            self.end.x,
            self.end.y,
            self.e.start.x,
            self.e.start.y,
            self.e.end.x,
            self.e.end.y,
        )

    # TODO: the name of this method is different from the original
    @staticmethod
    def _cross(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        xx1: float,
        yy1: float,
        xx2: float,
        yy2: float,
    ):
        a0, b0, c0 = y1 - y2, x2 - x1, y2 * x1 - x2 * y1
        a1, b1, c1 = yy1 - yy2, xx2 - xx1, yy2 * xx1 - xx2 * yy1
        return (a0 * xx1 + b0 * yy1 + c0) * (a0 * xx2 + b0 * yy2 + c0) <= 0 and (
            a1 * x1 + b1 * y1 + c1
        ) * (a1 * x2 + b1 * y2 + c1) <= 0

    @staticmethod
    def cross_point(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        xx1: float,
        yy1: float,
        xx2: float,
        yy2: float,
    ):
        a0, b0, c0 = y1 - y2, x2 - x1, y2 * x1 - x2 * y1
        a1, b1, c1 = yy1 - yy2, xx2 - xx1, yy2 * xx1 - xx2 * yy1
        if (
            abs(a0 * b1 - a1 * b0) < 1e-10
        ):  # Def.ERROR_RANGE replaced with a small number
            return None
        x = (b0 * c1 - b1 * c0) / (a0 * b1 - a1 * b0)
        y = (a0 * c1 - a1 * c0) / (a1 * b0 - a0 * b1)
        return Vertex2D(x, y)

    def right_side_of_edge(
        self, x1: float, y1: float, x2: float, y2: float, x: float, y: float
    ):
        a0, b0, c0 = y1 - y2, x2 - x1, y2 * x1 - x2 * y1
        return a0 * x + b0 * y + c0 < 0

    def equals(self, v0: Vertex2D, v1: Vertex2D):
        return (v0 == self.start and v1 == self.end) or (
            v1 == self.start and v0 == self.end
        )

    def equals_position(self, v0: Vertex2D, v1: Vertex2D):
        return (
            v0.x == self.start.x
            and v0.y == self.start.y
            and v1.x == self.end.x
            and v1.y == self.end.y
        ) or (
            v1.x == self.start.x
            and v1.y == self.start.y
            and v0.x == self.end.x
            and v0.y == self.end.y
        )

    def same_position_undirected(self, edge: Edge2D):
        return (
            self.start.x == edge.start.x
            and self.start.y == edge.start.y
            and self.end.x == edge.end.x
            and self.end.y == edge.end.y
        ) or (
            self.start.x == edge.end.x
            and self.start.y == edge.end.y
            and self.end.x == edge.start.x
            and self.end.y == edge.start.y
        )

    def same_position_directed(self, edge: Edge2D):
        return (
            self.start.x == edge.start.x
            and self.start.y == edge.start.y
            and self.end.x == edge.end.x
            and self.end.y == edge.end.y
        )

    def contains(self, v: Vertex2D) -> bool:
        return v == self.start or v == self.end

    def vector2(self) -> Vector2:
        return Vector2(self.start, self.end)

    def mid_point(self) -> Vertex2D:
        return Vertex2D(
            (self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2
        )

    def length(self) -> float:
        return Vertex2D.distance(self.start, self.end)

    def get_common_vertex(self, edge: Edge2D) -> Vertex2D:
        if self.start == edge.start or self.start == edge.end:
            return self.start
        elif self.end == edge.start or self.end == edge.end:
            return self.end
        else:
            raise ValueError("The edges do not share a common vertex")

    def distance(self, v: Vertex2D) -> float:
        a = self.end.x - self.start.x
        b = self.end.y - self.start.y
        bunbo = math.sqrt(a**2 + b**2)
        bunshi = a * (v.y - self.start.y) - b * (v.x - self.start.x)
        return math.fabs(bunshi) / bunbo if bunbo else 0

    def distance_as_a_segment(self, node: Vertex2D) -> float:
        vec0 = Vector2(self.start.x - self.end.x, self.start.y - self.end.y)
        vec1 = Vector2(self.start.x - node.x, self.start.y - node.y)

        if Vector2.dot_product(vec0, vec1) < 0:
            return Vector2.distance(self.start, node)

        vec1 = Vector2(self.end.x - node.x, self.end.y - node.y)

        if Vector2.dot_product(vec0, vec1) > 0:
            return Vector2.distance(self.end, node)

        # Use the distance formula for a perpendicular distance from a point to a line
        return self.distance(node)

    def is_boundary(self) -> bool:
        if self.seam_type is None or self.seam_type == Seam.NONE:
            return False
        assert (self.left_face is not None and self.left_face.inside == 1) or (
            self.right_face is not None and self.right_face.inside == 1
        )
        return True

    def is_cut(self) -> bool:
        """
        Judge whether the seam is cut inside a pattern or not
        Cut means the seam whose seam_type is not None but the right face and left face are inside
        """
        if self.seam_type is None or self.seam_type == Seam.NONE:
            return False
        if (self.left_face is not None and self.left_face.inside == 1) and (
            self.right_face is not None and self.right_face.inside == 1
        ):
            return True
        return False

    @staticmethod
    def is_straight_stroke(edges: List[Edge2D]) -> bool:
        for i in range(0, len(edges)):
            edge1 = edges[i]
            edge2 = edges[(i + 1) % len(edges)]
            if edge1.start.x != edge2.end.x and edge1.start.y != edge2.end.y:
                return False
        return True

    @staticmethod
    def encloses(v: Vertex2D, edges: List[Edge2D]) -> bool:
        """
        Check if the given vertex `v` is enclosed by the given list of edges.

        Parameters
        ----------
            v (Vertex2D): The vertex to check.
            edges (List[Edge2D]): The list of edges to check against.

        Returns
        -------
            bool: True if the vertex is enclosed by the edges, False otherwise.

        Algorithm:
            1. Initialize the `total` variable to 0.
            2. Iterate over each edge in the `edges` list.
            3. Get the start and end vertices of the edge.
            4. Create `v1` and `v2` vectors representing the direction from `v` to the start and end vertices.
            5. Calculate the cross product of `v1` and `v2`.
            6. If the cross product is greater than or equal to 0, add the signed 180 degree angle between `v1` and `v2` to `total`.
            7. If the cross product is less than 0, subtract the signed 180 degree angle between `v1` and `v2` from `total`.
            8. Repeat steps 2-7 for all edges.
            9. Return True if `total` is greater than 180, indicating that the vertex is enclosed by the edges.
        """
        total: float = 0
        for i in range(0, len(edges)):
            edge = edges[i]
            start = edge.start
            end = edge.end
            v1: Vector2 = Vector2(v, start)
            v2: Vector2 = Vector2(v, end)
            # TODO check the direction of the edge
            # cross_product = Vector2.cross_product(v1, v2)
            # if cross_product >= 0:
            #     total += Vector2.get_angle_signed_180(v1, v2)
            # else:
            #     total -= Vector2.get_angle_signed_180(v1, v2)
            total += Vector2.get_angle_signed_180(v1, v2)
        return total > 180


class Edge2D:
    HORIZONTAL = 0
    VERTICAL = 1

    # TODO: Handle the case where start and end are points, not Vertex2D objects
    def __init__(
        self,
        start: Optional[Vertex2D] = None,
        end: Optional[Vertex2D] = None,
    ):
        self.start: Optional[Vertex2D]
        self.end: Optional[Vertex2D]
        if start is None and end is None:
            self.start = None
            self.end = None
        else:
            assert isinstance(start, Vertex2D), type(start)
            assert isinstance(end, Vertex2D), type(end)
            self.start = start
            self.end = end
            self.start.edges.append(self)
            self.end.edges.append(self)

        self.global_index: Optional[int] = None  # index for index across all pieces
        self.index: Optional[int] = None
        self.left_face: Optional[Face2D] = None
        self.right_face: Optional[Face2D] = None
        self.template_edge: Optional[Edge2D] = None  # stitch
        self.seam_type: Optional[int] = None
        self.direction: Optional[int] = None

    def __str__(self):
        return f"Edge2D: ({Seam.boundary_int_to_name[self.seam_type]}) {self.start} -> {self.end}"

    def __repr__(self):
        return f"Edge2D: ({Seam.boundary_int_to_name[self.seam_type]}) {self.start} -> {self.end}"

    def find_stitched_edge(self, edges: List[Edge2D]) -> Optional[Edge2D]:
        for edge in edges:
            if edge is self:
                continue
            if edge.same_position_undirected(self):
                return edge
        return None

    def align_start_end_seam_type(self, edge: Edge2D):
        self.start.x = edge.start.x
        self.start.y = edge.start.y
        self.end.x = edge.end.x
        self.end.y = edge.end.y
        self.seam_type = edge.seam_type

    def align_seam_type(self, edge: Edge2D):
        self.seam_type = edge.seam_type

    def shallow_duplicate(self) -> Edge2D:
        edge: Edge2D = Edge2D(self.start, self.end)
        edge.index = self.index
        edge.left_face = self.left_face
        edge.right_face = self.right_face
        edge.template_edge = self.template_edge
        edge.seam_type = self.seam_type
        edge.direction = self.direction
        edge.global_index = self.global_index
        return edge

    def shallow_duplicate_reverse(self) -> Edge2D:
        edge: Edge2D = Edge2D(self.end, self.start)
        edge.index = self.index
        edge.left_face = self.right_face
        edge.right_face = self.left_face
        edge.template_edge = self.template_edge
        edge.seam_type = self.seam_type
        edge.direction = self.direction
        edge.global_index = self.global_index
        return edge

    def deep_duplicate(self) -> Edge2D:
        edge: Edge2D = Edge2D(self.start.duplicate(), self.end.duplicate())
        edge.index = self.index
        edge.left_face = self.left_face
        edge.right_face = self.right_face
        edge.template_edge = self.template_edge
        edge.seam_type = self.seam_type
        edge.direction = self.direction
        edge.global_index = self.global_index
        return edge

    def deep_duplicate_reverse(self) -> Edge2D:
        edge: Edge2D = Edge2D(self.end.duplicate(), self.start.duplicate())
        edge.index = self.index
        edge.left_face = self.right_face
        edge.right_face = self.left_face
        edge.template_edge = self.template_edge
        edge.seam_type = self.seam_type
        edge.direction = self.direction
        edge.global_index = self.global_index
        return edge

    def get_the_other_vertex(self, v: Vertex2D) -> Vertex2D:
        if self.start == v:
            return self.end
        elif self.end == v:
            return self.start
        else:
            return None

    def get_opposite_face(self, face: Face2D) -> Optional[Face2D]:
        if self.left_face == face:
            return self.right_face
        elif self.right_face == face:
            return self.left_face
        else:
            return None

    def cross(self, e: Edge2D) -> bool:
        return self._cross(
            self.start.x,
            self.start.y,
            self.end.x,
            self.end.y,
            self.e.start.x,
            self.e.start.y,
            self.e.end.x,
            self.e.end.y,
        )

    # TODO: the name of this method is different from the original
    @staticmethod
    def _cross(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        xx1: float,
        yy1: float,
        xx2: float,
        yy2: float,
    ):
        a0, b0, c0 = y1 - y2, x2 - x1, y2 * x1 - x2 * y1
        a1, b1, c1 = yy1 - yy2, xx2 - xx1, yy2 * xx1 - xx2 * yy1
        return (a0 * xx1 + b0 * yy1 + c0) * (a0 * xx2 + b0 * yy2 + c0) <= 0 and (
            a1 * x1 + b1 * y1 + c1
        ) * (a1 * x2 + b1 * y2 + c1) <= 0

    @staticmethod
    def cross_point(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        xx1: float,
        yy1: float,
        xx2: float,
        yy2: float,
    ):
        a0, b0, c0 = y1 - y2, x2 - x1, y2 * x1 - x2 * y1
        a1, b1, c1 = yy1 - yy2, xx2 - xx1, yy2 * xx1 - xx2 * yy1
        if (
            abs(a0 * b1 - a1 * b0) < 1e-10
        ):  # Def.ERROR_RANGE replaced with a small number
            return None
        x = (b0 * c1 - b1 * c0) / (a0 * b1 - a1 * b0)
        y = (a0 * c1 - a1 * c0) / (a1 * b0 - a0 * b1)
        return Vertex2D(x, y)

    def right_side_of_edge(
        self, x1: float, y1: float, x2: float, y2: float, x: float, y: float
    ):
        a0, b0, c0 = y1 - y2, x2 - x1, y2 * x1 - x2 * y1
        return a0 * x + b0 * y + c0 < 0

    def equals(self, v0: Vertex2D, v1: Vertex2D):
        return (v0 == self.start and v1 == self.end) or (
            v1 == self.start and v0 == self.end
        )

    def equals_position(self, v0: Vertex2D, v1: Vertex2D):
        return (
            v0.x == self.start.x
            and v0.y == self.start.y
            and v1.x == self.end.x
            and v1.y == self.end.y
        ) or (
            v1.x == self.start.x
            and v1.y == self.start.y
            and v0.x == self.end.x
            and v0.y == self.end.y
        )

    def same_position_undirected(self, edge: Edge2D):
        return (
            self.start.x == edge.start.x
            and self.start.y == edge.start.y
            and self.end.x == edge.end.x
            and self.end.y == edge.end.y
        ) or (
            self.start.x == edge.end.x
            and self.start.y == edge.end.y
            and self.end.x == edge.start.x
            and self.end.y == edge.start.y
        )

    def same_position_directed(self, edge: Edge2D):
        return (
            self.start.x == edge.start.x
            and self.start.y == edge.start.y
            and self.end.x == edge.end.x
            and self.end.y == edge.end.y
        )

    def contains(self, v: Vertex2D) -> bool:
        return v == self.start or v == self.end

    def vector2(self) -> Vector2:
        return Vector2(self.start, self.end)

    def mid_point(self) -> Vertex2D:
        return Vertex2D(
            (self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2
        )

    def length(self) -> float:
        return Vertex2D.distance(self.start, self.end)

    def get_common_vertex(self, edge: Edge2D) -> Vertex2D:
        if self.start == edge.start or self.start == edge.end:
            return self.start
        elif self.end == edge.start or self.end == edge.end:
            return self.end
        else:
            raise ValueError("The edges do not share a common vertex")

    def distance(self, v: Vertex2D) -> float:
        a = self.end.x - self.start.x
        b = self.end.y - self.start.y
        bunbo = math.sqrt(a**2 + b**2)
        bunshi = a * (v.y - self.start.y) - b * (v.x - self.start.x)
        return math.fabs(bunshi) / bunbo if bunbo else 0

    def distance_as_a_segment(self, node: Vertex2D) -> float:
        vec0 = Vector2(self.start.x - self.end.x, self.start.y - self.end.y)
        vec1 = Vector2(self.start.x - node.x, self.start.y - node.y)

        if Vector2.dot_product(vec0, vec1) < 0:
            return Vector2.distance(self.start, node)

        vec1 = Vector2(self.end.x - node.x, self.end.y - node.y)

        if Vector2.dot_product(vec0, vec1) > 0:
            return Vector2.distance(self.end, node)

        # Use the distance formula for a perpendicular distance from a point to a line
        return self.distance(node)

    def is_boundary(self) -> bool:
        if self.seam_type is None or self.seam_type == Seam.NONE:
            return False
        assert (self.left_face is not None and self.left_face.inside == 1) or (
            self.right_face is not None and self.right_face.inside == 1
        )
        return True

    def is_cut(self) -> bool:
        """
        Judge whether the seam is cut inside a pattern or not
        Cut means the seam whose seam_type is not None but the right face and left face are inside
        """
        if self.seam_type is None or self.seam_type == Seam.NONE:
            return False
        if (self.left_face is not None and self.left_face.inside == 1) and (
            self.right_face is not None and self.right_face.inside == 1
        ):
            return True
        return False

    @staticmethod
    def is_straight_stroke(edges: List[Edge2D]) -> bool:
        for i in range(0, len(edges)):
            edge1 = edges[i]
            edge2 = edges[(i + 1) % len(edges)]
            if edge1.start.x != edge2.end.x and edge1.start.y != edge2.end.y:
                return False
        return True

    @staticmethod
    def encloses(v: Vertex2D, edges: List[Edge2D]) -> bool:
        """
        Check if the given vertex `v` is enclosed by the given list of edges.

        Parameters
        ----------
            v (Vertex2D): The vertex to check.
            edges (List[Edge2D]): The list of edges to check against.

        Returns
        -------
            bool: True if the vertex is enclosed by the edges, False otherwise.

        Algorithm:
            1. Initialize the `total` variable to 0.
            2. Iterate over each edge in the `edges` list.
            3. Get the start and end vertices of the edge.
            4. Create `v1` and `v2` vectors representing the direction from `v` to the start and end vertices.
            5. Calculate the cross product of `v1` and `v2`.
            6. If the cross product is greater than or equal to 0, add the signed 180 degree angle between `v1` and `v2` to `total`.
            7. If the cross product is less than 0, subtract the signed 180 degree angle between `v1` and `v2` from `total`.
            8. Repeat steps 2-7 for all edges.
            9. Return True if `total` is greater than 180, indicating that the vertex is enclosed by the edges.
        """
        total: float = 0
        for i in range(0, len(edges)):
            edge = edges[i]
            start = edge.start
            end = edge.end
            v1: Vector2 = Vector2(v, start)
            v2: Vector2 = Vector2(v, end)
            # TODO check the direction of the edge
            # cross_product = Vector2.cross_product(v1, v2)
            # if cross_product >= 0:
            #     total += Vector2.get_angle_signed_180(v1, v2)
            # else:
            #     total -= Vector2.get_angle_signed_180(v1, v2)
            total += Vector2.get_angle_signed_180(v1, v2)
        return total > 180


class Seam:
    UNIT_LENGTH: int = 10
    LINE_WIDTH: int = 2

    NONE: int = 0
    SIDE_BY_SIDE: int = 1
    FRONT_TO_BACK: int = 2
    BOUNDARY: int = 3

    boundary_int_to_name: Dict[Optional[int], str] = {
        None: "None",
        NONE: "None",
        SIDE_BY_SIDE: "Side by side",
        FRONT_TO_BACK: "Front to back",
        BOUNDARY: "Boundary",
    }

    boundary_types_to_linestyle = {
        None: "-",
        NONE: "-",
        SIDE_BY_SIDE: ":",
        FRONT_TO_BACK: "--",
        BOUNDARY: "-",
    }

    # see https://help.masui.org/%E5%B2%A1%E9%83%A8%E3%83%BB%E4%BC%8A%E8%97%A4%E3%81%AE%E3%82%AB%E3%83%A9%E3%83%BC%E3%83%91%E3%83%AC%E3%83%83%E3%83%88-63cfa2f21e6783001e8395e0
    boundary_types_to_color = {
        NONE: "#56B4E9",
        SIDE_BY_SIDE: "#CC79A7",
        FRONT_TO_BACK: "#009E73",
        BOUNDARY: "#E69F00",
    }
    # constraint_color = "#0072B2"
    constraint_color = "#4339FC"

    def __init__(self, stroke: Optional[List[Vertex2D]]):
        self.start: Vertex2D
        self.end: Vertex2D
        self.points: List[Vertex2D]
        if stroke is None or len(stroke) == 0:
            self.start = Vertex2D()
            self.end = Vertex2D()
            self.points = []
        else:
            self.start = stroke[0]
            self.end = stroke[-1]
            self.points = stroke
        self.type: int = self.BOUNDARY
        self.stroke: Optional[List[Vertex2D]] = None
        self.piece: Optional[Piece] = None

    def __str__(self):
        return f"Seam: {self.start} -> {self.end}, type: {self.type_to_string()}"

    def __repr__(self):
        return f"Seam: {self.start} -> {self.end}, type: {self.type_to_string()}"

    def type_to_string(self) -> str:
        if self.type == self.NONE:
            return "None"
        elif self.type == self.SIDE_BY_SIDE:
            return "Side by side"
        elif self.type == self.FRONT_TO_BACK:
            return "Front to back"
        elif self.type == self.BOUNDARY:
            return "Boundary"
        else:
            return "Unknown"

    @staticmethod
    def to_sg_polygon(seams: List[Seam]) -> sg.Polygon:
        start = seams[0].start
        end = seams[-1].end
        # assert Vertex2D.same_position(start, end)
        polygon_points = []
        for seam in seams:
            start = seam.start
            end = seam.end
            polygon_points.append((start.x, start.y))
            for point in seam.points:
                polygon_points.append((point.x, point.y))
            polygon_points.append((end.x, end.y))
        return sg.Polygon(polygon_points)

    @classmethod
    def from_file(cls, lines: List[str], start_index: int) -> Tuple[Seam, int]:
        assert lines is not None
        index: int = start_index
        while True:
            if lines[index].strip() == "</seam>":
                index += 1
                break
            if lines[index].strip() == "<type>":
                index += 1
                line: str = lines[index].strip()
                type: int = int(line)
                index += 1
                line: str = lines[index].strip()
                assert line == "</type>"
                index += 1
            if lines[index].strip() == "<start>":
                index += 1
                start, index = cls.load_vertex_uv(lines, index)
                index += 1
            if lines[index].strip() == "<end>":
                index += 1
                end, index = cls.load_vertex_uv(lines, index)
                index += 1
            if lines[index].strip() == "<points>":
                index += 1
                points: List[Vertex2D] = []
                line: str = lines[index].strip()
                if line != "":
                    it: map[float] = map(float, line.split())
                    while True:
                        try:
                            x, y = next(it), next(it)
                            points.append(Vertex2D(x, y))
                        except StopIteration:
                            break
                index += 1
                assert lines[index].strip() == "</points>"
                index += 1
        # print("Seam loaded")
        # print(lines[index].strip())
        # print(lines[index - 1].strip())

        seam: Seam = Seam(points)
        seam.start = start
        seam.end = end
        seam.type = type
        seam.points = points
        seam.set_stroke()
        return seam, index

    def set_points(self, stroke: List[Vertex2D]):
        x_vector = Vector2(self.start.x - self.end.x, self.start.y - self.end.y)
        y_vector = Vector2(-x_vector.y, x_vector.x)
        self.points = []
        for v in stroke[1:-1]:
            u = self.get_relative_coords(v, self.start, x_vector, y_vector)
            self.points.append(u)

    @staticmethod
    def get_relative_coords(
        u: Vertex2D, base: Vertex2D, x_vector: Vertex2D, y_vector: Vertex2D
    ) -> Vertex2D:
        vec = Vector2(base.x - u.x, base.y - u.y)
        x = Vector2.dot_product(vec, x_vector) / Vector2.dot_product(x_vector, x_vector)
        y = Vector2.dot_product(vec, y_vector) / Vector2.dot_product(y_vector, y_vector)
        return Vertex2D(x, y)

    @staticmethod
    def get_absolute_coords(
        u: Vertex2D, base: Vertex2D, x_vector: Vertex2D, y_vector: Vertex2D
    ) -> Vertex2D:
        return Vertex2D.translate(
            base,
            Vector2.add(
                Vector2.multiply(x_vector, u.x), Vector2.multiply(y_vector, u.y)
            ),
        )

    def set_stroke(self):
        x_vector: Vector2 = Vector2(self.start, self.end)
        y_vector: Vector2 = Vector2(-x_vector.y, x_vector.x)

        self.stroke: List[Vertex2D] = []
        self.stroke.append(self.start)
        for i in range(len(self.points)):
            u: Vertex2D = self.points[i]
            if -1 < u.x < 1 and -1 < u.y < 1:
                v: Vertex2D = Seam.get_absolute_coords(
                    u, self.start, x_vector, y_vector
                )
            else:
                # NeuralTailorConverter
                v: Vertex2D = Vertex2D(u.x, u.y)
            self.stroke.append(v)
        self.stroke.append(self.end)

    # Example method to calculate the length of the seam
    def get_length(self) -> float:
        if self.stroke is None:
            self.set_stroke()
        length = 0
        for i in range(len(self.stroke) - 1):
            p = self.points[i]
            q = self.points[i + 1]
            length += Vertex2D.distance_static(p, q)
        return length

    # TODO: the following implementation might be different from the original
    def get_mid_point(self) -> Vertex2D:
        if len(self.stroke) % 2 == 1:
            return self.stroke[len(self.stroke) // 2]
        else:
            v0: Vertex2D = self.stroke[len(self.stroke) // 2 - 1]
            v1: Vertex2D = self.stroke[len(self.stroke) // 2]
            return Vertex2D.mid_point(v0, v1)

    def get_mid_point_normal(self):
        p: Vertex2D = self.stroke(len(self.stroke) // 2 - 1)
        q: Vertex2D = self.stroke(len(self.stroke) // 2)
        vec: Vector2 = Vector2(p, q)
        vec.normalize_self()
        if self.piece.reversed:
            vec.multiply_self(-1)
        return vec

    def get_opposite_vertex(self, v: Vertex2D) -> Vertex2D:
        if v == self.start:
            return self.end
        else:
            return self.start

    def get_flipped(self, axis_x: float):
        new_stroke: List[Vertex2D] = []
        for i in range(len(self.stroke) - 1, -1, -1):
            v: Vertex2D = self.stroke[i]
            u: Vertex2D = Vertex2D(axis_x + (axis_x - v.x), v.y)
            new_stroke.append(u)
        new_seam = Seam(new_stroke)
        return new_seam

    def reverse(self, axis_x: Optional[float]):
        if axis_x is not None:
            self.start.x = axis_x + (axis_x - self.start.x)
            for i in range(len(self.points)):
                v = self.points[i]
                v.y = -v.y
        else:
            end_: Vertex2D = self.end
            self.end = self.start
            self.start = end_
            # reverse the order of the points
            # reverse method is a standard method in python
            self.points.reverse()
            for i in range(len(self.points)):
                v = self.points[i]
                v.y = -v.y
                v.x = 1 - v.x

    def duplicate(self):
        self.set_stroke()
        new_seam: Seam = Seam(self.stroke)
        new_seam.start = Vertex2D(self.start)
        new_seam.start.corner = Vertex2D(self.start.corner)
        new_seam.end = Vertex2D(self.end)
        new_seam.end.corner = Vertex2D(self.end.corner)
        new_seam.type = self.type
        return new_seam

    # TODO: implement
    def save(self):
        pass

    # TODO: implement
    def save_inner(self):
        pass

    # TODO: implement
    def save_main(self):
        pass

    # TODO: check if the following implementation is correct
    @classmethod
    def load_vertex_uv(cls, lines: List[str], start_index: int) -> Tuple[Vertex2D, int]:
        index: int = start_index
        line: str = lines[start_index].strip()
        # the format of line is like "202.0 89.5 0.3125 0.1875"
        x, y, u, v = map(float, line.split())
        index += 1
        vertex: Vertex2D = Vertex2D(x, y)
        vertex.uv = Vertex2D(u, v)
        return vertex, index


class Piece:
    def __init__(self, seams: Optional[List[Seam]]) -> None:
        self.template_piece: Optional[TemplatePiece] = None
        self.original_piece: Optional[Piece] = None
        self.child_piece: Optional[Piece] = None
        self.reversed: bool = False
        self.layer: int = 0
        self.seams: Optional[List[Seam]] = None
        self.inner_seams: Optional[List[Seam]] = None
        self.seam_to_stroke_indices: Dict[Seam, List[int]] = {}
        self.triangle_points: List[Vertex2D] = []
        self.triangle_indices: List[List[int]] = []
        if seams is not None:
            start: Vertex2D = seams[0].start
            end: Vertex2D = seams[-1].end

            if start != end:
                if not Vertex2D.same_position(start, end):
                    print("WARRNING: The start and end of the seams are not the same")
                # TODO: Check if the following line is correct
                seams[-1].end = start
            self.seams = seams
            self.update_network()

    @staticmethod
    def sorted_seams(seams: List[Seam]) -> List[Seam]:
        # find the left bottom seam
        left_bottom_seam: Seam = seams[0]
        for seam in seams:
            if (
                seam.start.x + seam.start.y
                < left_bottom_seam.start.x + left_bottom_seam.start.y
            ):
                left_bottom_seam = seam
        # sort the seams
        sorted_seams: List[Seam] = []
        sorted_seams.append(left_bottom_seam)
        while len(sorted_seams) < len(seams):
            last_seam: Seam = sorted_seams[-1]
            for seam in seams:
                if last_seam.end == seam.start:
                    sorted_seams.append(seam)
        return sorted_seams

    @staticmethod
    def visualize_templates(
        pieces: List[Piece],
        piece_to_constraints: Optional[Dict[Piece, Dict[Vertex2D, Vertex2D]]] = None,
        show_markers: bool = True,
    ):
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")

        if not show_markers:
            marker = "None"
        else:
            marker = "o"

        constraint_marker = "x"

        colors = ["dodgerblue", "fuchsia", "lime", "orange"]
        color_index = 0

        for piece in pieces:
            color_index = 1 if piece.reversed else 0
            template_piece: TemplatePiece = piece.template_piece
            if piece_to_constraints is not None:
                constraints = piece_to_constraints[piece]
                for point in constraints.keys():
                    ax.plot(
                        point.x,
                        point.y,
                        marker=constraint_marker,
                        linestyle="-",
                        color=colors[color_index],
                        alpha=1.0,
                        markersize=10,
                    )
            prev_x = start_x = template_piece.outer_loop[0].x
            prev_y = start_y = template_piece.outer_loop[0].y
            for i in range(1, len(template_piece.outer_loop)):
                v = template_piece.outer_loop[i]
                boundary_type = template_piece.outer_loop_boundary_types[i - 1]
                linestyle = Seam.boundary_types_to_linestyle[boundary_type]
                ax.plot(
                    [prev_x, v.x],
                    [prev_y, v.y],
                    marker=marker,
                    linestyle=linestyle,
                    color=colors[color_index],
                    alpha=1.0,
                )
                prev_x = v.x
                prev_y = v.y
            boundary_type = template_piece.outer_loop_boundary_types[-1]
            linestyle = Seam.boundary_types_to_linestyle[boundary_type]
            ax.plot(
                [prev_x, start_x],
                [start_y, v.y],
                marker=marker,
                linestyle=linestyle,
                color=colors[color_index],
                alpha=1.0,
            )

        ax.set_xticks(
            range(
                int(
                    min(
                        v.x for piece in pieces for v in piece.template_piece.outer_loop
                    )
                )
                - 1,
                int(
                    max(
                        v.x for piece in pieces for v in piece.template_piece.outer_loop
                    )
                )
                + 2,
            ),
            minor=True,
        )
        ax.set_yticks(
            range(
                int(
                    min(
                        v.y for piece in pieces for v in piece.template_piece.outer_loop
                    )
                )
                - 1,
                int(
                    max(
                        v.y for piece in pieces for v in piece.template_piece.outer_loop
                    )
                )
                + 2,
            ),
            minor=True,
        )
        plt.show()

    @classmethod
    def from_file(cls, lines: List[str], start_index: int) -> Tuple[Piece, int]:
        assert lines is not None
        index: int = start_index
        seams: List[Seam] = []
        inner_seams: List[Seam] = []
        layer: Optional[int] = None
        reversed: Optional[bool] = None
        while True:
            # print(index, lines[index].strip())
            if lines[index].strip() == "</piece>":
                index += 1
                break
            if lines[index].strip() == "<reversed>":
                assert reversed is None
                index += 1
                line: str = lines[index].strip()
                reversed: bool = line == "true"
                index += 1
                line: str = lines[index].strip()
                assert line == "</reversed>"
                index += 1
            if lines[index].strip() == "<layer>":
                assert layer is None
                index += 1
                line: str = lines[index].strip()
                layer: int = int(line)
                index += 1
                line: str = lines[index].strip()
                assert line == "</layer>"
                index += 1
            if lines[index].strip() == "<seam>":
                index += 1
                seam, index = Seam.from_file(lines, index)
                seams.append(seam)
            if lines[index].strip() == "<inner_seam>":
                index += 1
                inner_seam, index = Seam.from_file(lines, index)
                inner_seams.append(inner_seam)
            if lines[index].strip() == "":
                index += 1
        piece: Piece = Piece(seams)
        piece.inner_seams = inner_seams
        if reversed is not None:
            if reversed and layer == 0:
                layer = 1
        piece.reversed = reversed
        if reversed is not None:
            if piece.reversed:
                piece.layer = 1
            else:
                piece.layer = 0
        elif layer is not None:
            piece.layer = layer
        piece.update_network()
        return piece, index

    def get_all_seams(self) -> List[Seam]:
        all_seams: List[Seam] = []
        if self.seams is not None:
            all_seams.extend(self.seams)
        if self.inner_seams is not None:
            all_seams.extend(self.inner_seams)
        return all_seams

    def update_triangle_mesh(self, max_area: float = 1000.0):
        boundary_edges: List[Edge2D] = []
        for i in range(len(self.seams)):
            seam: Seam = self.seams[i]
            for j in range(len(seam.stroke) - 1):
                v0: Vertex2D = seam.stroke[j]
                v1: Vertex2D = seam.stroke[j + 1]
                edge: Edge2D = Edge2D(v0, v1)
                boundary_edges.append(edge)
        boundary_vertices: List[Vertex2D] = (
            Mesh2D.get_boundary_vertices_of_undirected_edges(boundary_edges)
        )
        polygon_points = [(v.x, v.y) for v in boundary_vertices]
        polygon = sg.Polygon(polygon_points)
        a = dict(vertices=polygon.exterior.coords[:-1])
        t = tr.triangulate(a, f"qa{max_area}")
        self.triangle_points = [Vertex2D(v[0], v[1]) for v in t["vertices"]]
        self.triangle_indices = t["triangles"]

    def visualize_triangle_mesh(self):
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        triangle_points = np.array([[v.x, v.y] for v in self.triangle_points])
        plt.triplot(triangle_points[:, 0], triangle_points[:, 1], self.triangle_indices)
        plt.show()

    def duplicate(self) -> Piece:
        new_seams: List[Seam] = []
        if self.seams is not None:
            for seam in self.seams:
                new_seams.append(seam.duplicate())
        new_piece: Piece = Piece(new_seams)

        if self.inner_seams is not None:
            for seam in self.inner_seams:
                new_piece.inner_seams.append(seam.duplicate())

        new_piece.layer = self.layer
        new_piece.reversed = self.reversed
        new_piece.template_piece = self.template_piece.duplicate(new_piece)
        new_piece.update_network()
        return new_piece

    def get_backside(self) -> Piece:
        new_seams: List[Seam] = []
        for i in range(len(self.seams) - 1, -1, -1):
            seam: Seam = self.seams[i]
            seam.set_stroke()
            stroke: List[Vertex2D] = seam.stroke
            new_stroke: List[Vertex2D] = []
            for j in range(len(stroke) - 1, -1, -1):
                v: Vertex2D = stroke[j]
                new_stroke.append(Vertex2D(v.x, v.y))
            new_seam: Seam = Seam(new_stroke)
            new_seam.type = seam.type
            new_seams.append(new_seam)

        new_piece: Piece = Piece(new_seams)
        new_piece.reversed = True
        if self.layer == 0:
            new_piece.layer = 1
        elif self.layer == 1:
            new_piece.layer = 0
        elif self.layer == 3:
            new_piece.layer = 2
        elif self.layer == 2:
            new_piece.layer = 3
        else:
            raise ValueError(f"Invalid layer number: {self.layer}")

    def encloses(self, v: Vertex2D) -> bool:
        sign: int = -1 if self.reversed else 1
        total: float = 0
        for i in range(len(self.seams)):
            seam: Seam = self.seams[i]
            v0: Vertex2D = seam.start
            v1: Vertex2D = seam.end
            vec0: Vector2 = Vector2(v, v0)
            vec1: Vector2 = Vector2(v, v1)
            total += Vector2.get_angle_signed_180(vec0, vec1)

        total *= sign
        return total > 180

    @staticmethod
    def calculate_area(stroke: List[Vertex2D]) -> float:
        p: Vertex2D = stroke[-1]
        area: float = 0
        for q in stroke:
            area += Vector2.cross_product(p, q)
            p = q
        return area

    @staticmethod
    def adjust_loop_direction(seams: List[Seam]) -> List[Seam]:
        stroke: List[Vertex2D] = Piece.get_stroke(seams)
        if Piece.calculate_area(stroke) > 0:
            return seams
        return Piece.reverse_seams(seams)

    def update_network(self):
        if self.seams is not None:
            for i in range(len(self.seams)):
                seam0: Seam = self.seams[i]
                seam1: Seam = self.seams[(i + 1) % len(self.seams)]
                seam0.end.prev_seam = seam0
                seam0.end.next_seam = seam1
                seam1.start = seam0.end
                seam0.piece = self
        if self.inner_seams is not None:
            for i in range(len(self.inner_seams)):
                seam0: Seam = self.inner_seams[i]
                seam0.piece = self
                seam0.end.prev_seam = seam0
                seam0.start.next_seam = seam0
            for i in range(len(self.inner_seams) - 1):
                seam0: Seam = self.inner_seams[i]
                for j in range(i + 1, len(self.inner_seams)):
                    seam1: Seam = self.inner_seams[j]
                    if Vertex2D.same_position(seam0.end, seam1.start):
                        seam1.start = seam0.end
                        seam0.end.next_seam = seam1
                    elif Vertex2D.same_position(seam0.start, seam1.end):
                        seam0.start = seam1.end
                        seam1.end.next_seam = seam0

    def delete(self):
        for seam in self.seams:
            if seam.connector is not None:
                seam.connector.delete()

    def prepare_paint(self):
        for seam in self.get_all_seams():
            seam.set_stroke()

    # TODO: implement
    def paint(self, g):
        pass

    # TODO: implement
    def paint_inside(self, g):
        pass

    # TODO: implement
    # Polygonの実装がめんどくさそうなので飛ばす
    def inside(self, p):
        pass

    def translate(self, vec: Vector2):
        for seam in self.seams:
            seam.start.add_self(vec)
        for seam in self.inner_seams:
            seam.start.add_self(vec)

    def pick_seam(self, p: Point, d: List[float]) -> Optional[Seam]:
        min: float = 10000000
        closest_seam: Optional[Seam] = None

        for seam in self.get_all_seams():
            dist: int = seam.distance(p)
            if dist < min:
                min = dist
                closest_seam = seam
        d[0] = min
        return closest_seam

    # TODO: implement
    # DrawPanelの実装がめんどくさそうなので飛ばす
    def get_flipped(self) -> Piece:
        pass

    # TODO: implement
    def get_bbox(self):
        pass

    # TODO: implement
    def get_reverse(self):
        pass

    # TODO: implement
    def symmetrize(self):
        pass

    # TODO: implement
    def symmetrize_seam(self, seam: Seam):
        pass

    # TODO: implement
    def cut_seam(self, seam: Seam, p: Vertex2D):
        pass

    # TODO: implement
    def merge_seams(self, p: Vertex2D):
        pass

    # TODO: implement
    def cut_piece(self, p: Vertex2D, q: Vertex2D, inserted_seams: List[Seam]):
        pass

    # TODO: implement
    def replace_seam(
        self, old_seam: Seam, p: Vertex2D, q: Vertex2D, inserted_seams: List[Seam]
    ):
        pass

    @staticmethod
    def reverse_seams(seams: List[Seam]) -> List[Seam]:
        reversed_seams: List[Seam] = []
        for i in range(len(seams) - 1, -1, -1):
            seam: Seam = seams[i]
            reversed_seams.append(seam.reverse())
        return reversed_seams

    def duplicate_seams(self, original_seams: List[Seam]) -> List[Seam]:
        new_seams: List[Seam] = []
        for seam in original_seams:
            new_seams.append(seam.duplicate())
        return new_seams

    # TODO: implement
    def save(self):
        pass

    @staticmethod
    def mean_position(piece: Piece) -> Vertex2D:
        x_sum = 0
        y_sum = 0
        if len(piece.seams) == 0:
            return Vertex2D(0, 0)
        for seam in piece.seams:
            x_sum += seam.start.x
            y_sum += seam.start.y
        return Vertex2D(x_sum / len(piece.seams), y_sum / len(piece.seams))

    def min_y(self) -> float:
        if len(self.seams) == 0:
            return 0
        min_y_value = min([seam.start.y for seam in self.seams])
        return min_y_value

    def max_y(self) -> float:
        if len(self.seams) == 0:
            return 0
        max_y_value = max([seam.start.y for seam in self.seams])
        return max_y_value

    def min_x(self) -> float:
        if len(self.seams) == 0:
            return 0
        min_x_value = min([seam.start.x for seam in self.seams])
        return min_x_value

    def max_x(self) -> float:
        if len(self.seams) == 0:
            return 0
        max_x_value = max([seam.start.x for seam in self.seams])
        return max_x_value

    def mean_x(self) -> float:
        if len(self.seams) == 0:
            return 0
        x_sum = sum([seam.start.x for seam in self.seams])
        return x_sum / len(self.seams)

    def mean_y(self) -> float:
        if len(self.seams) == 0:
            return 0
        y_sum = sum([seam.start.y for seam in self.seams])
        return y_sum / len(self.seams)

    @staticmethod
    def get_stroke(seams: List[Seam]) -> List[Vertex2D]:
        stroke: List[Vertex2D] = []
        for seam in seams:
            if seam.stroke is not None:
                seam.set_stroke()
                stroke.extend(seam.stroke)
                stroke.pop()
        return stroke

    # TODO: the name of this method is different from the original
    def get_stroke_self(self, unit_length: int) -> List[Vertex2D]:
        self.seam_to_stroke_indices: Dict[Seam, List[int]] = {}
        stroke: List[Vertex2D] = []
        for seam in self.seams:
            if seam.stroke is None:
                seam.set_stroke()
            resampled_stroke: List[Vertex2D] = Piece.resample(seam.stroke, unit_length)
            n: int = len(stroke)
            indices: List[int] = [i + n for i in range(len(resampled_stroke))]
            self.seam_to_stroke_indices[seam] = indices
            stroke.extend(resampled_stroke[:-1])
        n: int = len(stroke)
        for indices in self.seam_to_stroke_indices.values():
            assert indices[-1] == n
            indices[-1] = 0

        return stroke

    @staticmethod
    def resample(stroke: List[Vertex2D], unit_length: int) -> List[Vertex2D]:
        if len(stroke) < 2:
            return stroke

        stroke_length: float = Piece.get_length(stroke)
        n: int = int(stroke_length / unit_length + 0.5)
        return Piece.resample_main(stroke, n, stroke_length)

    @staticmethod
    def resample_main(stroke: List[Vertex2D], n: int, stroke_length: float):
        assert n > 0
        unit: float = stroke_length / n
        v0: Vertex2D = stroke[0]
        v1: Vertex2D = stroke[-1]

        resampled_stroke: List[Vertex2D] = []
        resampled_stroke.append(v0)
        total: float = 0
        prev_total: float = 0
        prev: Vertex2D = v0
        next: Optional[Vertex2D] = None
        next_spot: float = 0
        index: int = 1
        count: int = 0
        while True:
            if count == n - 1 or index == len(stroke):
                break
            next = stroke[index]
            total += Vertex2D.distance_static(prev, next)
            while total >= next_spot:
                new_vertex: Vertex2D = Vertex2D.interpolate(
                    prev, next, (next_spot - prev_total) / (total - prev_total)
                )
                resampled_stroke.append(new_vertex)
                next_spot += unit
                count += 1
                if count == n - 1:
                    break
            prev = next
            prev_total = total
            index += 1
        resampled_stroke.append(v1)
        return resampled_stroke

    @staticmethod
    def resample_by_number(stroke: List[Vertex2D], n: int) -> List[Vertex2D]:
        stroke_length: float = Piece.get_length(stroke)
        return Piece.resample_main(stroke, n, stroke_length)

    @staticmethod
    def get_length(stroke: List[Vertex2D]) -> float:
        stroke_length: float = 0
        for i in range(len(stroke) - 1):
            stroke_length += Vertex2D.distance_static(stroke[i], stroke[i + 1])
        return stroke_length

    def get_stroke_indices(self, seam: Seam) -> List[int]:
        return self.seam_to_stroke_indices[seam]

    def get_updated_stroke(self) -> List[Vertex2D]:
        stroke: List[Vertex2D] = []
        for seam in self.seams:
            if seam.stroke is None:
                seam.set_stroke()
            indices: List[int] = self.get_stroke_indices(seam)
            resampled: List[Vertex2D] = Piece.resample(seam.stroke, len(indices) - 1)
            stroke.extend(resampled[:-1])
        return stroke

    @staticmethod
    def visualize_pieces(
        pieces: List[Piece],
        piece_to_constraints: Optional[Dict[Piece, Dict[Vertex2D, Vertex2D]]] = None,
        use_points: bool = True,
        show_markers: bool = True,
        output_file_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        # ax.axis("off")

        if not show_markers:
            marker = "None"
        else:
            marker = "o"

        constraint_marker = "X"

        colors = ["dodgerblue", "fuchsia", "lime", "orange"]
        color_index = 0

        for piece in pieces:
            color_index = 1 if piece.reversed else 0
            if piece_to_constraints is not None:
                constraints = piece_to_constraints[piece]
                for point in constraints.values():
                    ax.plot(
                        point.x,
                        point.y,
                        marker=constraint_marker,
                        linestyle="-",
                        color=Seam.constraint_color,
                        alpha=1.0,
                        markersize=7,
                    )
            min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
            for seam in piece.seams:
                linestyle = Seam.boundary_types_to_linestyle[seam.type]
                linecolor = Seam.boundary_types_to_color[seam.type]
                start = seam.start
                end = seam.end
                min_x = min(min_x, start.x, end.x)
                max_x = max(max_x, start.x, end.x)
                min_y = min(min_y, start.y, end.y)
                max_y = max(max_y, start.y, end.y)
                if use_points and seam.points:
                    vec: Vertex2D = Vertex2D(start, end)
                    vec_rotated: Vertex2D = Vertex2D.rotate(vec, 90)
                    prev_x: float = start.x
                    prev_y: float = start.y
                    for point in seam.points:
                        t: float = point.x
                        d: float = point.y
                        if -1 < t < 1 and -1 < d < 1:
                            point_x: float = start.x + vec.x * t + vec_rotated.x * d
                            point_y: float = start.y + vec.y * t + vec_rotated.y * d
                        else:
                            point_x: float = point.x
                            point_y: float = point.y
                        # ax.plot(
                        #     [prev_x, point_x],
                        #     [prev_y, point_y],
                        #     marker=marker,
                        #     linestyle=linestyle,
                        #     color=linecolor,
                        #     alpha=1.0,
                        # )
                        # prev_x = point_x
                        # prev_y = point_y
                    # ax.plot(
                    #     [prev_x, end.x],
                    #     [prev_y, end.y],
                    #     marker=marker,
                    #     linestyle=linestyle,
                    #     color=linecolor,
                    #     alpha=1.0,
                    # )
                else:
                    # ax.plot(
                    #     [start.x, end.x],
                    #     [start.y, end.y],
                    #     marker=marker,
                    #     linestyle=linestyle,
                    #     # color=colors[color_index],
                    #     color=linecolor,
                    #     alpha=1.0,
                    # )
                    pass
        ax.set_xticks(
            range(
                int(min_x) - 1,
                int(max_x) + 2,
            ),
            minor=True,
        )
        ax.set_yticks(
            range(
                int(min_x) - 1,
                int(max_x) + 2,
            ),
            minor=True,
        )
        if output_file_path is not None:
            fig.savefig(output_file_path)
            fig.savefig(output_file_path.replace(".png", ".svg"), format="svg")
        else:
            plt.show()
        # close the figure to avoid memory leak
        plt.close(fig)


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

    def set_constraints(self, constraints: Dict[Vertex2D, Vertex2D]):
        self.constraints = constraints

    def duplicate(self, new_piece: Piece) -> TemplatePiece:
        new_templatepiece: TemplatePiece = TemplatePiece()
        new_templatepiece.piece = new_piece
        new_templatepiece.template = self.template
        new_piece.template_piece = new_templatepiece
        new_templatepiece.update_points()
        return new_templatepiece

    # TODO: implement
    def drawPanel_to_templatePanel(self, v: Vertex2D) -> Vertex2D:
        # TODO: implement DrawPanel
        # CENTER_X: float = DrawPanel.drawPanel.getWidth()/2
        # CENTER_Y: float = DrawPanel.drawPanel.getHeight()/2
        CENTER_X: float = DrawPanel.get_width() / 2
        CENTER_Y: float = DrawPanel.get_height() / 2
        x: float = v.x - CENTER_X + Template.W / 2
        y: float = v.y - CENTER_Y + Template.W / 2
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
            Mesh2D(faces, integrate_adjascent_face_edges=True)
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

    # TODO: implement
    def paint(self, g):
        pass

    # TODO: implement
    def paint_inside(self, g):
        pass

    # TODO: implement
    def paint_boundary(self, g, color):
        pass

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

    @staticmethod
    def visualize_templates(
        pieces: List[Piece],
        piece_to_constraints: Optional[Dict[Piece, Dict[Vertex2D, Vertex2D]]] = None,
        show_markers: bool = True,
    ):
        Piece.visualize_templates(
            pieces, piece_to_constraints=piece_to_constraints, show_markers=show_markers
        )


class DrawPanel:
    width: int = 512
    height: int = 512

    def __init__(self, pieces: List[Piece] = []):
        self.pieces: List[Piece] = pieces

    @staticmethod
    def get_width():
        return DrawPanel.width

    @staticmethod
    def get_height():
        return DrawPanel.height

    def get_layer_pieces(
        self, reverse_for_pick: bool = False
    ) -> List[Optional[List[Piece]]]:
        layer_pieces: List[Optional[List[Piece]]] = [None] * 4
        for i in range(4):
            layer_pieces[i] = []
        for piece in self.pieces:
            layer_pieces[piece.layer].append(piece)

        if reverse_for_pick:
            for i in range(4):
                layer_pieces[i].reverse()
        return layer_pieces

    def load(self, filename: str, template_panel: TemplatePanel):
        pieces: List[Piece] = File.load(filename)
        template_panel.load_uv_pieces(pieces)


class Face2D:
    def __init__(self, *edges):
        self.index: Optional[int] = None
        self.grid_x: int = 0
        self.grid_y: int = 0
        self.T: List[List[float]] = [
            [0.0, 0.0],
            [0.0, 0.0],
        ]  # Placeholder for transformation matrix
        self.piece: float = 0.0
        self.inside: int = 0  # Placeholder for inside/outside flag
        self.H_EDGE0: Optional[Vector2] = None
        self.V_EDGE0: Optional[Vector2] = None
        self.H_EDGE1: Optional[Vector2] = None
        self.V_EDGE1: Optional[Vector2] = None
        if len(edges) > 0:
            self.edges: List[Edge2D] = list(edges)
            for i, edge in enumerate(edges):
                self.add_edge(i, edge, edges[(i + 1) % len(edges)])

    def __str__(self):
        return f"Face2D(inside: {self.inside}, x: {self.grid_x}, y: {self.grid_y}, {self.edges})"

    def __repr__(self):
        return self.__str__()

    def add_edge(self, i: int, e: Edge2D, next_edge: Edge2D):
        self.edges[i] = e
        if next_edge.contains(e.end):
            e.left_face = self
        else:
            e.right_face = self

    def edges_method(self, i: int) -> Edge2D:
        # Util.mod is not a built-in Python function, so using % operator for modulus
        return self.edges[i % len(self.edges)]

    def get_vertex(self, i: int) -> Vertex2D:
        return self.edges_method(i - 1).get_common_vertex(self.edges_method(i))

    def get_vertices(self) -> List[Vertex2D]:
        return [self.get_vertex(i) for i in range(len(self.edges))]

    def get_center(self) -> Vertex2D:
        center = Vertex2D()
        for vertex in self.get_vertices():
            center.x += vertex.x
            center.y += vertex.y
        center.x /= len(self.edges)
        center.y /= len(self.edges)
        return center

    def get_common_edge(self, next_face: Face2D) -> Optional[Edge2D]:
        for edge in self.edges:
            if edge in next_face.edges:
                return edge
        return None

    @staticmethod
    def get_boundary_edges(faces: List[Face2D]) -> List[Edge2D]:
        boundary_edges: List[Edge2D] = []
        checked_edges: List[Edge2D] = []
        for face in faces:
            for edge in face.edges:
                if edge in checked_edges:
                    continue
                if edge.is_boundary():
                    if edge.left_face.inside == 1:
                        boundary_edges.append(edge.shallow_duplicate())
                    else:
                        boundary_edges.append(edge.shallow_duplicate_reverse())
                    if edge.is_cut():
                        boundary_edges.append(
                            boundary_edges[-1].shallow_duplicate_reverse()
                        )
                    checked_edges.append(edge)
        return boundary_edges

    @staticmethod
    def count_zigzags(faces: List[Edge2D]) -> int:
        boundary_edges: List[Edge2D] = Face2D.get_boundary_edges(faces)
        zigzags = 0
        while boundary_edges:
            closed_path: List[Edge2D] = Template.find_closed_stroke(boundary_edges)
            assert len(closed_path) > 0, "Closed path should not be empty"
            prev_edge: Edge2D = closed_path[0]
            for i in range(0, len(closed_path)):
                edge: Edge2D = closed_path[(i + 1) % len(closed_path)]
                if prev_edge.start.x != edge.end.x and prev_edge.start.y != edge.end.y:
                    zigzags += 1
                prev_edge = edge
        return zigzags

    @staticmethod
    def extract_inside_edges_non_none_seam_type_from_faces(
        faces: List[Face2D],
    ) -> List[Edge2D]:
        edges: List[Edge2D] = []
        for face in faces:
            for edge in face.edges:
                if (edge.left_face is not None and edge.left_face.inside == 1) and (
                    edge.right_face is not None and edge.right_face.inside == 1
                ):
                    if edge.seam_type is not None and edge.seam_type != Seam.NONE:
                        edges.append(edge)
        return edges
