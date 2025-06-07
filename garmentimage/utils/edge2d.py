from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

from garmentimage.utils.vertex2d import Vector2, Vertex2D

if TYPE_CHECKING:
    from garmentimage.utils.face import Face2D
    from garmentimage.utils.seam import Seam


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

    def mid_point(self) -> Vertex2D:
        return Vertex2D(
            (self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2
        )

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
