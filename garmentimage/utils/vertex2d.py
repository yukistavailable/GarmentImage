from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from garmentimage.utils.utils import Number

if TYPE_CHECKING:
    from garmentimage.utils.edge2d import Edge2D
    from garmentimage.utils.face import Face2D
    from garmentimage.utils.seam import Seam


class Vector2:
    def __init__(
        self,
        start: Optional[Union[Vector2, Vertex2D, float, int]] = None,
        end: Optional[Union[Vector2, Vertex2D, float, int]] = None,
    ):
        self.x: float
        self.y: float
        if isinstance(start, Vector2):
            if end is None:
                self.x = start.x
                self.y = start.y
            else:
                assert isinstance(end, Vector2)
                self.x = end.x - start.x
                self.y = end.y - start.y
        elif isinstance(start, Vertex2D):
            if end is None:
                self.x = start.x
                self.y = start.y
            else:
                assert isinstance(end, Vertex2D)
                self.x = end.x - start.x
                self.y = end.y - start.y
        else:
            assert isinstance(start, Number), type(start)
            assert isinstance(end, Number), type(end)
            self.x = float(start)
            self.y = float(end)

    def __str__(self) -> str:
        return f"Vector2: ({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Vector2: ({self.x}, {self.y})"

    def duplicate(self):
        return Vector2(self.x, self.y)

    @staticmethod
    def normalize(v: Vector2) -> Vector2:
        l = v.length()
        if l != 0:
            return Vector2(v.x / l, v.y / l)
        else:
            # TODO: Rethink this
            return Vector2(0, 0)

    # special methods
    def __add__(self, other: Vector2):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, m: float):
        return Vector2(self.x * m, self.y * m)

    # TODO: the name of this method is different from the original
    def normalize_self(self):
        l = self.length()
        if l != 0:
            self.x /= l
            self.y /= l

    def negate(self):
        self.x *= -1
        self.y *= -1

    # TODO: the name of this method is different from the original
    @staticmethod
    def negate_static(v: Vector2):
        return Vector2(-v.x, -v.y)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    @staticmethod
    def add(u: Vector2, v: Vector2):
        return Vector2(u.x + v.x, u.y + v.y)

    # TODO: the name of this method is different from the original
    def add_self(self, v: Vector2):
        self.x += v.x
        self.y += v.y

    @staticmethod
    def subtract(u: Vector2, v: Vector2):
        return Vector2(u.x - v.x, u.y - v.y)

    # TODO: the name of this method is different from the original
    def subtract_self(self, v: Vector2):
        self.x -= v.x
        self.y -= v.y

    @staticmethod
    def multiply(v: Vector2, m: float):
        return Vector2(v.x * m, v.y * m)

    # TODO: the name of this method is different from the original
    def multiply_self(self, m: float):
        self.x *= m
        self.y *= m

    @staticmethod
    def dot_product(u: Vector2, v: Vector2) -> float:
        return u.x * v.x + u.y * v.y

    @staticmethod
    def cross_product(u: Vector2, v: Vector2) -> float:
        return u.x * v.y - u.y * v.x

    @staticmethod
    def cos(u: Vector2, v: Vector2):
        length = u.length() * v.length()
        return Vector2.dot_product(u, v) / length if length > 0 else 0

    @staticmethod
    def sin(u: Vector2, v: Vector2):
        length = u.length() * v.length()
        return Vector2.cross_product(u, v) / length if length > 0 else 0

    @staticmethod
    def get_angle_PI(u: Vector2, v: Vector2):
        cosine = Vector2.cos(u, v)
        return (
            math.acos(cosine) if -1 < cosine < 1 else (math.pi if cosine <= -1 else 0)
        )

    @staticmethod
    def get_angle_360(u: Vector2, v: Vector2):
        cos = Vector2.cos(u, v)
        sin = Vector2.sin(u, v)
        angle = math.degrees(math.atan2(sin, cos))
        return angle if angle >= 0 else angle + 360

    def get_angle_signed_180(u: Vector2, v: Vector2):
        cos_value = Vector2.cos(u, v)
        sin_value = Vector2.sin(u, v)

        if cos_value == 0:
            if sin_value > 0:
                return 90
            else:
                return -90
        if sin_value == 0:
            if cos_value > 0:
                return 0
            else:
                return 180

        angle = 180 * math.atan(sin_value / cos_value) / math.pi

        if angle > 0:
            if cos_value < 0:
                angle -= 180
        else:
            if cos_value < 0:
                angle += 180

        return angle

    def get_angle180(self, other: Vector2):
        cos_angle = self.cos(other)
        sin_angle = self.sin(other)

        if cos_angle == 0:
            if sin_angle > 0:
                return 90
            else:
                return -90
        if sin_angle == 0:
            if cos_angle > 0:
                return 0
            else:
                return 180

        angle = 180 * math.atan(sin_angle / cos_angle) / math.pi

        if cos_angle < 0:
            if angle > 0:
                angle -= 180
            else:
                angle += 180

        return angle

    @staticmethod
    def rotate90(v: Vector2):
        return Vector2(v.y, -v.x)

    @staticmethod
    def rotate(v: Vector2, degree: float):
        if degree == 0:
            return v
        elif degree == 90:
            return Vector2(-v.y, v.x)
        elif degree == 180:
            return Vector2(-v.x, -v.y)
        elif degree == 270:
            return Vector2(v.y, -v.x)

        radian = math.radians(degree)
        cos = math.cos(radian)
        sin = math.sin(radian)
        return Vector2(v.x * cos - v.y * sin, v.x * sin + v.y * cos)

    @staticmethod
    def distance(x1: int, y1: int, x2: int, y2: int):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class Vertex2D(Vector2):
    def __init__(
        self,
        x: Optional[Union[float, Vector2]] = None,
        y: Optional[Union[float, Vector2]] = None,
        index: Optional[int] = None,
        fixed: bool = False,
    ):
        self.x: float
        self.y: float
        if x is None and y is None:
            self.x = 0
            self.y = 0
        elif isinstance(x, Number):
            assert isinstance(y, Number), type(y)
            self.x = x
            self.y = y
        elif isinstance(x, Vertex2D):
            if y is None:
                self.x = x.x
                self.y = x.y
            else:
                assert isinstance(y, Vertex2D)
                self.x = y.x - x.x
                self.y = y.y - x.y
        else:
            assert isinstance(x, Vector2)
            if y is None:
                self.x = x.x
                self.y = x.y
            else:
                assert isinstance(y, Vector2)
                self.x = y.x - x.x
                self.y = y.y - x.y

        self.index: Optional[int] = index
        self.fixed: bool = fixed
        self.edges: List[Edge2D] = []
        self.prev_seam: Optional[Seam] = None
        self.next_seam: Optional[Seam] = None
        self.uv: Optional[Vertex2D] = None
        self.corner: Optional[Vertex2D] = None
        self.grid_xy: List[int] = [0, 0]
        self.TX: Optional[Vector2] = None
        self.TY: Optional[Vector2] = None
        self.X_EDGE_TYPE: int = 0
        self.Y_EDGE_TYPE: int = 0

    def update(self, x: float, y: float):
        self.x = x
        self.y = y

    def to_np(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def to_np_four_dim(self) -> np.ndarray:
        return np.array([self.x, self.y, 0, 1])

    @staticmethod
    def from_np(array: np.ndarray) -> Vertex2D:
        return Vertex2D(array[0], array[1])

    def __str__(self) -> str:
        return f"Vertex2D: ({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Vertex2D: ({self.x}, {self.y})"

    # special methods
    def __add__(self, other: Vector2):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, m: float):
        return Vector2(self.x * m, self.y * m)

    @staticmethod
    def encloses(v: Vertex2D, path: List[Vertex2D], is_reversed: bool = False) -> bool:
        """
        Checks if v is within the closed path
        """
        sign: int = -1 if is_reversed else 1
        total: float = 0
        for i in range(len(path)):
            v0: Vertex2D = path[i % len(path)]
            v1: Vertex2D = path[(i + 1) % len(path)]
            vec0: Vector2 = Vector2(v, v0)
            vec1: Vector2 = Vector2(v, v1)
            total += Vector2.get_angle_signed_180(vec0, vec1)
        total *= sign
        return total > 180

    def get_top_edge(self) -> Optional[Edge2D]:
        for edge in self.edges:
            if edge.start == self and self.y < edge.end.y:
                return edge
        return None

    def get_bottom_edge(self) -> Optional[Edge2D]:
        for edge in self.edges:
            if edge.end == self and self.y > edge.end.y:
                return edge
        return None

    def get_right_edge(self) -> Optional[Edge2D]:
        for edge in self.edges:
            if edge.start == self and self.x < edge.end.x:
                return edge
        return None

    def get_left_edge(self) -> Optional[Edge2D]:
        for edge in self.edges:
            if edge.end == self and self.x > edge.start.x:
                return edge
        return None

    def get_edges(self) -> List[Edge2D]:
        return self.edges

    def get_common_seam(self, v: Vertex2D) -> Optional[Seam]:
        if v.next_seam == self.next_seam or v.prev_seam == self.next_seam:
            return self.next_seam
        elif v.next_seam == self.prev_seam or v.prev_seam == self.prev_seam:
            return self.prev_seam
        else:
            return None

    def get_opposite_seam(self, seam: Seam) -> Seam:
        if seam == self.next_seam:
            return self.prev_seam
        elif seam == self.prev_seam:
            return self.next_seam
        else:
            raise ValueError("The seam is not connected to the vertex")

    def copy(self) -> Vertex2D:
        new_v = Vertex2D(self.x, self.y)
        new_v.index = self.index
        new_v.fixed = self.fixed
        return new_v

    def duplicate(self) -> Vertex2D:
        return self.copy()

    @staticmethod
    def same_position(a: Vertex2D, b: Vertex2D):
        return a.x == b.x and a.y == b.y

    @staticmethod
    def mid_point(a: Vertex2D, b: Vertex2D) -> float:
        return Vertex2D((a.x + b.x) / 2, (a.y + b.y) / 2)

    @staticmethod
    def translate(u: Vertex2D, v: Vertex2D):
        return Vertex2D(u.x + v.x, u.y + v.y)

    def same(self, v: Vertex2D) -> bool:
        # Assuming Def.equal is a method for checking equality with some tolerance
        return self.x == v.x and self.y == v.y

    def distance(self, node: Vertex2D) -> float:
        return math.sqrt((node.x - self.x) ** 2 + (node.y - self.y) ** 2)

    def warp(self, v: Vertex2D):
        self.x = v.x
        self.y = v.y

    @staticmethod
    def distance_static(n1: Vertex2D, n2: Vertex2D) -> float:
        return math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)

    @staticmethod
    def interpolate(start: Vertex2D, end: Vertex2D, t: float) -> Vertex2D:
        return Vertex2D(start.x * (1 - t) + end.x * t, start.y * (1 - t) + end.y * t)

    def get_common_edge(self, v: Vertex2D) -> Optional[Edge2D]:
        for edge in self.edges:
            if edge.start == v or edge.end == v:
                return edge
        return None

    def get_surrounding_vertices(self) -> List[Vertex2D]:
        vertices = []
        for edge in self.edges:
            vertices.append(edge.get_the_other_vertex(self))
        return vertices

    def get_adjacent_faces(self) -> List[Face2D]:
        faces = set()
        for edge in self.edges:
            if edge.left_face:
                faces.add(edge.left_face)
            if edge.right_face:
                faces.add(edge.right_face)
        return faces
