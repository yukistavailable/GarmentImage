from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from garmentimage.utils.vector2 import Vector2
    from garmentimage.utils.utils_2d import Edge2D, Seam, Face2D

EdgeInfoType = List[Dict[str, Union[np.ndarray, Dict, int]]]
PanelToEdgeInfoType = Dict[str, EdgeInfoType]
Number = Union[int, float, np.floating]
GarmentImageType = Union[str, np.ndarray, torch.Tensor]


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
