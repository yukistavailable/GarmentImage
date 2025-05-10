from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from garmentimage.utils.vertex2d import Vertex2D

EdgeInfoType = List[Dict[str, Union[np.ndarray, Dict, int]]]
PanelToEdgeInfoType = Dict[str, EdgeInfoType]
Number = Union[int, float, np.floating]
GarmentImageType = Union[str, np.ndarray, torch.Tensor]


class Vector2:
    def __init__(
        self,
        start: Optional[Union[Vector2, Vertex2D, float, int]] = None,
        end: Optional[Union[Vector2, Vertex2D, float, int]] = None,
    ):
        self.x: float
        self.y: float
        if isinstance(start):
            if end is None:
                self.x = start.x
                self.y = start.y
            else:
                assert isinstance(end)
                self.x = end.x - start.x
                self.y = end.y - start.y
        elif isinstance(start, Vector2):
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
