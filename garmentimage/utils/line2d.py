from __future__ import annotations

from garmentimage.utils.vertex2d import Vector2, Vertex2D


class Line2D:
    def __init__(self, _base: Vertex2D, _direction: Vector2):
        self.base: Vertex2D = _base
        self.direction: Vector2 = Vector2.normalize(_direction)

    def distance(self, v: Vertex2D) -> float:
        vec: Vector2 = Vector2(self.base, v)
        return abs(Vector2.cross_product(self.direction, vec))

    @staticmethod
    def cross_point(v: Line2D, u: Line2D):
        v0: Vertex2D = v.base
        v1: Vector2 = v.direction
        u0: Vertex2D = u.base
        u1: Vector2 = u.direction

        a: float = v1.x
        b: float = -u1.x
        c: float = v1.y
        d: float = -u1.y
        e: float = u0.x - v0.x
        f: float = u0.y - v0.y

        det: float = a * d - b * c
        if det == 0:
            return None

        _a: float = d / det
        _b: float = -b / det
        _c: float = -c / det
        _d: float = a / det
        t: float = _a * e + _b * f
        s: float = _c * e + _d * f
        x: float = v0.x + v1.x * t
        # TODO: check the followig implementation
        y: float = v0.y + v1.y * s
        return Vertex2D(x, y)
