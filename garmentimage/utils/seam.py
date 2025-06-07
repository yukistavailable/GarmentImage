from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import shapely.geometry as sg

from garmentimage.utils.vertex2d import Vector2, Vertex2D

if TYPE_CHECKING:
    from garmentimage.utils.piece import Piece


class Seam:
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
