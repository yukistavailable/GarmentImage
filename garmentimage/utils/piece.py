from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from garmentimage.utils.seam import Seam
    from garmentimage.utils.template_piece import TemplatePiece
    from garmentimage.utils.vector2 import Vector2
    from garmentimage.utils.vertex2d import Vertex2D


class Piece:
    def __init__(self, seams: Optional[List[Seam]]) -> None:
        self.template_piece: Optional[TemplatePiece] = None
        self.original_piece: Optional[Piece] = None
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

    def get_all_seams(self) -> List[Seam]:
        all_seams: List[Seam] = []
        if self.seams is not None:
            all_seams.extend(self.seams)
        if self.inner_seams is not None:
            all_seams.extend(self.inner_seams)
        return all_seams

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
    def get_length(stroke: List[Vertex2D]) -> float:
        stroke_length: float = 0
        for i in range(len(stroke) - 1):
            stroke_length += Vertex2D.distance_static(stroke[i], stroke[i + 1])
        return stroke_length

    def get_stroke_indices(self, seam: Seam) -> List[int]:
        return self.seam_to_stroke_indices[seam]

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
        constraint_marker = "X"

        for piece in pieces:
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
