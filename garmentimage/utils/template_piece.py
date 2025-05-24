from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from garmentimage.utils.draw_panel import DrawPanel
    from garmentimage.utils.face import Face2D
    from garmentimage.utils.piece import Piece
    from garmentimage.utils.seam import Seam
    from garmentimage.utils.template import Mesh2D, Template
    from garmentimage.utils.vector2 import Vector2
    from garmentimage.utils.vertex2d import Vertex2D


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
