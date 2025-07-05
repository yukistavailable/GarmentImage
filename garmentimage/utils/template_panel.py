from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from garmentimage.utils.errors import SeamTypeMismatchError
from garmentimage.utils.template import Template, TemplatePiece
from garmentimage.utils.vertex2d import Vector2, Vertex2D

if TYPE_CHECKING:
    from garmentimage.utils.draw_panel import DrawPanel
    from garmentimage.utils.edge2d import Edge2D
    from garmentimage.utils.encoder import Encoder
    from garmentimage.utils.face import Face2D
    from garmentimage.utils.piece import Piece
    from garmentimage.utils.seam import Seam


class TemplatePanel:
    def __init__(self):
        # assume four templates: front, back, above front, below back
        self.templates: List[Template] = [
            Template(),
            Template(),
            Template(),
            Template(),
        ]

    def convert_to_np_array(self) -> np.ndarray:
        templates: List[Template] = self.templates
        img: np.ndarray = np.zeros((34, Template.N, Template.N))
        for i, template in enumerate(templates):
            if i == 0 or i == 1:
                for x in range(Template.N):  # Assuming Template.N is the grid size
                    for y in range(Template.N):
                        face: Face2D = template.grid_faces[x][y]
                        v: Vertex2D = template.grid_vertices[x][y]
                        is_inside = 1 if face.inside == 1 else 0
                        ch_idx = 0 if i == 0 else 17
                        img[ch_idx + 0][x][y] = is_inside

                        # Normalize h_edge0 and v_edge0 vectors
                        h_edge0 = Vector2(face.H_EDGE0)
                        h_edge1 = Vector2(face.H_EDGE1)
                        v_edge0 = Vector2(face.V_EDGE0)
                        v_edge1 = Vector2(face.V_EDGE1)

                        h_edge0.subtract_self(Vector2(Template.W / Template.N, 0))
                        h_edge0.multiply_self(1.0 / (Template.W / Template.N))
                        h_edge1.subtract_self(Vector2(Template.W / Template.N, 0))
                        h_edge1.multiply_self(1.0 / (Template.W / Template.N))

                        v_edge0.subtract_self(Vector2(0, Template.W / Template.N))
                        v_edge0.multiply_self(1.0 / (Template.W / Template.N))
                        v_edge1.subtract_self(Vector2(0, Template.W / Template.N))
                        v_edge1.multiply_self(1.0 / (Template.W / Template.N))

                        # Write four floats
                        img[ch_idx + 1][x][y] = h_edge0.x
                        img[ch_idx + 2][x][y] = h_edge0.y
                        img[ch_idx + 3][x][y] = v_edge0.x
                        img[ch_idx + 4][x][y] = v_edge0.y
                        img[ch_idx + 5][x][y] = h_edge1.x
                        img[ch_idx + 6][x][y] = h_edge1.y
                        img[ch_idx + 7][x][y] = v_edge1.x
                        img[ch_idx + 8][x][y] = v_edge1.y

                        # one-hot encoding
                        img[ch_idx + 9 + v.X_EDGE_TYPE][x][y] = 1
                        img[ch_idx + 13 + v.Y_EDGE_TYPE][x][y] = 1
        return img

    def convert_from_np_array(
        self, img: np.ndarray, inside_threshold: float = 0.5
    ) -> None:
        self.init_templates()
        for i, template in enumerate(self.templates):
            if i == 0 or i == 1:
                # TODO: rethink if we need to consider grid_vertices[N+1][N+1]
                for x in range(Template.N):
                    for y in range(Template.N):
                        face: Face2D = template.grid_faces[x][y]
                        v: Vertex2D = template.grid_vertices[x][y]
                        ch_idx = 0 if i == 0 else 17
                        # TODO: check if the following implementation is correct
                        _inside = img[ch_idx + 0][x][y]
                        face.inside = 1 if _inside > inside_threshold else 0

                        h_edge0 = Vector2(img[ch_idx + 1][x][y], img[ch_idx + 2][x][y])
                        h_edge0.add_self(Vector2(1, 0))
                        h_edge0.multiply_self(Template.W / Template.N)
                        face.H_EDGE0 = h_edge0

                        h_edge1 = Vector2(img[ch_idx + 5][x][y], img[ch_idx + 6][x][y])
                        h_edge1.add_self(Vector2(1, 0))
                        h_edge1.multiply_self(Template.W / Template.N)
                        face.H_EDGE1 = h_edge1

                        v_edge0 = Vector2(img[ch_idx + 3][x][y], img[ch_idx + 4][x][y])
                        v_edge0.add_self(Vector2(0, 1))
                        v_edge0.multiply_self(Template.W / Template.N)
                        face.V_EDGE0 = v_edge0

                        v_edge1 = Vector2(img[ch_idx + 7][x][y], img[ch_idx + 8][x][y])
                        v_edge1.add_self(Vector2(0, 1))
                        v_edge1.multiply_self(Template.W / Template.N)
                        face.V_EDGE1 = v_edge1

                        one_hot_vector_x_edge = img[ch_idx + 9 : ch_idx + 13, x, y]
                        one_hot_vector_y_edge = img[ch_idx + 13 : ch_idx + 17, x, y]
                        v.X_EDGE_TYPE = np.argmax(one_hot_vector_x_edge)
                        v.Y_EDGE_TYPE = np.argmax(one_hot_vector_y_edge)
                        right_edge: Edge2D = v.get_right_edge()
                        if right_edge is not None:
                            right_edge.seam_type = v.X_EDGE_TYPE
                        top_edge: Edge2D = v.get_top_edge()
                        if top_edge is not None:
                            top_edge.seam_type = v.Y_EDGE_TYPE

    def check_seam_type_match(self) -> bool:
        front_template: Template = self.templates[0]
        back_template: Template = self.templates[1]
        if not TemplatePanel.check_seam_type_match_front_to_back(
            front_template, back_template
        ):
            return False
        return True

    @staticmethod
    def check_seam_type_match_front_to_back(
        front_template: Template, back_template: Template
    ) -> bool:
        for x in range(Template.N):
            for y in range(Template.N):
                front_v, back_v = (
                    front_template.grid_vertices[x][y],
                    back_template.grid_vertices[x][y],
                )

                # Check both X and Y edge types for both front and back vertices
                if (
                    (
                        front_v.X_EDGE_TYPE == Seam.FRONT_TO_BACK
                        and back_v.X_EDGE_TYPE != Seam.FRONT_TO_BACK
                    )
                    or (
                        front_v.Y_EDGE_TYPE == Seam.FRONT_TO_BACK
                        and back_v.Y_EDGE_TYPE != Seam.FRONT_TO_BACK
                    )
                    or (
                        back_v.X_EDGE_TYPE == Seam.FRONT_TO_BACK
                        and front_v.X_EDGE_TYPE != Seam.FRONT_TO_BACK
                    )
                    or (
                        back_v.Y_EDGE_TYPE == Seam.FRONT_TO_BACK
                        and front_v.Y_EDGE_TYPE != Seam.FRONT_TO_BACK
                    )
                ):
                    return False

        return True

    def init_templates(self):
        self.templates: List[Template] = [
            Template(),
            Template(),
            Template(),
            Template(),
        ]

    def load_uv_pieces(self, pieces: List[Piece]):
        for piece in pieces:
            template: Template = (
                self.templates[0] if not piece.reversed else self.templates[1]
            )
            piece.template_piece = TemplatePiece(piece, template)
            template.template_pieces.append(piece.template_piece)
            for seam in piece.get_all_seams():
                seam.start.corner = Template.convert_relative_to_absolute(seam.start.uv)
                seam.end.corner = Template.convert_relative_to_absolute(seam.end.uv)
            piece.template_piece.update_points()
            piece.template_piece.linked = False

    def load_pieces(
        self,
        pieces: List[Piece],
        update_corners: bool = True,
        update_points: bool = True,
    ):
        for piece in pieces:
            template: Template = (
                self.templates[0] if not piece.reversed else self.templates[1]
            )
            piece.template_piece = TemplatePiece(
                piece, template, update_corners=update_corners
            )
            if update_corners:
                for seam in piece.get_all_seams():
                    seam.start.corner = (
                        Vertex2D(seam.start.uv)
                        if seam.start.uv is not None
                        else seam.start
                    )
                    seam.end.corner = (
                        Vertex2D(seam.end.uv) if seam.end.uv is not None else seam.end
                    )
            if update_points:
                piece.template_piece.update_points()
            piece.template_piece.linked = False

    def add_piece(self, piece: Piece, is_reversed: bool):
        template: Template = self.templates[0] if not is_reversed else self.templates[1]
        template_piece: TemplatePiece = TemplatePiece(piece, template)
        template_piece.update_corners()

    def encode(
        self,
        draw_panel: DrawPanel,
        encoder: Encoder,
        draw_panel_original_shape: Optional[DrawPanel] = None,
        strict_seam_type_match: bool = False,
    ):
        for template in self.templates:
            template.clear_meshes()
        layer_pieces: List[Optional[List[Piece]]] = draw_panel.get_layer_pieces(False)
        layer_pieces_original_shape: Optional[List[Optional[List[Piece]]]] = (
            None
            if draw_panel_original_shape is None
            else draw_panel_original_shape.get_layer_pieces(False)
        )
        for i in range(2):
            # i = 1
            reversed: bool = False if (i == 0 or i == 2) else True
            assert layer_pieces[i] is not None
            encoder.encode_pieces_to_template(
                layer_pieces[i],
                self.templates[i],
                reversed,
                pieces_original_shape=None
                if layer_pieces_original_shape is None
                else layer_pieces_original_shape[i],
                is_seam_reversed=True if i == 1 else False,
            )

        if strict_seam_type_match and not self.check_seam_type_match():
            raise SeamTypeMismatchError(
                "Seam type does not match between front and back templates."
            )
