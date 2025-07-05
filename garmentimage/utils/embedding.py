from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from garmentimage.utils.face import Face2D
from garmentimage.utils.mesh import Mesh2D
from garmentimage.utils.seam import Seam
from garmentimage.utils.template import Template
from garmentimage.utils.vertex2d import Vector2, Vertex2D

if TYPE_CHECKING:
    from garmentimage.utils.piece import Piece


class SparseMatrixScipy:
    def __init__(self, n_rows: int, n_columns: int):
        self.n_rows: int = n_rows
        self.n_columns: int = n_columns
        # print(n_rows, n_columns)
        self.matrix: csr_matrix = csr_matrix((n_rows, n_columns))
        self.lu_factor: Optional[scipy.sparse.linalg.splu] = None

    # TODO: the following implementation is different from the original code in SparseMatrixUmfpack.add
    def add(self, row: int, col: int, value: float):
        self.matrix[row, col] = value

    def solve(self, Bx, By) -> Tuple[List[float], List[float]]:
        xs = spsolve(self.matrix, Bx)
        ys = spsolve(self.matrix, By)
        return xs, ys


class Embedding:
    use_umfpack: bool = True

    def __init__(self) -> None:
        pass

    @staticmethod
    def embed_seam_type_into_template(
        piece: Piece, faces: List[Face2D], template: Template
    ):
        """
        Parameters
        ----------
        piece : Piece
            The piece whose seams will be embedded into the template.
        faces : List[Face2D]
            The faces of the template that will be used to find the nearest vertices.
        template : Template
            The template into which the seams will be embedded.

        This method iterates through the seams of the piece, finds the nearest vertices on the template faces for each seam point,
        and updates the seam points accordingly. It also updates the edge types in the template's grid vertices based on the seam type.
        The method assumes that the piece has seams defined and that the faces are part of the template's grid structure.
        """
        seams: List[Seam] = piece.seams if piece.seams else []
        for seam in seams:
            assert seam.start.corner is not None and seam.end.corner is not None, (
                "The start and end of the seam are not set. Please set them before embedding."
            )
            new_points = []
            for i in range(len(seam.points) - 1):
                v0: Vertex2D = seam.points[i]
                v1: Vertex2D = seam.points[i + 1]

                # xy0: List[int] = Template.get_grid_xy(nearest_v0_on_faces)
                # xy1: List[int] = Template.get_grid_xy(nearest_v1_on_faces)

                nearest_v0_on_faces = Face2D.find_nearest_vertex_on_faces(faces, v0)
                nearest_v1_on_faces = Face2D.find_nearest_vertex_on_faces(faces, v1)
                xy0 = nearest_v0_on_faces.grid_xy
                xy1 = nearest_v1_on_faces.grid_xy

                # left and bottom
                x: int = min(xy0[0], xy1[0])
                y: int = min(xy0[1], xy1[1])

                if xy0[1] == xy1[1]:
                    template.grid_vertices[x][y].X_EDGE_TYPE = seam.type
                else:
                    template.grid_vertices[x][y].Y_EDGE_TYPE = seam.type

                new_points.append(nearest_v0_on_faces)
            new_points.append(nearest_v1_on_faces)
            seam.points = new_points

        for x in range(Template.N):  # Assuming Template.N defines the grid size
            for y in range(Template.N):
                template.h_edges[x][y].seam_type = template.grid_vertices[x][
                    y
                ].X_EDGE_TYPE
                template.v_edges[x][y].seam_type = template.grid_vertices[x][
                    y
                ].Y_EDGE_TYPE

    @staticmethod
    def embed_transformation_into_template(piece_mesh: Mesh2D, template: Template):
        """
        Parameters
        ----------
        piece_mesh : Mesh2D
            The mesh of the piece whose transformation will be embedded into the template.
        template : Template
            The template into which the piece's transformation will be embedded.
        """
        for face in piece_mesh.faces:
            grid_xy: List[int] = face.get_vertex(0).grid_xy

            v0: Vertex2D = face.get_vertex(0)
            v1: Vertex2D = face.get_vertex(1)
            v2: Vertex2D = face.get_vertex(2)
            v3: Vertex2D = face.get_vertex(3)

            h_edge0: Vector2 = Vector2(v0, v1)
            h_edge1: Vector2 = Vector2(v3, v2)
            v_edge0: Vector2 = Vector2(v0, v3)
            v_edge1: Vector2 = Vector2(v1, v2)

            template_face = template.grid_faces[grid_xy[0]][grid_xy[1]]

            # Direct assignment is commented out in the original Java code
            template_face.H_EDGE0 = h_edge0
            template_face.V_EDGE0 = v_edge0
            template_face.H_EDGE1 = h_edge1
            template_face.V_EDGE1 = v_edge1


class BoundaryEmbedding:
    def __init__(self):
        pass

    @staticmethod
    def embed_v2(
        template: Template, pieces: List[Piece], is_reversed: bool = False
    ) -> Tuple[Dict[Piece, List[Face2D]], Dict[Face2D, List[Piece]]]:
        """
        Parameters
        ----------
        template : Template
            The template to embed the pieces into.
            The faces of the template will be used to determine if a piece is inside a face.
        pieces : List[Piece]
            The pieces to embed into the template.
            Each piece should have its seams defined.
        is_reversed : bool, optional
            If True, the seams will be processed in reverse order.

        Returns
        -------
        Tuple[Dict[Piece, List[Face2D]], Dict[Face2D, List[Piece]]]
            A tuple containing two dictionaries:
            - piece_to_faces: Maps each piece to a list of faces that it contains.
            - face_to_pieces: Maps each face to a list of pieces that contain it.

        This method finds the nearest vertex on the template for each seam start and end point,
        and then constructs the path between them based on the template's grid structure.
        In the end, it assigns faces to pieces based on whether the piece's path encloses the face's center.

        The method assumes that the template has a grid structure defined by its vertices and edges.
        It uses the `Template.get_path_v2` method to find the path between seam start and end points,
        taking into account the seam type and previous direction.
        The method also checks if the last two vertices of the path are adjacent, ensuring that the
        path is valid and follows the grid structure of the template.
        """

        face_to_pieces: Dict[Face2D, List[Piece]] = defaultdict(list)
        piece_to_faces: Dict[Piece, List[Face2D]] = defaultdict(list)

        for piece in pieces:
            seams: List[Seam] = piece.sorted_seams(piece.seams if piece.seams else [])
            for seam in seams:
                start = seam.start
                end = seam.end
                seam.start.corner = Vertex2D(template.find_nearest_vertex(seam.start))
                seam.end.corner = Vertex2D(template.find_nearest_vertex(seam.end))

        piece_to_path: Dict[Piece, List[Vertex2D]] = {}
        for piece in pieces:
            piece_to_path[piece] = []
            seams: List[Seam] = piece.sorted_seams(piece.seams if piece.seams else [])
            if is_reversed:
                seams = list(reversed(seams))
            if seams != []:
                prev_seam_type = None
                prev_start: Optional[Vertex2D] = None
                prev_end: Optional[Vertex2D] = None
                prev_direction: Optional[Template.Direction] = None
                for i, seam in enumerate(seams):
                    if not is_reversed:
                        start = seam.start.corner
                        end = seam.end.corner
                    else:
                        start = seam.end.corner
                        end = seam.start.corner
                    if prev_start is not None and prev_end is not None:
                        if not Vertex2D.same_position(start, prev_end):
                            assert Vertex2D.same_position(end, prev_end), (
                                f"The start and end of the seam are not adjacent. {start}, {end}, {prev_start}, {prev_end}"
                            )
                            start, end = end, start

                    seam_type = seam.type
                    if start is not None and end is not None:
                        path = Template.get_path_v2(
                            start=start,
                            end=end,
                            template=template,
                            prev_seam_type=prev_seam_type,
                            seam_type=seam_type,
                            prev_direction=prev_direction,
                        )
                        prev_seam_type = seam_type
                        prev_seam = seam
                        prev_start = start
                        prev_end = end
                        delta = int(Template.W / Template.N)
                        if len(path) >= 2:
                            last_vertices_diff = path[-1] - path[-2]
                            assert (
                                int(
                                    abs(last_vertices_diff.x)
                                    + abs(last_vertices_diff.y)
                                )
                                == delta
                            ), (
                                f"The last two vertices of the path are not adjacent. The delta is {abs(last_vertices_diff.x) + abs(last_vertices_diff.y)}"
                            )
                            if last_vertices_diff.x == delta:
                                prev_direction = Template.Direction.RIGHT
                            elif last_vertices_diff.x == -delta:
                                prev_direction = Template.Direction.LEFT
                            elif last_vertices_diff.y == delta:
                                prev_direction = Template.Direction.UP
                            elif last_vertices_diff.y == -delta:
                                prev_direction = Template.Direction.DOWN
                            else:
                                raise ValueError(
                                    "The last two vertices of the path are not adjacent."
                                )
                    if path is not None:
                        if i == 0:
                            piece_to_path[piece].append(path[0])
                        assert len(path) >= 2, "path needs to have at least 2 vertices"
                        piece_to_path[piece].extend(path[1:])
                        seam.points = path

        # fill the faces
        for face in template.faces:
            for piece, path in piece_to_path.items():
                if Vertex2D.encloses(face.get_center(), path):
                    face.inside = 1
                    piece_to_faces[piece].append(face)
                    face_to_pieces[face].append(piece)
                    # break

        return piece_to_faces, face_to_pieces

    @staticmethod
    def embed(
        template: Template, pieces: List[Piece]
    ) -> Tuple[Dict[Piece, List[Face2D]], Dict[Face2D, List[Piece]]]:
        face_to_pieces: Dict[Face2D, List[Piece]] = defaultdict(list)
        piece_to_faces: Dict[Piece, List[Face2D]] = defaultdict(list)

        for face in template.faces:
            container: Optional[Piece] = None
            face.inside = 0
            for piece in pieces:
                # Checks if the center of the face is within the bounds of the piece, taking into account whether the piece is reversed.
                if piece.template_piece.encloses(face.get_center(), piece.reversed):
                    container = piece
                    face.inside = 1
                    piece_to_faces[container].append(face)
                    face_to_pieces[face].append(container)

        return piece_to_faces, face_to_pieces

    @staticmethod
    def assign_faces_to_pieces(
        template: Template, pieces: List[Piece]
    ) -> Dict[Piece, List[Face2D]]:
        """
        Find the faces that each piece contains

        Parameters
        ----------
        template : Template
            The template which the faces belong to
        pieces : List[Piece]
            The pieces to assign faces to

        Returns
        -------
        Dict[Piece, List[Face2D]]
            A dictionary mapping each piece to a list of faces that it contains
        """
        face_to_piece: Dict[Face2D, Piece] = {}
        piece_to_faces: Dict[Piece, List[Face2D]] = {}

        for piece in pieces:
            piece_to_faces[piece] = []

        for face in template.faces:
            container: Optional[Piece] = None
            for piece in pieces:
                # Checks if the center of the face is within the bounds of the piece, taking into account whether the piece is reversed.
                if piece.encloses(face.get_center()):
                    container = piece
                    break
            if container is not None:
                face.inside = 1
                piece_to_faces[container].append(face)
            else:
                face.inside = 0
            face_to_piece[face] = container
        return piece_to_faces
