from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from garmentimage.utils.template import Template
from garmentimage.utils.vertex2d import Vector2, Vertex2D

if TYPE_CHECKING:
    from garmentimage.utils.edge2d import Edge2D
    from garmentimage.utils.face import Face2D
    from garmentimage.utils.mesh import Mesh2D
    from garmentimage.utils.piece import Piece
    from garmentimage.utils.seam import Seam


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
    def embed_seam_type_into_template_v2(piece: Piece, template: Template):
        seams: List[Seam] = piece.seams if piece.seams else []
        for seam in seams:
            assert seam.start.corner is not None and seam.end.corner is not None, (
                "The start and end of the seam are not set. Please set them before embedding."
            )
            print(f"seam.points: {seam.points}")
            for i in range(len(seam.points) - 1):
                v0: Vertex2D = seam.points[i]
                v1: Vertex2D = seam.points[i + 1]

                xy0: List[int] = Template.get_grid_xy(v0)
                xy1: List[int] = Template.get_grid_xy(v1)
                print(f"xy0: {xy0}, xy1: {xy1}")

                # left and bottom
                x: int = min(xy0[0], xy1[0])
                y: int = min(xy0[1], xy1[1])

                if xy0[1] == xy1[1]:
                    template.grid_vertices[x][y].X_EDGE_TYPE = seam.type
                else:
                    template.grid_vertices[x][y].Y_EDGE_TYPE = seam.type

        for x in range(Template.N):  # Assuming Template.N defines the grid size
            for y in range(Template.N):
                template.h_edges[x][y].seam_type = template.grid_vertices[x][
                    y
                ].X_EDGE_TYPE
                template.v_edges[x][y].seam_type = template.grid_vertices[x][
                    y
                ].Y_EDGE_TYPE

    @staticmethod
    def embed_seam_type_into_template(
        piece: Piece, template: Template, faces: Optional[List[Face2D]] = None
    ):
        seams: List[Seam] = piece.get_all_seams()
        tmp_mesh: Optional[Mesh2D] = (
            Mesh2D(faces, integrate_adjascent_face_edges=True)
            if faces is not None
            else None
        )
        for seam in seams:
            if faces is not None:
                v0: Vertex2D = seam.start.corner
                v1: Vertex2D = seam.end.corner
                seam_points: List[Vertex2D] = Template.get_path(
                    tmp_mesh,
                    v0,
                    v1,
                    boundary_only=False,
                    is_reversed=False,
                    seam_type=None,
                )
            else:
                seam_points: List[Vertex2D] = piece.template_piece.seam_to_points[seam]
            for j in range(len(seam_points) - 1):
                v0: Vertex2D = seam_points[j]
                v1: Vertex2D = seam_points[j + 1]

                xy0: List[int] = Template.get_grid_xy(v0)
                xy1: List[int] = Template.get_grid_xy(v1)

                # top and right
                x: int = min(xy0[0], xy1[0])
                y: int = min(xy0[1], xy1[1])

                if xy0[1] == xy1[1]:
                    template.grid_vertices[x][y].X_EDGE_TYPE = seam.type
                else:
                    template.grid_vertices[x][y].Y_EDGE_TYPE = seam.type

        for x in range(Template.N):  # Assuming Template.N defines the grid size
            for y in range(Template.N):
                template.h_edges[x][y].seam_type = template.grid_vertices[x][
                    y
                ].X_EDGE_TYPE
                template.v_edges[x][y].seam_type = template.grid_vertices[x][
                    y
                ].Y_EDGE_TYPE

    @staticmethod
    def modify_seam_points(piece: Piece, template: Template, faces: List[Face2D]):
        seams = piece.get_all_seams()
        tmp_mesh: Optional[Mesh2D] = (
            Mesh2D(faces, integrate_adjascent_face_edges=True)
            if faces is not None
            else None
        )
        for seam in seams:
            if seam.type != Seam.BOUNDARY:
                seam_points = piece.template_piece.seam_to_points[seam]
                new_seam_points: List[Vertex2D] = [point for point in seam_points]
                updated_flag = False
                for j in range(len(seam_points) - 1):
                    template_v0: Vertex2D = seam_points[j]
                    template_v1: Vertex2D = seam_points[j + 1]
                    v0: Vertex2D = tmp_mesh.find_nearest_vertex(template_v0)
                    v1: Vertex2D = tmp_mesh.find_nearest_vertex(template_v1)
                    edge: Edge2D = v0.get_common_edge(v1)
                    if (edge.left_face is not None and edge.left_face.inside == 1) and (
                        edge.right_face is not None and edge.right_face.inside == 1
                    ):
                        if j == 1:
                            x0, y0 = Template.get_grid_xy(template_v0)
                            x1, y1 = Template.get_grid_xy(template_v1)
                            template_new_v0: Vertex2D = template.grid_vertices[x0][
                                y0 - 1
                            ]
                            new_v0: Vertex2D = tmp_mesh.find_nearest_vertex(
                                template_new_v0
                            )
                            new_seam_points[0] = new_v0
                            if x0 < x1:
                                template.grid_vertices[x0][y0].X_EDGE_TYPE = Seam.NONE
                                template.grid_vertices[x0][
                                    y0 - 1
                                ].X_EDGE_TYPE = seam.type
                                template.grid_vertices[x1][
                                    y0 - 1
                                ].Y_EDGE_TYPE = seam.type
                            else:
                                template.grid_vertices[x1][y0].X_EDGE_TYPE = Seam.NONE
                                template.grid_vertices[x1][
                                    y0 - 1
                                ].X_EDGE_TYPE = seam.type
                                template.grid_vertices[x1][
                                    y0 - 1
                                ].Y_EDGE_TYPE = seam.type

                        elif j == len(seam_points) - 3:
                            x0, y0 = Template.get_grid_xy(template_v0)
                            x1, y1 = Template.get_grid_xy(template_v1)
                            template_new_v1: Vertex2D = template.grid_vertices[x0][
                                y0 - 1
                            ]
                            new_v1: Vertex2D = tmp_mesh.find_nearest_vertex(
                                template_new_v1
                            )
                            new_seam_points[j + 1] = new_v1
                            if x0 < x1:
                                template.grid_vertices[x0][y0].X_EDGE_TYPE = Seam.NONE
                                template.grid_vertices[x0][
                                    y0 - 1
                                ].X_EDGE_TYPE = seam.type
                                template.grid_vertices[x0][
                                    y0 - 1
                                ].Y_EDGE_TYPE = seam.type
                            else:
                                template.grid_vertices[x1][y1].X_EDGE_TYPE = Seam.NONE
                                template.grid_vertices[x1][
                                    y1 - 1
                                ].X_EDGE_TYPE = seam.type
                                template.grid_vertices[x0][
                                    y1 - 1
                                ].Y_EDGE_TYPE = seam.type

                        updated_flag = True
                        break
                if updated_flag:
                    piece.template_piece.seam_to_points[seam] = new_seam_points

    @staticmethod
    def embed_transformation_into_template(piece_mesh: Mesh2D, template: Template):
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
        Embed pieces into the template based on edges of pieces, not depending on the center of the face.
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

                    print(f"start: {start}, end: {end}")
                    seam_type = seam.type
                    if start is not None and end is not None:
                        print(f"prev_seam_type: {prev_seam_type}")
                        print("prev_direction: ", prev_direction)
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
                        print(f"path: {path}")
                        seam.points = path

        # fill the faces
        for face in template.faces:
            for piece, path in piece_to_path.items():
                if Vertex2D.encloses(face.get_center(), path):
                    face.inside = 1
                    piece_to_faces[piece].append(face)
                    face_to_pieces[face].append(piece)
                    break

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
