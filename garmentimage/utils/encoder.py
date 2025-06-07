from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from garmentimage.utils.edge2d import Edge2D
from garmentimage.utils.embedding import BoundaryEmbedding, Embedding
from garmentimage.utils.face import Face2D
from garmentimage.utils.mesh import Mesh2D
from garmentimage.utils.piece import Piece
from garmentimage.utils.seam import Seam
from garmentimage.utils.template import Template
from garmentimage.utils.vertex2d import Vector2, Vertex2D


class Encoder:
    def __init__(self):
        self.mesh2D_to_piece: Dict[Mesh2D, Piece] = {}

    def encode_pieces_to_template_v2(
        self,
        pieces: List[Piece],
        template: Template,
        is_reversed: bool,
        visualize: bool = False,
        symmetrize: bool = False,  # deprecated
        pieces_original_shape: Optional[List[Piece]] = None,
        symmetrize_is_left_base: bool = True,
        is_seam_reversed: bool = False,
    ):
        template.clear()
        # faces that each piece contains
        piece_to_faces, face_to_pieces = BoundaryEmbedding.embed_v2(
            template, pieces, is_reversed=is_seam_reversed
        )

        # adjust overlapping faces
        # piece_to_faces, face_to_piece = Encoder.adjust_overlapping_faces(
        #     piece_to_faces, face_to_pieces
        # )

        meshes: List[Mesh2D] = template.meshes
        original_meshes: List[Mesh2D] = template.original_meshes
        template.clear_meshes()
        template.original_meshes.clear()
        for piece in pieces:
            # boundary_only = False becuase faces have no information about seam_type, which means it is impossible to detect kireme
            # piece.template_piece.update_points(
            #     faces=piece_to_faces[piece],
            #     boundary_only=False,
            #     consider_seam_type=True,
            # )
            Embedding.embed_seam_type_into_template_v2(piece, template)
            # Embedding.modify_seam_points(piece, template, faces=piece_to_faces[piece])

        # template.visualize_faces("output_debug/faces.png")
        # exit()

        for i, piece in enumerate(pieces):
            # print(f"Encoding piece {piece}")
            piece_mesh: Mesh2D = Encoder.get_mesh2D(
                piece,
                piece_to_faces[piece],
                template=template,
            )
            original_meshes.append(piece_mesh.duplicate())
            # print("Constraints")
            constraints: Dict[Vertex2D, Vertex2D] = (
                Encoder.find_transfer_constraints_v2(
                    piece_mesh,
                    piece,
                    is_reversed,
                    piece_original_shape=None
                    if pieces_original_shape is None
                    else pieces_original_shape[i],
                )
            )
            assert piece.template_piece is not None, "piece.template_piece is None"
            piece.template_piece.add_constraints(constraints)
            # piece.template_piece.add_original_constraints(deepcopy(constraints))
            original_constraints: Dict[Vertex2D, Vertex2D] = {
                k.duplicate(): v.duplicate() for k, v in constraints.items()
            }
            piece.template_piece.add_original_constraints(original_constraints)
            Encoder.deform_template_to_piece_edge(piece_mesh, constraints)
            Embedding.embed_transformation_into_template(piece_mesh, template)
            self.mesh2D_to_piece[piece_mesh] = piece
            meshes.append(piece_mesh)

    @staticmethod
    def adjust_overlapping_faces(
        piece_to_faces: Dict[Piece, List[Face2D]],
        face_to_pieces: Dict[Face2D, List[Piece]],
    ) -> Tuple[Dict[Piece, List[Face2D]], Dict[Face2D, Piece]]:
        """
        Adjusts overlapping faces by assigning each face to the nearest piece.
        """
        new_piece_to_faces: Dict[Piece, List[Face2D]] = defaultdict(list)
        piece_to_centers: Dict[Piece, Vertex2D] = {}
        for piece, faces in piece_to_faces.items():
            if len(faces) == 0:
                print("Warning: piece with no faces")
                continue
            sum_x, sum_y = 0, 0
            for face in faces:
                sum_x += face.get_center().x
                sum_y += face.get_center().y
            piece_to_centers[piece] = Vertex2D(sum_x / len(faces), sum_y / len(faces))

        face_to_piece: Dict[Face2D, Piece] = {}
        for face, pieces in face_to_pieces.items():
            if len(pieces) == 0:
                print("Warning: face with no pieces")
                continue
            nearest_piece: Optional[Piece] = None
            min_distance = math.inf
            for piece in pieces:
                distance = piece_to_centers[piece].distance(face.get_center())
                if distance < min_distance:
                    min_distance = distance
                    nearest_piece = piece
            face_to_piece[face] = nearest_piece
            new_piece_to_faces[nearest_piece].append(face)
        return new_piece_to_faces, face_to_piece

    @staticmethod
    def deform_template_to_piece_edge(
        piece_mesh: Mesh2D, constraints: Dict[Vertex2D, Vertex2D]
    ):
        piece_mesh.set_indices()

        num_constraints = len(constraints)
        num_edges = len(piece_mesh.edges)
        num_vertices = len(piece_mesh.vertices)

        # solve linear constrained least squares problem
        # A: edge adjacency matrix
        # L: A.T @ A
        # bx, by: target edge vector
        # C: constraint matrix
        # cx, cy: target constraint vector
        #
        # min || A x - b || s.t. C x = c
        #

        # Build A and bx, by
        bx = np.zeros(num_edges)
        by = np.zeros(num_edges)
        A_elem, A_ind_i, A_ind_j = [], [], []

        for i, edge in enumerate(piece_mesh.edges):
            v0: Vertex2D = edge.start
            v1: Vertex2D = edge.end

            vec0_1 = Vector2(v0, v1)
            A_elem.append(1)
            A_ind_i.append(i)
            A_ind_j.append(v0.index)

            A_elem.append(-1)
            A_ind_i.append(i)
            A_ind_j.append(v1.index)

            bx[i] = vec0_1.x
            by[i] = vec0_1.y

        # Build C and c
        cx = np.zeros(num_constraints)
        cy = np.zeros(num_constraints)
        C_elem, C_ind_i, C_ind_j = [], [], []
        for i, (key, target) in enumerate(constraints.items()):
            C_elem.append(1)
            C_ind_i.append(i)
            C_ind_j.append(key.index)
            cx[i] = target.x
            cy[i] = target.y

        A = csr_matrix((A_elem, (A_ind_i, A_ind_j)), shape=(num_edges, num_vertices))
        C = csr_matrix(
            (C_elem, (C_ind_i, C_ind_j)), shape=(num_constraints, num_vertices)
        )

        # Build as block matrix
        # [ [A, -C]
        #   [C,  0] ]

        L = scipy.sparse.bmat([[A.T @ A, -C.T], [C, None]])
        Bx = np.concatenate([A.T @ bx, cx])
        By = np.concatenate([A.T @ by, cy])

        # Solve the linear least squares problem
        xs = spsolve(L, Bx)
        ys = spsolve(L, By)

        # Store results
        for i, v in enumerate(piece_mesh.vertices):
            v.x = xs[i]
            v.y = ys[i]

    @staticmethod
    def get_mesh2D(
        piece: Piece,
        inside_faces: List[Face2D],
        decode_mode: bool = False,
        template: Optional[Template] = None,
        address_inside_seam: bool = False,
    ) -> Mesh2D:
        if decode_mode:
            boundary_edges: Set[Edge2D] = Encoder.get_boundary_edges(piece)
        else:
            boundary_edges: Set[Edge2D] = Encoder.get_boundary_edges_from_faces(
                piece, inside_faces, template=template
            )
        seam_edges: Set[Edge2D] = set()
        seam_edges_id: Set[Tuple[int, int, int, int]] = set()
        # add boundary_edges to seam_edges
        for edge in boundary_edges:
            seam_edges.add(edge)
            edge_id = (
                edge.start.grid_xy[0],
                edge.start.grid_xy[1],
                edge.end.grid_xy[0],
                edge.end.grid_xy[1],
            )
            seam_edges_id.add(
                edge_id,
            )

        if address_inside_seam:
            inside_seam_edges = (
                Face2D.extract_inside_edges_non_none_seam_type_from_faces(inside_faces)
            )
            for edge in inside_seam_edges:
                edge_id = (
                    edge.start.grid_xy[0],
                    edge.start.grid_xy[1],
                    edge.end.grid_xy[0],
                    edge.end.grid_xy[1],
                )
                if edge_id not in seam_edges_id:
                    seam_edges.add(edge)
                    seam_edges_id.add(edge_id)

                reversed_edge = edge.shallow_duplicate_reverse()
                reversed_edge_id = (
                    reversed_edge.start.grid_xy[0],
                    reversed_edge.start.grid_xy[1],
                    reversed_edge.end.grid_xy[0],
                    reversed_edge.end.grid_xy[1],
                )
                if reversed_edge_id not in seam_edges_id:
                    seam_edges.add(reversed_edge)
                    seam_edges_id.add(reversed_edge_id)

        mesh2D: Mesh2D = Mesh2D(inside_faces, seam_edges)
        return mesh2D

    @staticmethod
    def get_boundary_edges(piece: Piece) -> Set[Edge2D]:
        """
        Retrieve the set of boundary edges for a given piece.

        Parameters
        ----------
        piece: Piece
            The piece object for which to find the boundary edges.

        Returns
        -------
        Set[Edge2D]
            A set of boundary edges, each represented as an Edge2D object.
        """
        boundary_edges: Set[Edge2D] = set()
        seams: List[Seam] = piece.get_all_seams()
        for seam in seams:
            seam_points: List[Vertex2D] = piece.template_piece.seam_to_points[seam]
            for j in range(len(seam_points) - 1):
                v0: Vertex2D = seam_points[j]
                v1: Vertex2D = seam_points[j + 1]
                # xy0: List[int] = Template.get_grid_xy(v0)
                # xy1: List[int] = Template.get_grid_xy(v1)
                # t0: List[Vertex2D] = template.grid_vertices[xy0[0]][xy0[1]]
                # t1: List[Vertex2D] = template.grid_vertices[xy1[0]][xy1[1]]
                edge: Edge2D = v0.get_common_edge(v1)
                boundary_edges.add(edge)
        return boundary_edges

    @staticmethod
    def get_boundary_edges_from_faces(
        piece: Piece, faces: List[Face2D], template: Optional[Template] = None
    ) -> Set[Edge2D]:
        """
        Retrieve the set of boundary edges for a given piece.

        Parameters
        ----------
        piece: Piece
            The piece object for which to find the boundary edges.

        faces: List[Face2D]
            The list of faces to find the boundary edges for.

        template: Template
            The template object to find the seam type for each edge.

        Returns
        -------
        Set[Edge2D]
            A set of boundary edges, each represented as an Edge2D object.
        """
        boundary_edges: Set[Edge2D] = set()
        seams: List[Seam] = piece.get_all_seams()
        for seam in seams:
            v0: Vertex2D = seam.start.corner
            v1: Vertex2D = seam.end.corner
            tmp_mesh: Mesh2D = Mesh2D(faces, integrate_adjacent_face_edges=True)
            if template is not None:
                seam_points: List[Vertex2D] = Template.get_path(
                    tmp_mesh,
                    v0,
                    v1,
                    boundary_only=True,
                    is_reversed=False,
                    seam_type=seam.type,
                    template=template,
                )
            else:
                seam_points: List[Vertex2D] = Template.get_path(
                    tmp_mesh, v0, v1, boundary_only=True, is_reversed=False
                )
            # seam_points: List[Vertex2D] = piece.template_piece.seam_to_points[seam]
            for j in range(len(seam_points) - 1):
                v0: Vertex2D = seam_points[j]
                v1: Vertex2D = seam_points[j + 1]
                # xy0: List[int] = Template.get_grid_xy(v0)
                # xy1: List[int] = Template.get_grid_xy(v1)
                # t0: List[Vertex2D] = template.grid_vertices[xy0[0]][xy0[1]]
                # t1: List[Vertex2D] = template.grid_vertices[xy1[0]][xy1[1]]
                edge: Edge2D = v0.get_common_edge(v1)
                boundary_edges.add(edge)
        return boundary_edges

    @staticmethod
    def find_transfer_constraints_v2(
        piece_mesh: Mesh2D,
        piece: Piece,
        is_reversed: bool,
        piece_original_shape: Optional[Piece] = None,
    ) -> Dict[Vertex2D, Vertex2D]:
        constraints: Dict[Vertex2D, Vertex2D] = {}
        if piece_original_shape is not None:
            seams = Piece.sorted_seams(piece.seams if piece.seams else [])
            target_seams = Piece.sorted_seams(
                piece_original_shape.seams if piece_original_shape.seams else []
            )
            if is_reversed:
                seams = list(reversed(seams))
                target_seams = list(reversed(target_seams))
            iter_seams = zip(
                seams,
                target_seams,
            )
        else:
            seams = Piece.sorted_seams(piece.get_all_seams())
            iter_seams = zip(seams, seams)
        for seam, target_seam in iter_seams:
            points: List[Vertex2D] = seam.points
            if is_reversed:
                points = list(reversed(points))
                print("is_reversed")
            path: List[Vertex2D] = []
            print(f"points: {points}")
            print(f"seam: {seam}")
            print(f"target_seam: {target_seam}")
            for i in range(len(points) - 1):
                v0: Vertex2D = points[i]
                v1: Vertex2D = points[i + 1]
                # find an edge in a mesh that corresponds to two given vertices v0 and v1
                mesh_edge: Optional[List[Vertex2D]] = (
                    Encoder.find_corresponding_mesh_edge(
                        piece_mesh, v0, v1, is_reversed=is_reversed
                    )
                )
                if mesh_edge is not None:
                    if len(path) == 0:
                        path.append(mesh_edge[0])
                    path.append(mesh_edge[1])
            n: int = len(path)
            for i in range(n):
                piece_mesh_v: Vertex2D = path[i]
                target_position: Vertex2D = Encoder.sample_point_on_curve(
                    target_seam, i / (n - 1)
                )
                constraints[piece_mesh_v] = target_position
        return constraints

    @staticmethod
    def find_transfer_constraints(
        piece_mesh: Mesh2D,
        piece: Piece,
        is_reversed: bool,
        piece_original_shape: Optional[Piece] = None,
    ) -> Dict[Vertex2D, Vertex2D]:
        constraints: Dict[Vertex2D, Vertex2D] = {}
        if piece_original_shape is not None:
            seams = Piece.sorted_seams(piece.get_all_seams())
            target_seams = Piece.sorted_seams(piece_original_shape.get_all_seams())
            iter_seams = zip(
                seams,
                target_seams,
            )
        else:
            seams = Piece.sorted_seams(piece.get_all_seams())
            iter_seams = zip(seams, seams)
        for seam, target_seam in iter_seams:
            points: List[Vertex2D] = piece.template_piece.seam_to_points[seam]
            path: List[Vertex2D] = []
            for i in range(len(points) - 1):
                v0: Vertex2D = points[i]
                v1: Vertex2D = points[i + 1]
                # find an edge in a mesh that corresponds to two given vertices v0 and v1
                mesh_edge: Optional[List[Vertex2D]] = (
                    Encoder.find_corresponding_mesh_edge(
                        piece_mesh, v0, v1, is_reversed
                    )
                )
                if mesh_edge is not None:
                    if len(path) == 0:
                        path.append(mesh_edge[0])
                    path.append(mesh_edge[1])
            n: int = len(path)
            for i in range(n):
                piece_mesh_v: Vertex2D = path[i]
                target_position: Vertex2D = Encoder.sample_point_on_curve(
                    target_seam, i / (n - 1)
                )
                constraints[piece_mesh_v] = target_position
        return constraints

    @staticmethod
    def find_corresponding_mesh_edge(
        piece_mesh: Mesh2D, v0: Vertex2D, v1: Vertex2D, is_reversed: bool
    ) -> Optional[List[Vertex2D]]:
        mesh_edge: Optional[List[Vertex2D]] = None
        for edge in piece_mesh.edges:
            if edge.left_face is not None and edge.right_face is not None:
                continue
            flag: bool = (edge.right_face is None) != is_reversed
            start: Vertex2D = edge.start if flag else edge.end
            end: Vertex2D = edge.end if flag else edge.start
            if Template.same_grid_position(v0, start) and Template.same_grid_position(
                v1, end
            ):
                mesh_edge = [start, end]
        return mesh_edge

    @staticmethod
    def sample_point_on_curve(seam: Seam, ratio: float) -> Vertex2D:
        total: float = 0
        for i in range(len(seam.stroke) - 1):
            v0: Vertex2D = seam.stroke[i]
            v1: Vertex2D = seam.stroke[i + 1]
            total += Vertex2D.distance_static(v0, v1)
        remaining: float = total * ratio
        for i in range(len(seam.stroke) - 1):
            v0: Vertex2D = seam.stroke[i]
            v1: Vertex2D = seam.stroke[i + 1]
            d: float = Vertex2D.distance_static(v0, v1)
            if d > remaining:
                return Vertex2D.interpolate(v0, v1, remaining / d)
            remaining -= d
        return seam.stroke[-1]

    @staticmethod
    def update_piece_to_faces_aligned_with_template(
        template: Template,
        closed_paths: List[List[Edge2D]],
        piece_to_faces: Dict[Piece, List[Face2D]],
        face_to_piece: Dict[Face2D, Piece],
    ) -> None:
        """
        Updates the `piece_to_faces` dictionary and the `face_to_piece` dictionary based on the given `closed_paths` and `template`.
        Applies the updates to the template as well.
        """

        checked_faces: List[Face2D] = []
        checked_pieces: List[Piece] = []
        for closed_path in closed_paths:
            inner_faces: List[Face2D] = []
            piece_to_count: Dict[Piece, int] = defaultdict(int)
            for face in template.faces:
                if face in checked_faces:
                    continue
                if Edge2D.encloses(face.get_center(), closed_path):
                    # if the face is inside the closed path, then it is an inner face
                    inner_faces.append(face)
                    checked_faces.append(face)
                    if face_to_piece.get(face) is not None:
                        piece_to_count[face_to_piece[face]] += 1
            if piece_to_count == {}:
                continue
            most_common_piece = max(piece_to_count, key=piece_to_count.get)
            assert most_common_piece not in checked_pieces, (
                "This piece is already checked. Different closed paths should not share the same piece."
            )
            checked_pieces.append(most_common_piece)
            for face in inner_faces:
                face.inside = 1
                prev_piece = face_to_piece.get(face)
                face_to_piece[face] = most_common_piece
                if prev_piece is None or prev_piece != most_common_piece:
                    if prev_piece is not None:
                        piece_to_faces[prev_piece].remove(face)
                    if face not in piece_to_faces[most_common_piece]:
                        piece_to_faces[most_common_piece].append(face)

        for face in template.faces:
            if face not in checked_faces:
                face.inside = 0
                piece = face_to_piece.get(face)
                face_to_piece[face] = None
                if piece is not None:
                    piece_to_faces[piece].remove(face)
