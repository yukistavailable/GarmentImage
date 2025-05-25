from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from garmentimage.utils.draw_panel import DrawPanel
    from garmentimage.utils.edge2d import Edge2D
    from garmentimage.utils.embedding import BoundaryEmbedding
    from garmentimage.utils.encoder import Encoder
    from garmentimage.utils.face import Face2D
    from garmentimage.utils.mesh import Mesh2D
    from garmentimage.utils.piece import Piece
    from garmentimage.utils.seam import Seam
    from garmentimage.utils.template import Template
    from garmentimage.utils.template_panel import TemplatePanel
    from garmentimage.utils.vertex2d import Vertex2D


class Decoder:
    @staticmethod
    def decode_embedded_template_to_piece_edge(
        target_template: Template,
        new_template: Template,
        use_vertex_constraints: bool = True,
        store_original_meshes: bool = False,
        address_inside_seam: bool = False,
    ) -> None:
        pieces = [
            template_piece.piece for template_piece in target_template.template_pieces
        ]
        piece_to_faces: Dict[Piece, List[Face2D]] = (
            BoundaryEmbedding.assign_faces_to_pieces(target_template, pieces)
        )
        new_template.clear_meshes()
        meshes: List[Mesh2D] = new_template.meshes
        original_meshes: Optional[List[Mesh2D]] = None
        if store_original_meshes:
            original_meshes = new_template.original_meshes
            new_template.original_meshes.clear()
        for piece_index, piece in enumerate(pieces):
            faces: List[Face2D] = piece_to_faces[piece]
            # construct a mesh from the boundary of the piece and the target faces
            piece_mesh: Mesh2D = Encoder.get_mesh2D(
                piece,
                faces,
                decode_mode=True,
                address_inside_seam=address_inside_seam,
            )
            if store_original_meshes:
                original_meshes.append(piece_mesh.duplicate())
            if len(piece_mesh.vertices) == 0:
                print(f"Piece {piece} has no vertices")
                continue
            Decoder.deform_piece_edge(
                faces,
                target_template,
                piece_mesh,
                use_vertex_constraints=use_vertex_constraints,
            )
            # TODO: set seam type
            meshes.append(piece_mesh)
        new_template.edge_global_index_to_piece_index_and_local_index = (
            target_template.edge_global_index_to_piece_index_and_local_index
        )
        new_template.stitched_edges_pair = target_template.stitched_edges_pair

    @staticmethod
    def deform_piece_edge(
        faces: List[Face2D],
        template: Template,
        piece_mesh: Mesh2D,
        use_vertex_constraints: bool = True,
    ) -> None:
        piece_mesh.set_indices()
        num_edges = len(faces) * 4
        num_vertices = len(piece_mesh.vertices)
        if use_vertex_constraints:
            # the following line cau cause division by zero error
            weight = 1 / num_vertices
            L = lil_matrix((num_edges + num_vertices, num_vertices))
            Bx = np.zeros(num_edges + num_vertices)
            By = np.zeros(num_edges + num_vertices)
            # L = lil_matrix((num_edges + 1, num_vertices))
            # Bx = np.zeros(num_edges + 1)
            # By = np.zeros(num_edges + 1)
        else:
            L = lil_matrix((num_edges, num_vertices))
            Bx = np.zeros(num_edges)
            By = np.zeros(num_edges)

        for i, face in enumerate(faces):
            x = face.grid_x
            y = face.grid_y
            target_face = template.grid_faces[x][y]
            target_h_edge0 = target_face.H_EDGE0
            target_h_edge1 = target_face.H_EDGE1
            target_v_edge0 = target_face.V_EDGE0
            target_v_edge1 = target_face.V_EDGE1
            face_edges: List[Edge2D] = target_face.edges
            h_edge0, h_edge1, v_edge0, v_edge1 = Decoder.sort_face_edges(face_edges)
            piece_h_edge0 = Decoder.extract_edge_from_mesh(
                piece_mesh, h_edge0, edge_idx=0
            )
            piece_h_edge1 = Decoder.extract_edge_from_mesh(
                piece_mesh, h_edge1, edge_idx=1
            )
            piece_v_edge0 = Decoder.extract_edge_from_mesh(
                piece_mesh, v_edge0, edge_idx=2
            )
            piece_v_edge1 = Decoder.extract_edge_from_mesh(
                piece_mesh, v_edge1, edge_idx=3
            )
            edge_index = i * 4
            L[edge_index, piece_h_edge0.start.index] = -1
            L[edge_index, piece_h_edge0.end.index] = 1
            Bx[edge_index] = target_h_edge0.x
            By[edge_index] = target_h_edge0.y

            L[edge_index + 1, piece_h_edge1.start.index] = -1
            L[edge_index + 1, piece_h_edge1.end.index] = 1
            Bx[edge_index + 1] = target_h_edge1.x
            By[edge_index + 1] = target_h_edge1.y

            L[edge_index + 2, piece_v_edge0.start.index] = -1
            L[edge_index + 2, piece_v_edge0.end.index] = 1
            Bx[edge_index + 2] = target_v_edge0.x
            By[edge_index + 2] = target_v_edge0.y

            L[edge_index + 3, piece_v_edge1.start.index] = -1
            L[edge_index + 3, piece_v_edge1.end.index] = 1
            Bx[edge_index + 3] = target_v_edge1.x
            By[edge_index + 3] = target_v_edge1.y

            # set seam type
            piece_h_edge0.seam_type = h_edge0.seam_type
            piece_h_edge1.seam_type = h_edge1.seam_type
            piece_v_edge0.seam_type = v_edge0.seam_type
            piece_v_edge1.seam_type = v_edge1.seam_type

        if use_vertex_constraints:
            for i, v in enumerate(piece_mesh.vertices):
                index = num_edges + i
                L[index, v.index] = 1 * weight
                Bx[index] = v.x * weight
                By[index] = v.y * weight
            # v = piece_mesh.vertices[0]
            # index = num_edges
            # L[index, v.index] = 1
            # Bx[index] = v.x
            # By[index] = v.y

        L = L.tocsr()
        xs = spsolve(L.T @ L, L.T @ Bx)
        ys = spsolve(L.T @ L, L.T @ By)

        for i, v in enumerate(piece_mesh.vertices):
            v.x = xs[i]
            v.y = ys[i]

    @staticmethod
    def extract_edge_from_mesh(
        mesh: Mesh2D, target_edge: Edge2D, edge_idx: Optional[int] = None
    ) -> Optional[Edge2D]:
        """
        Return the edge in a mesh that has the same start and end vertices as the target edge
        Parameters
        ----------
        mesh : Mesh2D
            A 2D mesh
        target_edge : Edge2D
            A target edge
        edge_idx:
            0: h_edge0
            1: h_edge1
            2: v_edge0
            3: v_edge1
        Returns
        -------
        Edge2D
            The edge in the mesh that has the same start and end vertices as the target edge
        """
        edges: List[Edge2D] = []
        for edge in mesh.edges:
            if Vertex2D.same_position(
                edge.start, target_edge.start
            ) and Vertex2D.same_position(edge.end, target_edge.end):
                if edge_idx is None:
                    return edge
                else:
                    edges.append(edge)
        if edge_idx is not None and len(edges) > 1:
            # this is kireme: https://github.com/yukistavailable/Dresscode/issues/171
            for edge in edges:
                face = edge.left_face if edge.left_face is not None else edge.right_face
                assert face is not None
                face_edges = Decoder.sort_face_edges(face.edges)
                for i, face_edge in enumerate(face_edges):
                    if Vertex2D.same_position(
                        edge.start, face_edge.start
                    ) and Vertex2D.same_position(edge.end, face_edge.end):
                        actual_edge_idx = i
                if actual_edge_idx == edge_idx:
                    return edge
        elif len(edges) == 1:
            return edges[0]

        return None

    @staticmethod
    def sort_face_edges(edges: List[Edge2D]) -> Tuple[Edge2D]:
        """
        Sort a list of edges in a face in a specific order
        """
        assert len(edges) == 4
        edges.sort(
            key=lambda edge: edge.start.x + edge.start.y + edge.end.x + edge.end.y
        )
        if edges[0].end.y > edges[1].end.y:
            edges[0], edges[1] = edges[1], edges[0]
        if edges[2].start.x > edges[3].start.x:
            edges[2], edges[3] = edges[3], edges[2]
        h_edge0 = edges[0]
        h_edge1 = edges[2]
        v_edge0 = edges[1]
        v_edge1 = edges[3]
        return (h_edge0, h_edge1, v_edge0, v_edge1)

    @staticmethod
    def reconstruct_pieces(
        template: Template,
        is_reversed: bool,
        draw_panel: Optional[DrawPanel] = None,
        template_panel: Optional[TemplatePanel] = None,
    ) -> None:
        meshes: List[Mesh2D] = template.meshes
        for mesh in meshes:
            boundary: List[Vertex2D] = Decoder.trace_boundary(mesh)
            piece: Piece = Decoder.generate_piece(boundary)
            piece.reversed = is_reversed
            if draw_panel is not None:
                draw_panel.pieces.append(piece)
            template_panel.add_piece(piece)

    @staticmethod
    def generate_piece(boundary: List[Vertex2D]) -> Piece:
        seams: List[Seam] = []
        n: int = len(boundary)
        for i in range(n):
            v0: Vertex2D = boundary[i]
            v1: Vertex2D = boundary[(i + 1) % n]
            stroke: List[Vertex2D] = []
            stroke.append(v0)
            stroke.append(Vertex2D.mid_point(v0, v1))
            stroke.append(v1)
            seam: Seam = Seam(stroke)
            seams.append(seam)
        piece: Piece = Piece(seams)
        return piece

    @staticmethod
    def trace_boundary(mesh: Mesh2D) -> List[Vertex2D]:
        start_edge: Optional[Edge2D] = None
        for edge in mesh.edges:
            if edge.left_face is None or edge.right_face is None:
                start_edge = edge
                break
        loop: List[Vertex2D] = []
        prev_v: Vertex2D = (
            start_edge.start if start_edge.left_face is None else start_edge.end
        )
        start_v: Vertex2D = prev_v
        edge: Edge2D = start_edge
        while True:
            loop.append(Vertex2D(prev_v))
            next_v: Vertex2D = edge.get_the_other_vertex(prev_v)
            if next_v == start_v:
                return loop
            edge = Decoder.find_next_boundary_edge(next_v, edge)
            assert edge is not None, (
                "trace_boundary() failed to find next boundary edge"
            )
            prev_v = next_v

    @staticmethod
    def find_next_boundary_edge(v: Vertex2D, prev_edge: Edge2D) -> Optional[Edge2D]:
        for edge in v.edges:
            if edge == prev_edge:
                continue
            if edge.left_face is None or edge.right_face is None:
                return edge
        return None
