from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from garmentimage.utils.draw_panel import DrawPanel
from garmentimage.utils.edge2d import Edge2D
from garmentimage.utils.embedding import BoundaryEmbedding
from garmentimage.utils.encoder import Encoder
from garmentimage.utils.face import Face2D
from garmentimage.utils.mesh import Mesh2D
from garmentimage.utils.neural_tailor_converter import NeuralTailorConverter
from garmentimage.utils.piece import Piece
from garmentimage.utils.seam import Seam
from garmentimage.utils.template import Template
from garmentimage.utils.template_panel import TemplatePanel
from garmentimage.utils.utils import GarmentImageType
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


def decode_garmentimage(
    garmentimage: GarmentImageType,
    output_file_path: Optional[str] = None,
    visualize: bool = False,
    visualize_faces: bool = False,
    use_vertex_constraints: bool = True,
    predefined_scale: float = 2.75,
    reconstruct_spec_json: bool = False,
    inside_theshold: float = 0.5,
    garment_type: Optional[str] = None,
    strict_garment_type: bool = True,
    reject_two_pieces: bool = False,
    desirable_piece_num: Optional[int] = None,
    n_tries: int = 5,
    address_inside_seam: bool = False,
) -> Optional[Dict[str, Any]]:
    visualize_faces = visualize_faces or visualize
    if reconstruct_spec_json:
        assert garment_type is not None or desirable_piece_num is not None, (
            "garment_type or desirable_piece_num should be specified"
        )
        assert garment_type in [
            None,
            "dress",
            "jumpsuit",
            "dress_sleeveless",
            "jumpsuit_sleeveless",
            "dress_sleeveless_centerseparated_skirtremoved",
            "dress_sleeveless_skirtremoved",
            "dress_centerseparated_skirtremoved",
            "dress_skirtremoved",
            "unmerged_dress",
            "unmerged_dress_sleeveless",
            "jumpsuit_centerseparated",
            "jumpsuit_sleeveless_centerseparated",
            "unmerged_dress_centerseparated",
            "one_genus_jumpsuit_sleeveless",
            "merged_jumpsuit_sleeveless",
        ], (
            f"garment_type should be either dress, jumpsuit, dress_sleeveless, or jumpsuit_sleeveless, but got {garment_type}"
        )

    if garment_type is not None and desirable_piece_num is None:
        panel_to_desirable_piece_num = {
            "dress_sleeveless": 1,
            "dress": 3,
            "jumpsuit_sleeveless": 3,
            "jumpsuit": 5,
            "dress_sleeveless_centerseparated_skirtremoved": 2,
            "dress_sleeveless_skirtremoved": 1,
            "dress_centerseparated_skirtremoved": 4,
            "dress_skirtremoved": 3,
            "unmerged_dress": 4,
            "unmerged_dress_sleeveless": 2,
            "jumpsuit_centerseparated": 6,
            "jumpsuit_sleeveless_centerseparated": 4,
            "unmerged_dress_centerseparated": 5,
            "one_genus_jumpsuit_sleeveless": 4,
            "merged_jumpsuit_sleeveless": 1,
        }
        desirable_piece_num = panel_to_desirable_piece_num[garment_type]

    if output_file_path is not None:
        output_file_dir_path = os.path.dirname(output_file_path)
        front_output_file_path = os.path.join(
            output_file_dir_path,
            os.path.splitext(os.path.basename(output_file_path))[0] + "_front.png",
        )
        back_output_file_path = os.path.join(
            output_file_dir_path,
            os.path.splitext(os.path.basename(output_file_path))[0] + "_back.png",
        )
        spec_json_file_path = os.path.join(
            output_file_dir_path,
            os.path.splitext(os.path.basename(output_file_path))[0]
            + "_specification.json",
        )
    template_panel = TemplatePanel()
    # convert the garment image to np array
    if isinstance(garmentimage, str):
        garmentimage = np.load(garmentimage)
    if isinstance(garmentimage, torch.Tensor):
        if len(garmentimage.shape) == 4:
            garmentimage = garmentimage[0]
        garmentimage = garmentimage.cpu().detach().numpy()

    template_panel.convert_from_np_array(garmentimage, inside_threshold=inside_theshold)

    for i, template in enumerate(template_panel.templates):
        if i == 0 or i == 1:
            is_reversed = False if i == 0 else True
            try:
                template.reconstruct_pieces_from_faces(
                    is_reversed=is_reversed,
                    reject_two_pieces=reject_two_pieces,
                    desirable_piece_num=desirable_piece_num,
                    n_tries=n_tries,
                )
            except Exception as e:
                raise e
            if visualize_faces:
                template.visualize_faces(
                    output_file_path=(
                        front_output_file_path.replace(".png", "_faces.png")
                        if i == 0
                        else back_output_file_path.replace(".png", "_faces.png")
                    )
                    if output_file_path is not None
                    else None
                )

    new_template_panel = TemplatePanel()
    for i, (new_template, template) in enumerate(
        zip(new_template_panel.templates, template_panel.templates)
    ):
        if i == 0 or i == 1:
            Decoder.decode_embedded_template_to_piece_edge(
                template,
                new_template,
                use_vertex_constraints=use_vertex_constraints,
                address_inside_seam=address_inside_seam,
            )
            if output_file_path is not None:
                new_template.visualize_meshes(
                    output_file_path=front_output_file_path
                    if i == 0
                    else back_output_file_path
                )
            elif visualize:
                new_template.visualize_meshes()

    if not reconstruct_spec_json:
        return

    new_template_front = new_template_panel.templates[0]
    new_template_back = new_template_panel.templates[1]

    new_template_front.assign_global_index_to_edges()
    new_template_back.assign_global_index_to_edges()

    side_by_side_stitched_edges_pairs_front = (
        Template.correspond_side_by_side_stitched_edges(new_template_front)
    )
    side_by_side_stitched_edges_pairs_back = (
        Template.correspond_side_by_side_stitched_edges(new_template_back)
    )
    new_template_front.side_by_side_stitched_edges_pairs = (
        side_by_side_stitched_edges_pairs_front
    )
    new_template_back.side_by_side_stitched_edges_pairs = (
        side_by_side_stitched_edges_pairs_back
    )

    front_to_back_stitched_edges_pairs = (
        Template.correspond_front_to_back_stitched_edges(
            new_template_front, new_template_back
        )
    )
    new_template_front.front_to_back_stitched_edges_pairs = (
        front_to_back_stitched_edges_pairs
    )
    new_template_back.front_to_back_stitched_edges_pairs = (
        front_to_back_stitched_edges_pairs
    )

    panel_to_edge_info_front = NeuralTailorConverter.convert_to_panel_to_edge_info(
        new_template_front,
        garment_type=garment_type,
        strict_garment_type=strict_garment_type,
        is_front=True,
        predefined_scale=predefined_scale,
    )
    panel_to_edge_info_back = NeuralTailorConverter.convert_to_panel_to_edge_info(
        new_template_back,
        garment_type=garment_type,
        strict_garment_type=strict_garment_type,
        is_front=False,
        predefined_scale=predefined_scale,
    )
    panel_to_edge_info = {**panel_to_edge_info_front, **panel_to_edge_info_back}
    panel_to_edge_info = NeuralTailorConverter.add_front_to_back_stitch_info(
        panel_to_edge_info, new_template_front, new_template_back
    )
    panel_id_to_translation = {
        k: [0.0, 0.0, 18.0]
        if NeuralTailorConverter.judge_front(k)
        else [0.0, 0.0, -18.0]
        for k in panel_to_edge_info.keys()
    }
    panel_id_to_rotation = {k: [0.0, 0.0, 0.0] for k in panel_to_edge_info.keys()}

    spec_json = (
        NeuralTailorConverter.construct_specification_json_from_panel_to_edge_info(
            panel_to_edge_info,
            panel_id_to_translation=panel_id_to_translation,
            panel_id_to_rotation=panel_id_to_rotation,
        )
    )

    if output_file_path is not None:
        with open(spec_json_file_path, "w") as f:
            json.dump(spec_json, f, indent=4)

    return spec_json
