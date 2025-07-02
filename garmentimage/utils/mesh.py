from __future__ import annotations

import queue
from copy import copy
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt

from garmentimage.utils.edge2d import Edge2D
from garmentimage.utils.face import Face2D
from garmentimage.utils.seam import Seam
from garmentimage.utils.vertex2d import Vector2, Vertex2D


class Mesh2D:
    def __init__(
        self,
        inside_faces: Optional[List[Face2D]] = None,
        seam_edges: Optional[Set[Edge2D]] = None,
        integrate_adjacent_face_edges: bool = False,
    ):
        """
        Initialize a Mesh2D object with optional inside faces and seam edges.

        This constructor sets up the 2D mesh by initializing its faces, vertices,
        and edges.
        If `inside_faces` is provided, it constructs the mesh based on
        these faces.
        If `seam_edges` is also provided, it takes into account the
        seam edges while constructing the mesh, ensuring appropriate merging of
        adjacent faces along the seams.

        Parameters
        ----------
        inside_faces: Optional[List[Face2D]]
            A list of Face2D objects that represent the faces inside the mesh. If None, the mesh will not be constructed.
        seam_edges: Optional[Set[Edge2D]]
            A set of Edge2D objects that represent the seam edges, the boundaries of the mesh.
            These edges will be considered during the merging of adjacent faces. If None, the mesh is created by directly copying the inside faces.
        """
        self.faces: List[Face2D] = []
        self.vertices: List[Vertex2D] = []
        self.edges: List[Edge2D] = []
        if inside_faces is None:
            return

        if seam_edges is None:
            # if seam_edges is not provided, the mesh is created by copying the inside_faces
            old_to_new_edges: Dict[Edge2D, Edge2D] = {}
            old_to_new_vertices: Dict[Vertex2D, Vertex2D] = {}
            new_vertices = set()
            new_edges = set()
            for face in inside_faces:
                n: int = len(face.edges)
                _new_edges: List[Edge2D] = []
                for i in range(n):
                    if integrate_adjacent_face_edges:
                        _new_edges.append(
                            self.get_or_create_edge(
                                face.edges[i], old_to_new_edges, old_to_new_vertices
                            )
                        )
                    else:
                        new_edge = Edge2D(face.edges[i].start, face.edges[i].end)
                        _new_edges.append(new_edge)
                        new_edges.add(new_edge)
                        new_vertices.add(face.edges[i].start)
                        new_vertices.add(face.edges[i].end)
                new_face: Face2D = Face2D(*_new_edges)
                new_face.grid_x = face.grid_x
                new_face.grid_y = face.grid_y
                new_face.T = copy(face.T)
                new_face.inside = 1
                self.faces.append(new_face)

            for v in old_to_new_vertices.values():
                self.vertices.append(v)

            for edge in old_to_new_edges.values():
                self.edges.append(edge)

            for v in new_vertices:
                self.vertices.append(v)

            for e in new_edges:
                self.edges.append(e)

            self.set_indices()

        else:
            face_to_old_vertex_to_new_vertex: Dict[
                Face2D, Dict[Vertex2D, Vertex2D]
            ] = {}
            vertex_to_merged_vertex: Dict[Vertex2D, Vertex2D] = {}

            for face in inside_faces:
                old_vertex_to_new_vertex: Dict[Vertex2D, Vertex2D] = {}
                face_to_old_vertex_to_new_vertex[face] = old_vertex_to_new_vertex
                for vertex in face.get_vertices():
                    new_vertex: Vertex2D = Vertex2D(vertex.x, vertex.y)
                    new_vertex.TX = Vector2(vertex.TX)
                    new_vertex.TY = Vector2(vertex.TY)
                    new_vertex.grid_xy = copy(vertex.grid_xy)
                    old_vertex_to_new_vertex[vertex] = new_vertex

            already_merged: Set[Edge2D] = set()
            # boundary_edges: List[Edge2D] = Face2D.get_boundary_edges(inside_faces)
            for face in inside_faces:
                for edge in face.edges:
                    if edge in already_merged:
                        continue
                    already_merged.add(edge)
                    adjacent_face: Face2D = edge.get_opposite_face(face)
                    if adjacent_face is None:
                        continue
                    if adjacent_face not in inside_faces:
                        continue
                    if Mesh2D.check_if_edge_is_seam(edge, seam_edges):
                        # if Mesh2D.check_if_edge_is_seam(edge, boundary_edges):
                        continue
                    self.merge(
                        face,
                        adjacent_face,
                        edge.start,
                        face_to_old_vertex_to_new_vertex,
                        vertex_to_merged_vertex,
                    )
                    self.merge(
                        face,
                        adjacent_face,
                        edge.end,
                        face_to_old_vertex_to_new_vertex,
                        vertex_to_merged_vertex,
                    )

            merged_vertices: Set[Vertex2D] = set()
            for face in inside_faces:
                n: int = len(face.edges)
                new_edges: List[Optional[Edge2D]] = [None] * n
                for i in range(n):
                    template_edge: Edge2D = face.edges[i]
                    seam_type: int = template_edge.seam_type
                    start: Vertex2D = face_to_old_vertex_to_new_vertex[face][
                        template_edge.start
                    ]
                    start = self.trace_merged_vertex(start, vertex_to_merged_vertex)
                    end: Vertex2D = face_to_old_vertex_to_new_vertex[face][
                        template_edge.end
                    ]
                    end = self.trace_merged_vertex(end, vertex_to_merged_vertex)
                    merged_vertices.add(start)
                    merged_vertices.add(end)

                    if template_edge in seam_edges:
                        new_edges[i] = Edge2D(start, end)
                        self.edges.append(new_edges[i])
                    else:
                        new_edges[i] = self.get_or_create_edge_of_start_end(start, end)
                    new_edges[i].template_edge = template_edge  # Stich!
                    new_edges[i].seam_type = seam_type

                new_face: Face2D = Face2D(*new_edges)
                new_face.grid_x = face.grid_x
                new_face.grid_y = face.grid_y
                new_face.T = copy(face.T)
                new_face.inside = 1
                self.faces.append(new_face)

            self.vertices.extend(merged_vertices)
            self.set_indices()

    @staticmethod
    def check_if_edge_is_seam(edge: Edge2D, seam_edges: Set[Edge2D]) -> bool:
        for seam_edge in seam_edges:
            if edge.same_position_undirected(seam_edge):
                return True
        return False

    @staticmethod
    def get_boundary_vertices_of_undirected_edges(
        edge_list: List[Edge2D],
    ) -> List[Vertex2D]:
        """
        Returns the boundary of the undirected edges.
        The implementation is based on the assumption that the start and end points of the connected edges are the same.
        """
        edge_queue = queue.Queue()
        for edge in edge_list:
            edge_queue.put(edge)
        sorted_vertices = []
        count = 0
        refer_end = True
        prev_edge: Optional[Edge2D] = None
        while not edge_queue.empty():
            count += 1
            if count > 1000:
                raise ValueError("The loop is too long.")
            edge: Edge2D = edge_queue.get()
            if len(sorted_vertices) == 0:
                sorted_vertices.append(edge.start)
                prev_edge = edge
                continue
            if refer_end:
                if prev_edge.end.x == edge.start.x and prev_edge.end.y == edge.start.y:
                    refer_end = True
                    sorted_vertices.append(edge.start)
                    prev_edge = edge
                elif prev_edge.end.x == edge.end.x and prev_edge.end.y == edge.end.y:
                    refer_end = False
                    sorted_vertices.append(edge.end)
                    prev_edge = edge
                else:
                    edge_queue.put(edge)
            else:
                if (
                    prev_edge.start.x == edge.start.x
                    and prev_edge.start.y == edge.start.y
                ):
                    refer_end = True
                    sorted_vertices.append(edge.start)
                    prev_edge = edge
                elif (
                    prev_edge.start.x == edge.end.x and prev_edge.start.y == edge.end.y
                ):
                    refer_end = False
                    sorted_vertices.append(edge.end)
                    prev_edge = edge
                else:
                    if (
                        edge.left_face is not None
                        and edge.left_face.inside == 1
                        and edge.right_face is not None
                        and edge.right_face.inside == 1
                    ):
                        continue
                    edge_queue.put(edge)
        return sorted_vertices

    def get_boundary_vertices(self) -> List[Vertex2D]:
        """
        Returns the sorted boundary vertices of the mesh.
        """
        boundary_edges: List[Edge2D] = [
            edge
            for edge in self.edges
            if edge.seam_type is not None and edge.seam_type != Seam.NONE
        ]
        try:
            result = Mesh2D.get_boundary_vertices_of_undirected_edges(boundary_edges)
        except ValueError:
            # If the boundary edges are not connected, we need to consider only the inside faces, not seam types.
            boundary_edges: List[Edge2D] = [
                edge
                for edge in self.edges
                if (
                    (edge.left_face is None or edge.left_face.inside == 1)
                    and not (edge.right_face is None or edge.right_face.inside == 1)
                )
                or (
                    not (edge.left_face is None or edge.left_face.inside == 1)
                    and (edge.right_face is None or edge.right_face.inside == 1)
                )
            ]
            result = Mesh2D.get_boundary_vertices_of_undirected_edges(boundary_edges)
        return result

    def get_boundary_edges(self) -> List[Edge2D]:
        sorted_boundary_vertices = self.get_boundary_vertices()
        sorted_boundary_edges: List[Edge2D] = []
        for i, v in enumerate(sorted_boundary_vertices):
            original_edge = v.get_common_edge(
                sorted_boundary_vertices[(i + 1) % len(sorted_boundary_vertices)]
            )
            if original_edge is None:
                raise ValueError("The edge is not found.")
            created_edge = (
                original_edge.shallow_duplicate()
                if original_edge.start == v
                else original_edge.shallow_duplicate_reverse()
            )
            sorted_boundary_edges.append(created_edge)
        return sorted_boundary_edges

    def min_x(self) -> float:
        if len(self.vertices) == 0:
            return 0
        min_x_value = min([v.x for v in self.vertices])
        return min_x_value

    def max_x(self) -> float:
        if len(self.vertices) == 0:
            return 0
        max_x_value = self.vertices[0].x
        for vertex in self.vertices:
            if vertex.x > max_x_value:
                max_x_value = vertex.x
        return max_x_value

    def min_y(self) -> float:
        if len(self.vertices) == 0:
            return 0
        min_y_value = self.vertices[0].y
        for vertex in self.vertices:
            if vertex.y < min_y_value:
                min_y_value = vertex.y
        return min_y_value

    def max_y(self) -> float:
        if len(self.vertices) == 0:
            return 0
        max_y_value = self.vertices[0].y
        for vertex in self.vertices:
            if vertex.y > max_y_value:
                max_y_value = vertex.y
        return max_y_value

    def mean_x(self) -> float:
        if len(self.vertices) == 0:
            return 0
        x_sum = sum([v.x for v in self.vertices])
        return x_sum / len(self.vertices)

    def mean_y(self) -> float:
        if len(self.vertices) == 0:
            return 0
        y_sum = sum([v.y for v in self.vertices])
        return y_sum / len(self.vertices)

    def duplicate(self) -> Mesh2D:
        mesh: Mesh2D = Mesh2D()
        mesh_edges: List[Edge2D] = []
        for face in self.faces:
            new_edges: List[Edge2D] = [edge.deep_duplicate() for edge in face.edges]
            mesh_edges.extend(new_edges)
            new_face: Face2D = Face2D(*new_edges)
            new_face.grid_x = face.grid_x
            new_face.grid_y = face.grid_y
            new_face.T = copy(face.T)
            mesh.faces.append(new_face)
        mesh.edges = mesh_edges
        return mesh

    def visualize(self, constraints: Optional[Dict[Vertex2D, Vertex2D]] = None) -> None:
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")

        constraint_marker = "x"

        for edge in self.edges:
            seam_type = edge.seam_type if edge.seam_type is not None else Seam.NONE
            linecolor = Seam.boundary_types_to_color[seam_type]
            if constraints is not None:
                for point in constraints.keys():
                    ax.plot(
                        point.x,
                        point.y,
                        marker=constraint_marker,
                        linestyle="-",
                        color=Seam.constraint_color,
                        alpha=1.0,
                        markersize=10,
                    )
            ax.plot(
                [edge.start.x, edge.end.x],
                [edge.start.y, edge.end.y],
                marker="o",
                linestyle="-",
                color=linecolor,
                alpha=1.0,
            )

        ax.set_xticks(
            range(
                int(min(v.x for v in self.vertices) - 1),
                int(max(v.x for v in self.vertices) + 2),
            ),
            minor=True,
        )
        ax.set_yticks(
            range(
                int(min(v.y for v in self.vertices) - 1),
                int(max(v.y for v in self.vertices) + 2),
            ),
            minor=True,
        )
        plt.show()

    def merge(
        self,
        face0: Face2D,
        face1: Face2D,
        template_vertex: Vertex2D,
        face_to_old_vertex_to_new_vertex: Dict[Face2D, Dict[Vertex2D, Vertex2D]],
        vertex_to_merged_vertex: Dict[Vertex2D, Vertex2D],
    ) -> None:
        """
        Merge vertices from two faces (face0 and face1) at a shared vertex (template_vertex) into a new, unified vertex (new_vertex), while preserving geometric properties and relationships.

        Parameters
        ----------
        face0: Face2D
            The first Face2D object representing the face that is being considered for merging.
        face1: Face2D
            The second Face2D object representing the face that is being considered for merging.
        template_vertex: Vertex2D
            The Vertex2D object representing the vertex at which face0 and face1 meet and should be merged.
        face_to_old_vertex_to_new_vertex: Dict[Face2D, Dict[Vertex2D, Vertex2D]]
            A dictionary mapping each face to another dictionary, which maps original vertices of the face to their duplicated counterparts in the new mesh context.
        vertex_to_merged_vertex: Dict[Vertex2D, Vertex2D]
            A dictionary that tracks which vertices have been merged together, mapping original vertices to their merged counterparts.
        """
        new_vertex: Vertex2D = Vertex2D(template_vertex.x, template_vertex.y)
        v0: Vertex2D = face_to_old_vertex_to_new_vertex[face0][template_vertex]
        v0 = self.trace_merged_vertex(v0, vertex_to_merged_vertex)
        new_vertex.grid_xy = copy(v0.grid_xy)
        new_vertex.TX = v0.TX
        new_vertex.TY = v0.TY
        v1: Vertex2D = face_to_old_vertex_to_new_vertex[face1][template_vertex]
        v1 = self.trace_merged_vertex(v1, vertex_to_merged_vertex)
        vertex_to_merged_vertex[v0] = new_vertex
        vertex_to_merged_vertex[v1] = new_vertex

    def trace_merged_vertex(
        self, vertex: Vertex2D, vertex_to_merged_vertex: Dict[Vertex2D, Vertex2D]
    ):
        """
        Find the final merged vertex for any given vertex by following the chain of merges recorded in vertex_to_merged_vertex.
        """
        while True:
            if vertex in vertex_to_merged_vertex:
                vertex = vertex_to_merged_vertex[vertex]
            else:
                break
        return vertex

    def get_or_create_edge_of_start_end(
        self, start: Vertex2D, end: Vertex2D
    ) -> Optional[Edge2D]:
        common_edge: Optional[Edge2D] = start.get_common_edge(end)
        if common_edge is None:
            common_edge = Edge2D(start, end)
            self.edges.append(common_edge)
        return common_edge

    def get_or_create_edge(
        self,
        edge: Edge2D,
        old_to_new_edges: Dict[Edge2D, Edge2D],
        old_to_new_vertices: Dict[Vertex2D, Vertex2D],
    ) -> Edge2D:
        """
        Retrieve or create an Edge2D object based on the given edge.
        """
        if edge not in old_to_new_edges:
            start: Vertex2D = self.get_vertex(edge.start, old_to_new_vertices)
            end: Vertex2D = self.get_vertex(edge.end, old_to_new_vertices)
            new_edge: Edge2D = Edge2D(start, end)
            new_edge.seam_type = edge.seam_type
            old_to_new_edges[edge] = new_edge
        return old_to_new_edges[edge]

    def get_vertex(self, v: Vertex2D, old_to_new_vertices: Dict[Vertex2D, Vertex2D]):
        """
        Get the corresponding Vertex2D object from the map_vertices dictionary, creating a new one if necessary.
        """
        if v not in old_to_new_vertices:
            old_to_new_vertices[v] = Vertex2D(v.x, v.y)
        return old_to_new_vertices[v]

    def set_indices(self):
        for i, vertex in enumerate(self.vertices):
            vertex.index = i
        for i, edge in enumerate(self.edges):
            edge.index = i
        for i, face in enumerate(self.faces):
            face.index = i

    def find_nearest_vertex(self, v: Vertex2D) -> Vertex2D:
        min: float = 100000
        nearest: Optional[Vertex2D] = None
        for vertex in self.vertices:
            d: float = Vertex2D.distance_static(v, vertex)
            if d < min:
                min = d
                nearest = vertex
        return nearest

    def find_nearest_vertex_specified_seam_type(
        self, v: Vertex2D, seam_type: int
    ) -> Vertex2D:
        min: float = 100000
        nearest: Optional[Vertex2D] = None
        for vertex in self.vertices:
            edges: List[Edge2D] = vertex.get_edges()
            flag = False
            for edge in edges:
                if edge.seam_type == seam_type:
                    flag = True
                    break
            if not flag:
                continue
            d: float = Vertex2D.distance_static(v, vertex)
            if d < min:
                min = d
                nearest = vertex
        return nearest
