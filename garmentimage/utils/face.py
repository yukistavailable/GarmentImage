from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from garmentimage.utils.vertex2d import Vector2, Vertex2D

if TYPE_CHECKING:
    from garmentimage.utils.edge2d import Edge2D
    from garmentimage.utils.seam import Seam


class Face2D:
    def __init__(self, *edges):
        self.index: Optional[int] = None
        self.grid_x: int = 0
        self.grid_y: int = 0
        self.T: List[List[float]] = [
            [0.0, 0.0],
            [0.0, 0.0],
        ]  # Placeholder for transformation matrix
        self.piece: float = 0.0
        self.inside: int = 0  # Placeholder for inside/outside flag
        self.H_EDGE0: Optional[Vector2] = None
        self.V_EDGE0: Optional[Vector2] = None
        self.H_EDGE1: Optional[Vector2] = None
        self.V_EDGE1: Optional[Vector2] = None
        if len(edges) > 0:
            self.edges: List[Edge2D] = list(edges)
            for i, edge in enumerate(edges):
                self.add_edge(i, edge, edges[(i + 1) % len(edges)])

    def __str__(self):
        return f"Face2D(inside: {self.inside}, x: {self.grid_x}, y: {self.grid_y}, {self.edges})"

    def __repr__(self):
        return self.__str__()

    def add_edge(self, i: int, e: Edge2D, next_edge: Edge2D):
        self.edges[i] = e
        if next_edge.contains(e.end):
            e.left_face = self
        else:
            e.right_face = self

    def edges_method(self, i: int) -> Edge2D:
        # Util.mod is not a built-in Python function, so using % operator for modulus
        return self.edges[i % len(self.edges)]

    def get_vertex(self, i: int) -> Vertex2D:
        return self.edges_method(i - 1).get_common_vertex(self.edges_method(i))

    def get_vertices(self) -> List[Vertex2D]:
        return [self.get_vertex(i) for i in range(len(self.edges))]

    def get_center(self) -> Vertex2D:
        center = Vertex2D()
        for vertex in self.get_vertices():
            center.x += vertex.x
            center.y += vertex.y
        center.x /= len(self.edges)
        center.y /= len(self.edges)
        return center

    @staticmethod
    def get_boundary_edges(faces: List[Face2D]) -> List[Edge2D]:
        boundary_edges: List[Edge2D] = []
        checked_edges: List[Edge2D] = []
        for face in faces:
            for edge in face.edges:
                if edge in checked_edges:
                    continue
                if edge.is_boundary():
                    if edge.left_face.inside == 1:
                        boundary_edges.append(edge.shallow_duplicate())
                    else:
                        boundary_edges.append(edge.shallow_duplicate_reverse())
                    if edge.is_cut():
                        boundary_edges.append(
                            boundary_edges[-1].shallow_duplicate_reverse()
                        )
                    checked_edges.append(edge)
        return boundary_edges

    @staticmethod
    def extract_inside_edges_non_none_seam_type_from_faces(
        faces: List[Face2D],
    ) -> List[Edge2D]:
        edges: List[Edge2D] = []
        for face in faces:
            for edge in face.edges:
                if (edge.left_face is not None and edge.left_face.inside == 1) and (
                    edge.right_face is not None and edge.right_face.inside == 1
                ):
                    if edge.seam_type is not None and edge.seam_type != Seam.NONE:
                        edges.append(edge)
        return edges

    @staticmethod
    def find_nearest_vertex_on_faces(faces: List[Face2D], v: Vertex2D) -> Vertex2D:
        """
        Get the grid coordinates [x, y] of a given Vertex2D object.
        """
        nearest_vertex: Optional[Vertex2D] = None
        min_distance: float = float("inf")
        for face in faces:
            vertices = face.get_vertices()
            for vertex in vertices:
                distance = vertex.distance(v)
                if distance < min_distance:
                    min_distance = distance
                    nearest_vertex = vertex
        assert nearest_vertex is not None, "No nearest vertex found"
        return nearest_vertex
