from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from garmentimage.utils.piece import Piece


class DrawPanel:
    width: int = 512
    height: int = 512

    def __init__(self, pieces: List[Piece] = []):
        self.pieces: List[Piece] = pieces

    @staticmethod
    def get_width():
        return DrawPanel.width

    @staticmethod
    def get_height():
        return DrawPanel.height

    def get_layer_pieces(
        self, reverse_for_pick: bool = False
    ) -> List[Optional[List[Piece]]]:
        layer_pieces: List[Optional[List[Piece]]] = [None] * 4
        for i in range(4):
            layer_pieces[i] = []
        for piece in self.pieces:
            layer_pieces[piece.layer].append(piece)

        if reverse_for_pick:
            for i in range(4):
                layer_pieces[i].reverse()
        return layer_pieces
