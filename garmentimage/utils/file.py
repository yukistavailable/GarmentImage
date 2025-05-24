from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from garmentimage.utils.piece import Piece
    from garmentimage.utils.template_panel import TemplatePanel


class File:
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def load(filename: str) -> List[Piece]:
        pieces: List[Piece] = []
        with open(filename, "r") as f:
            lines: List[str] = f.readlines()

        index: int = 0
        while index < len(lines):
            if lines[index].strip() == "<piece>":
                index += 1
                piece, index = Piece.from_file(lines, index)
                pieces.append(piece)
            index += 1

        # TODO: implement load_connectors
        # load_connectors(pieces, lines, index)
        return pieces

    @staticmethod
    def load_binary_as_npy(template_panel: TemplatePanel, filename: str):
        assert filename.endswith(".npy"), (
            f"The filename must end with '.npy', {filename}"
        )
        img = np.load(filename)
        template_panel.convert_from_np_array(img)

    @staticmethod
    def save_binary_as_npy(template_panel: TemplatePanel, filename: str):
        img: np.ndarray = template_panel.convert_to_np_array()
        np.save(filename, img)
