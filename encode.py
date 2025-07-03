import argparse
import os
from typing import List

import numpy as np

from garmentimage.utils.draw_panel import DrawPanel
from garmentimage.utils.encoder import Encoder
from garmentimage.utils.file import File
from garmentimage.utils.neural_tailor_converter import NeuralTailorConverter
from garmentimage.utils.piece import Piece
from garmentimage.utils.template_panel import TemplatePanel
from garmentimage.utils.utils2d import visualize_np_garmentimage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a Pattern into a GarmentImage")
    parser.add_argument(
        "-i",
        "--input_pattern_file_path",
        type=str,
        help="Path to a input pattern file",
    )
    parser.add_argument(
        "-p",
        "--parameterize",
        action="store_true",
        help="Whether do parametric manipulation or not",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=str,
        help="Path to the output file",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Whether visualize the result or not",
    )
    parser.add_argument(
        "-r",
        "--random_parameterize",
        action="store_true",
        help="Whether do random parametric manipulation or not",
    )
    parser.add_argument(
        "--disable_align_stitches",
        action="store_false",
        help="Whether disable align_stitches or not",
    )
    parser.add_argument(
        "-f",
        "--force_alignment",
        action="store_true",
        help="Whether align stitches forcibly or not",
    )
    parser.add_argument(
        "-y",
        "--y_axis_alignment",
        action="store_true",
        help="Whether align stitches or not",
    )
    parser.add_argument(
        "-w",
        "--wb_alignment",
        action="store_true",
        help="Whether align stitches of waist band or not",
    )
    parser.add_argument(
        "--dart_alignment",
        action="store_true",
        help="Whether align stitches of dart or not",
    )
    parser.add_argument(
        "--skirt_alignment",
        action="store_true",
        help="Whether align stitches of skirt or not. Deprecated",
    )
    parser.add_argument(
        "--front_to_back_alignment",
        action="store_true",
        help="Whether align front_to_back stitches or not",
    )
    parser.add_argument(
        "-m",
        "--merged_panels",
        nargs="*",
        help="Names of panels to be merged",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.75,
        help="Scale of the garment",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Whether seam-type match strictly or not",
    )

    args = parser.parse_args()
    input_pattern_file_path = args.input_pattern_file_path
    parameterize = args.parameterize
    random_parameterize = args.random_parameterize
    output_file_path = args.output_file_path
    force_alignment = args.force_alignment
    y_axis_alignment = args.y_axis_alignment
    wb_alignment = args.wb_alignment
    merged_panels = args.merged_panels
    predefined_scale = args.scale
    # removed_panels = ["skirt_front", "skirt_back"]
    removed_panels = None

    input_pattern_file_ext = os.path.splitext(input_pattern_file_path)[1]
    if input_pattern_file_ext == ".json":
        # Load the pattern from a JSON file
        # `pieces_original_shape` is used to store the original shape of the pieces, which is needed for constraint encoding
        pieces_original_shape: List[Piece] = NeuralTailorConverter.load_spec_json(
            input_pattern_file_path,
            parameterize=parameterize or random_parameterize,
            random_parameterize=random_parameterize,
            align_stitches=args.disable_align_stitches,
            force_alignment=False,
            y_axis_alignment=y_axis_alignment,
            wb_alignment=False,
            dart_alignment=False,
            skirt_alignment=False,
            front_to_back_alignment=False,
            removed_panels=removed_panels,
            merged_panels=merged_panels,
            predefined_scale=predefined_scale,
        )
        draw_panel_original_shape = DrawPanel(pieces_original_shape)
        template_panel_original_shape = TemplatePanel()
        template_panel_original_shape.load_pieces(
            pieces_original_shape,
            update_corners=False,
            update_points=False,
            # pieces_original_shape
        )

        # Load the pattern from a JSON file
        # `pieces` is deformed and translated to guarantee that the stitches are aligned
        pieces: List[Piece] = NeuralTailorConverter.load_spec_json(
            input_pattern_file_path,
            parameterize=parameterize or random_parameterize,
            random_parameterize=random_parameterize,
            align_stitches=True,
            force_alignment=force_alignment,
            y_axis_alignment=y_axis_alignment,
            wb_alignment=wb_alignment,
            dart_alignment=args.dart_alignment,
            skirt_alignment=args.skirt_alignment,
            front_to_back_alignment=args.front_to_back_alignment,
            removed_panels=removed_panels,
            merged_panels=merged_panels,
            predefined_scale=predefined_scale,
        )
        draw_panel = DrawPanel(pieces)
        template_panel = TemplatePanel()
        template_panel.load_pieces(pieces, update_corners=False, update_points=False)
    else:
        raise ValueError(
            f"Invalid file format. Only .json is supported but got {input_pattern_file_ext}"
        )

    Piece.visualize_pieces(pieces, output_file_path="output_debug4/pieces.png")
    encoder = Encoder()
    template_panel.encode(
        draw_panel,
        encoder,
        draw_panel_original_shape=draw_panel_original_shape,
        strict_seam_type_match=args.strict,
    )
    if output_file_path is None:
        output_file_path = os.path.join(
            "output", os.path.splitext(input_pattern_file_path)[0] + ".npy"
        )
    # if output file path does not exist, create it
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    File.save_binary_as_npy(template_panel, output_file_path)

    if args.visualize:
        garmentimage_np = np.load(output_file_path)
        visualize_np_garmentimage(
            garmentimage_np,
            deform_scale=0.6,
            output_file_path=output_file_path.replace(".npy", "_with_deform.png"),
        )
        for i, template in enumerate(template_panel.templates):
            if i == 0 or i == 1:
                output_dir_path = os.path.dirname(output_file_path)
                output_file_path_for_pieces = os.path.join(
                    output_dir_path,
                    os.path.splitext(os.path.basename(output_file_path))[0]
                    + f"_pieces_{'front' if i == 0 else 'back'}.png",
                )
                output_file_path_for_original_meshes = os.path.join(
                    output_dir_path,
                    (
                        os.path.splitext(os.path.basename(output_file_path))[0]
                        + f"_originl_meshes_{'front' if i == 0 else 'back'}.png"
                    ),
                )
                output_file_path_for_meshes = os.path.join(
                    output_dir_path,
                    (
                        os.path.splitext(os.path.basename(output_file_path))[0]
                        + f"_meshes_{'front' if i == 0 else 'back'}.png"
                    ),
                )
                output_file_path_for_faces = os.path.join(
                    output_dir_path,
                    (
                        os.path.splitext(os.path.basename(output_file_path))[0]
                        + f"_originl_faces_{'front' if i == 0 else 'back'}.png"
                    ),
                )
                template.visualize_pieces(output_file_path=output_file_path_for_pieces)
                template.visualize_meshes(
                    show_constraints=False,
                    show_original_meshes=True,
                    output_file_path=output_file_path_for_original_meshes,
                    markersize=3,
                    axis_off=False,
                )
                template.visualize_meshes(
                    show_constraints=False,
                    output_file_path=output_file_path_for_meshes,
                    markersize=3,
                    axis_off=False,
                )
                template.visualize_faces(
                    output_file_path=output_file_path_for_faces,
                    markersize=4,
                    axis_off=False,
                )
