import argparse
import os

import numpy as np

from garmentimage.utils.decoder import decode_garmentimage
from garmentimage.utils.utils2d import (
    convert_spec_json_to_polygon_and_save,
    visualize_np_garmentimage,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode a GarmentImage into a Pattern")
    parser.add_argument(
        "-i",
        "--garmentimage_file_path",
        type=str,
        help="Path to the garmentimage file",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Whether visualize the result or not",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=str,
        help="Path to the output file. '_front' or '_back' will be added to the end of the file name",
        default=None,
    )
    parser.add_argument(
        "--address_inside_seam",
        action="store_true",
        help="Whether cut the edges of inside seams",
    )
    parser.add_argument(
        "--garment_type",
        type=str,
        help="Garment type",
        required=True,
    )
    parser.add_argument(
        "--n_tries",
        type=int,
        help="Number of tries to reconstruct the pattern",
        default=10,
    )

    args = parser.parse_args()

    file_path = args.garmentimage_file_path
    output_file_path = args.output_file_path
    visualize = args.visualize
    reconstructed_spec_json = decode_garmentimage(
        file_path,
        output_file_path=output_file_path,
        visualize=visualize,
        use_vertex_constraints=True,
        reconstruct_spec_json=True,
        garment_type=args.garment_type,
        address_inside_seam=args.address_inside_seam,
        n_tries=args.n_tries,
    )
    if output_file_path is not None:
        output_file_dir_path = os.path.dirname(output_file_path)
        if not os.path.exists(output_file_dir_path):
            os.makedirs(output_file_dir_path)

        garmentimage_np = np.load(file_path)
        tmp_output_file_path = output_file_path.replace(".png", "_show_deform.png")
        visualize_np_garmentimage(
            garmentimage_np, tmp_output_file_path, deform_scale=0.6
        )

        if reconstructed_spec_json is not None:
            convert_spec_json_to_polygon_and_save(
                reconstructed_spec_json, output_file_path
            )
