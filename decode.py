import argparse
import os
import numpy as np

from garmentimage.utils.decoder import decode_garmentimage
from garmentimage.utils.utils2d import (
    convert_spec_json_to_polygon_and_save,
    visualize_np_garmentimage,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Decode a GarmentImage into a Pattern")

    parser.add_argument(
        "-i",
        "--garmentimage_file_path",
        type=str,
        required=True,
        help="Path to the garmentimage .npy file",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=str,
        default=None,
        help="Output file path; '_front' or '_back' will be added automatically",
    )
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize the result"
    )
    parser.add_argument(
        "--address_inside_seam",
        action="store_true",
        help="Whether to cut the edges of inside seams",
    )
    parser.add_argument(
        "--garment_type",
        type=str,
        required=True,
        help="Garment type (e.g., 'shirt', 'pants')",
    )
    parser.add_argument(
        "--n_tries", type=int, default=10, help="Number of reconstruction attempts"
    )

    return parser.parse_args()


def ensure_output_directory(path: str):
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def visualize_garment(garmentimage_path: str, output_file_path: str):
    garment_np = np.load(garmentimage_path)
    visual_output_path = output_file_path.replace(".png", "_show_deform.png")
    visualize_np_garmentimage(garment_np, visual_output_path, deform_scale=0.6)


def decode_and_export(args):
    spec_json = decode_garmentimage(
        args.garmentimage_file_path,
        output_file_path=args.output_file_path,
        visualize=args.visualize,
        use_vertex_constraints=True,
        reconstruct_spec_json=True,
        garment_type=args.garment_type,
        address_inside_seam=args.address_inside_seam,
        n_tries=args.n_tries,
    )

    if args.output_file_path:
        ensure_output_directory(args.output_file_path)
        visualize_garment(args.garmentimage_file_path, args.output_file_path)

        if spec_json is not None:
            convert_spec_json_to_polygon_and_save(spec_json, args.output_file_path)


def main():
    args = parse_arguments()
    decode_and_export(args)


if __name__ == "__main__":
    main()
