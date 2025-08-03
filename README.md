# Official Implementation of GarmentImage: Raster Encoding of Garment Sewing Patterns with Diverse Topologies

SIGGRAPH 2025 (Conference Track)

Project Page: https://yukistavailable.github.io/garmentimage.github.io/

Paper: https://www.arxiv.org/abs/2505.02592

# Set up an environment

## Install UV

see [an official document](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

Linux

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Set up environment with UV

```sh
uv sync
```

# Execute commands

## Encode

tops with darts

```
uv run python encode.py -i assets/rand_CFG7B8V834_specification.json -o output/rand_CFG7B8V834.npy -v --dart_alignment --front_to_back_alignment --disable_align_stitches --scale 8
```

dress with sleeves

```
uv run python encode.py -i assets/dress_0XAVEH5G53_specification.json -o output/dress_0XAVEH5G53.npy -v --front_to_back_alignment --disable_align_stitches
```

pants

```
uv run python encode.py -i assets/jumpsuit_sleeveless_0C9A7U6789_specification.json -o output/jumpsuit_sleeveless_0C9A7U6789_specification.npy -v --front_to_back_alignment --disable_align_stitches
```

## Decode

```
uv run python decode.py -i output/rand_CFG7B8V834.npy -v -o output_vis/rand_CFG7B8V834.png --garment_type dress_sleeveless_centerseparated_skirtremoved
```
