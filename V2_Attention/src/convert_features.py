import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import numpy as np


def convert_legacy_features(
    input_path,
    output_dir,
    allow_overwrite=False,
    format_version="2.0"
):
    """
    Converts legacy dictionary-based spatial feature storage (.npy or .npz)
    into a contiguous memory-mapped numeric .npy array accompanied by a manifest.json index.

    Requirements enforced:
    - Input file must exist and be readable.
    - Output directory must be distinct from input path.
    - Destination overwrite protection:
      * Default behavior strictly refuses overwrites if destination exists and is non-empty.
      * On Windows, safely replacing an existing populated destination cannot be guaranteed
        due to file-locking and non-atomic directory replacement. Replacement is refused and a
        new destination directory is required to protect existing valid stores.
    - Validates image identifiers: unique, non-empty strings.
    - Validates feature shapes: accepts (64, 2048), (8, 8, 2048), (1, 64, 2048), normalizing each to (64, 2048).
    - Rejects invalid dimensions, non-finite values (NaN, Inf).
    - Writes through temporary staging directory, promoted atomically upon successful completion.
    - Memory efficiency: Preallocates disk-backed memmap with mode='w+' and writes row-by-row,
      avoiding allocating a second full-dataset array in Python RAM.
    - Interrupted or failed conversions clean up staging directories and leave source and existing
      destinations 100% intact.
    """
    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input feature file not found at: {input_file}")

    output_path = Path(output_dir).resolve()
    if output_path == input_file:
        raise ValueError(f"Output directory cannot be the same as input file: {input_file}")
    if output_path == input_file.parent and output_path.name == input_file.name:
        raise ValueError("Output directory conflicts with input path.")

    # Guard against accidental overwrite and unsafe replacement
    if output_path.exists():
        if output_path.is_file():
            raise FileExistsError(f"Output path '{output_path}' already exists as a file.")
        existing_files = list(output_path.iterdir())
        if existing_files:
            if not allow_overwrite:
                raise FileExistsError(
                    f"Output directory '{output_path}' already exists and contains {len(existing_files)} file(s). "
                    "Default behavior refuses overwrites. Specify a new destination directory."
                )
            if sys.platform == "win32" or os.name == "nt":
                raise FileExistsError(
                    f"Output directory '{output_path}' already exists and contains {len(existing_files)} file(s). "
                    "On Windows, safely replacing an existing populated directory cannot be guaranteed "
                    "(file-locking semantics prevent atomic directory replacement). "
                    "Refusing replacement to preserve existing destination; please specify a new destination directory."
                )

    print(f"Loading legacy features from: {input_file}...")
    # One-time peak memory note: deserializing the legacy pickled dictionary
    # requires loading the legacy dictionary into memory once during conversion.
    data = np.load(str(input_file), allow_pickle=True)
    if hasattr(data, "files"):
        raw_dict = {k: data[k] for k in data.files}
    elif hasattr(data, "item"):
        raw_dict = data.item()
    else:
        raw_dict = dict(data)

    if not isinstance(raw_dict, dict):
        raise TypeError(f"Legacy feature data must deserialize to a dict, got {type(raw_dict).__name__}")

    if not raw_dict:
        raise ValueError("Legacy feature dictionary is empty.")

    # Stable, deterministic sorting of image keys
    image_names = sorted(raw_dict.keys())
    num_images = len(image_names)
    print(f"Discovered {num_images} image entries in legacy feature dictionary.")

    # Check for empty or invalid identifiers
    for idx, name in enumerate(image_names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid image identifier at index {idx}: {name!r}")

    # Ensure parent directory of output exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Create temporary staging directory in the same parent folder (ensures atomic rename on same volume)
    staging_dir = output_path.parent / f"{output_path.name}.tmp_convert_{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, exist_ok=True)

    target_array_filename = "features.npy"
    staging_array_path = staging_dir / target_array_filename
    manifest_filename = "manifest.json"
    staging_manifest_path = staging_dir / manifest_filename

    mmap_out = None
    try:
        # Preallocate disk-backed memory-mapped array directly on storage (mode='w+')
        # Avoids allocating a second copy of dataset in Python heap memory
        print(f"Preallocating memory-mapped array of shape ({num_images}, 64, 2048) on disk...")
        mmap_out = np.lib.format.open_memmap(
            str(staging_array_path),
            mode="w+",
            dtype=np.float32,
            shape=(num_images, 64, 2048)
        )

        image_to_index = {}
        for row_idx, img_name in enumerate(image_names):
            raw_feat = raw_dict[img_name]
            feat_arr = np.asarray(raw_feat, dtype=np.float32)

            # Shape validation & per-image normalization
            if feat_arr.shape == (64, 2048):
                norm_feat = feat_arr
            elif feat_arr.shape == (8, 8, 2048):
                norm_feat = feat_arr.reshape(64, 2048)
            elif feat_arr.shape == (1, 64, 2048):
                norm_feat = feat_arr.reshape(64, 2048)
            elif feat_arr.size == 64 * 2048:
                norm_feat = feat_arr.reshape(64, 2048)
            else:
                raise ValueError(
                    f"Invalid feature shape for '{img_name}': {feat_arr.shape} (size: {feat_arr.size}), "
                    "expected (64, 2048), (8, 8, 2048), or (1, 64, 2048)."
                )

            # Numeric validation: reject NaN or Inf
            if not np.all(np.isfinite(norm_feat)):
                raise ValueError(f"Feature array for '{img_name}' contains NaN or infinite values.")

            mmap_out[row_idx] = norm_feat
            image_to_index[img_name] = row_idx

        # Flush memmap buffer to disk and close file handle safely
        mmap_out.flush()
        mmap_obj = getattr(mmap_out, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass
        del mmap_out
        mmap_out = None

        # Build complete manifest
        manifest_data = {
            "format_version": format_version,
            "storage_type": "mmap_numeric_array",
            "feature_file": target_array_filename,
            "num_images": num_images,
            "feature_shape": [64, 2048],
            "full_shape": [num_images, 64, 2048],
            "dtype": "float32",
            "image_to_index": image_to_index
        }

        # Write manifest atomically via temporary file
        tmp_manifest = staging_dir / f"{manifest_filename}.tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_manifest), str(staging_manifest_path))

        # Promotion of staging directory to final output_path
        if output_path.exists():
            if output_path.is_dir() and not any(output_path.iterdir()):
                output_path.rmdir()
            else:
                raise FileExistsError(f"Target directory '{output_path}' already exists.")

        os.replace(str(staging_dir), str(output_path))
        print(f"Successfully converted {num_images} features to memory-mapped store at: {output_path}")

    except Exception as e:
        # Explicitly close memory-mapped file handle before deleting directory on Windows
        if mmap_out is not None:
            mmap_obj = getattr(mmap_out, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
            del mmap_out
            mmap_out = None

        # Cleanup incomplete staging output on failure
        if staging_dir.exists():
            try:
                shutil.rmtree(staging_dir, ignore_errors=True)
            except Exception:
                pass
        raise e

    return output_path / manifest_filename


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="Convert legacy V2 dictionary features (.npy/.npz) to memory-mapped store."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to legacy feature file (.npy or .npz)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory for memory-mapped store")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting destination directory if it exists")
    return parser.parse_args(args)


if __name__ == "__main__":
    cli_args = parse_arguments()
    convert_legacy_features(
        input_path=cli_args.input,
        output_dir=cli_args.output,
        allow_overwrite=cli_args.overwrite
    )
