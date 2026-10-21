import os
import json
from pathlib import Path
import numpy as np


class MmapFeatureStore:
    """
    Disk-backed, memory-mapped feature store for spatial image features.

    Reads a contiguous float32 numeric array via np.load(..., mmap_mode="r", allow_pickle=False).
    Provides dictionary-compatible access:
        image_name in store
        store[image_name] -> ndarray of shape (64, 2048), dtype float32
        len(store) -> num_images
        store.get(image_name, default=None)
        store.keys()

    Eliminates full-dataset RAM allocation:
        Zero full-array copying, zero pickle deserialization, zero eager heap allocation.
        Only individual slices are paged into OS cache on demand.
    """

    def __init__(self, manifest_or_dir):
        path = Path(manifest_or_dir).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Feature path does not exist: {path}")

        if path.is_dir():
            manifest_path = path / "manifest.json"
        elif path.suffix.lower() == ".json":
            manifest_path = path
        elif path.suffix.lower() == ".npy" and (path.parent / "manifest.json").exists():
            manifest_path = path.parent / "manifest.json"
        else:
            manifest_path = path

        if not manifest_path.exists():
            raise FileNotFoundError(f"Feature manifest not found at: {manifest_path}")

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read feature manifest at {manifest_path}: {e}")

        self.manifest_path = manifest_path
        self.root_dir = manifest_path.parent.resolve()

        self._validate_manifest(self.metadata)

        self.image_to_index = self.metadata["image_to_index"]
        self.feature_shape = tuple(self.metadata["feature_shape"])
        self.dtype = np.dtype(self.metadata["dtype"])
        self.num_images = int(self.metadata["num_images"])

        # Resolve and validate relative feature array path
        feature_filename = self.metadata["feature_file"]
        p = Path(feature_filename)
        self.array_path = (self.root_dir / p).resolve()
        if not self.array_path.exists():
            raise FileNotFoundError(f"Memory-mapped feature array not found at: {self.array_path}")

        # Open contiguous numeric array strictly with mmap_mode="r" and allow_pickle=False
        try:
            self._data = np.load(str(self.array_path), mmap_mode="r", allow_pickle=False)
        except Exception as e:
            raise RuntimeError(f"Failed to open feature array with memory mapping: {e}")

        if not isinstance(self._data, np.memmap):
            raise RuntimeError(f"Array at {self.array_path} was not opened as a memory-mapped object.")

        # Validate array shape and dtype
        expected_full_shape = (self.num_images, 64, 2048)
        if self._data.shape != expected_full_shape:
            self.close()
            raise ValueError(
                f"Feature array shape mismatch: disk array has shape {self._data.shape}, "
                f"expected {expected_full_shape}"
            )
        if self._data.dtype != np.float32:
            self.close()
            raise ValueError(
                f"Feature array dtype mismatch: disk array has dtype {self._data.dtype}, "
                "expected float32"
            )

    def _validate_manifest(self, meta):
        if not isinstance(meta, dict):
            raise ValueError("Feature manifest must be a JSON object (dictionary).")

        required_keys = [
            "format_version",
            "storage_type",
            "feature_file",
            "num_images",
            "feature_shape",
            "full_shape",
            "dtype",
            "image_to_index"
        ]
        for k in required_keys:
            if k not in meta or meta[k] is None:
                raise ValueError(f"Malformed feature manifest: missing required key '{k}'")

        if meta.get("storage_type") != "mmap_numeric_array":
            raise ValueError(f"Unsupported storage_type: {meta.get('storage_type')}")

        num_images = meta.get("num_images")
        if not isinstance(num_images, int) or isinstance(num_images, bool) or num_images < 0:
            raise ValueError(f"Invalid num_images in manifest: {num_images}, expected non-negative integer.")

        if list(meta.get("feature_shape", [])) != [64, 2048]:
            raise ValueError(f"Invalid feature_shape in manifest: {meta.get('feature_shape')}, expected [64, 2048]")

        if list(meta.get("full_shape", [])) != [num_images, 64, 2048]:
            raise ValueError(
                f"Invalid full_shape in manifest: {meta.get('full_shape')}, "
                f"expected [{num_images}, 64, 2048]"
            )

        dtype_str = str(meta.get("dtype", "")).lower()
        if dtype_str not in ("float32", "<f4", "f4"):
            raise ValueError(f"Invalid dtype in manifest: {meta.get('dtype')}, expected 'float32'")

        # Image index validation: duplicate or missing row indices check
        image_to_index = meta.get("image_to_index")
        if not isinstance(image_to_index, dict):
            raise ValueError("Manifest image_to_index must be a dictionary mapping filenames to indices.")

        if len(image_to_index) != num_images:
            raise ValueError(
                f"Manifest image_to_index count ({len(image_to_index)}) "
                f"does not match num_images ({num_images})"
            )

        indices = []
        for img_name, row_idx in image_to_index.items():
            if not isinstance(img_name, str) or not img_name.strip():
                raise ValueError(f"Invalid image identifier in image_to_index: {img_name!r}")
            if not isinstance(row_idx, int) or isinstance(row_idx, bool):
                raise ValueError(f"Row index for '{img_name}' must be an integer, got {row_idx!r}")
            indices.append(row_idx)

        # Check for duplicate, missing, or out-of-bounds row indices
        if sorted(indices) != list(range(num_images)):
            raise ValueError(
                f"Row indices in image_to_index must be a complete, contiguous 1:1 mapping from 0 to {num_images - 1} "
                "with zero duplicate, missing, or out-of-bounds indices."
            )

        # Path traversal and safety checks for feature_file
        feature_filename = meta.get("feature_file")
        if not isinstance(feature_filename, str) or not feature_filename.strip():
            raise ValueError(f"Invalid feature_file in manifest: {feature_filename!r}")

        feat_path = Path(feature_filename)
        if feat_path.is_absolute() or ".." in feat_path.parts:
            raise ValueError(f"Unsafe feature_file path in manifest: {feature_filename}")

        resolved_array_path = (self.root_dir / feat_path).resolve()
        try:
            resolved_array_path.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(f"Unsafe feature_file path escapes store root directory: {feature_filename}")

    @property
    def is_closed(self):
        """Returns True if the store file handle has been closed."""
        return self._data is None

    def is_memory_mapped(self):
        """Returns True if the underlying array is an active memory-mapped array (np.memmap)."""
        return isinstance(self._data, np.memmap)

    def __contains__(self, key):
        return key in self.image_to_index

    def __len__(self):
        return self.num_images

    def __getitem__(self, key):
        if self._data is None:
            raise RuntimeError("Feature store has been closed.")
        if key not in self.image_to_index:
            raise KeyError(f"Image identifier '{key}' not found in feature store.")
        row_idx = self.image_to_index[key]
        # Direct row slice: accesses only the required feature row from the memmap
        slice_data = self._data[row_idx]
        if slice_data.shape != (64, 2048):
            slice_data = slice_data.reshape(64, 2048)
        return np.asarray(slice_data, dtype=np.float32)

    def get(self, key, default=None):
        if self._data is None:
            raise RuntimeError("Feature store has been closed.")
        if key in self:
            return self[key]
        return default

    def keys(self):
        return self.image_to_index.keys()

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def close(self):
        """Releases the underlying memory-mapped file handle safely."""
        if hasattr(self, "_data") and self._data is not None:
            mmap_obj = getattr(self._data, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
            self._data = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
