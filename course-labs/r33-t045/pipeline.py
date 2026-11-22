"""
Course Lab R33-T045: Image Preprocessing Contract
Scales float32 pixels from [0..255] to [0..1] with explicit channel axis (4, 4, 1).
Strictly validates dimensions, rejects out-of-range pixels and prevents test-set contamination.
"""
from typing import List, Dict, Any, Tuple
import math

class ImagePreprocessor:
    def __init__(self, target_shape: Tuple[int, int] = (4, 4)):
        self.target_shape = target_shape

    def transform(self, image: List[List[Any]]) -> List[List[List[float]]]:
        """
        Scales 2D grayscale image (4x4) to float32 values in [0.0, 1.0] with channel axis (4, 4, 1).
        Rejects invalid dimensions, NaN/Inf, and out-of-range values.
        """
        if not isinstance(image, list) or len(image) != self.target_shape[0]:
            raise ValueError(f"Invalid height: expected {self.target_shape[0]}, got {len(image) if isinstance(image, list) else type(image)}")

        transformed_img = []
        for r_idx, row in enumerate(image):
            if not isinstance(row, list) or len(row) != self.target_shape[1]:
                raise ValueError(f"Row {r_idx} invalid width: expected {self.target_shape[1]}")
            transformed_row = []
            for c_idx, pixel in enumerate(row):
                if isinstance(pixel, bool) or not isinstance(pixel, (int, float)):
                    raise TypeError(f"Pixel at ({r_idx}, {c_idx}) must be numeric, got {type(pixel).__name__}")
                if math.isnan(pixel) or math.isinf(pixel):
                    raise ValueError(f"Pixel at ({r_idx}, {c_idx}) has invalid float value: {pixel}")
                if pixel < 0 or pixel > 255:
                    raise ValueError(f"Pixel at ({r_idx}, {c_idx}) out of range [0, 255]: {pixel}")

                scaled_val = float(pixel) / 255.0
                # Add explicit channel axis
                transformed_row.append([scaled_val])
            transformed_img.append(transformed_row)

        return transformed_img

    def transform_dataset(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for r in records:
            processed_img = self.transform(r["image"])
            processed.append({
                "record_id": r["record_id"],
                "group_id": r.get("group_id"),
                "label": r.get("label"),
                "processed_image": processed_img,
                "shape": (len(processed_img), len(processed_img[0]), len(processed_img[0][0]))
            })
        return processed
