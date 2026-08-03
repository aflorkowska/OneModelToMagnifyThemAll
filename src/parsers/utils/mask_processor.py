import cv2
import csv
import pyvips
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict


class MaskProcessor:
    """
    Utility class for mask processing, creation, and saving operations.
    """

    @staticmethod
    def create_array_from_coordinates(
        image_dim: Tuple[float, float],
        all_polys_coordinates: List[Tuple[float, float]],
        group_to_value: Dict[str, int],
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Create mask with given dimension, based on coordinates of the found polys contours.

        Parameters
        ----------
        image_dim : Tuple[float, float]
            Dimension of the input image (width, height).

        all_polys_coordinates : Dict[str, List[List[Tuple[float, float]]]]
            Dictionary with group types and lists of all found polygons and coordinates.

        group_to_value : Dict[str, int]
            Dictionary mapping group names to unique integer identifiers.

        Returns
        -------
        Tuple[np.ndarray, List[int], List[int]]
            Tuple of (mask_rgba, labels, counts)
        """
        width, height = image_dim
        dimension_converted_from_openslide_to_numpy = (height, width)
        mask = np.zeros(dimension_converted_from_openslide_to_numpy, dtype=np.uint8)

        mask = MaskProcessor.fill_array_with_poly(
            mask, all_polys_coordinates, group_to_value
        )
        labels, counts = np.unique(mask, return_counts=True)
        mask_rgba = MaskProcessor.convert_binary_to_rgb(mask)
        return mask_rgba, labels, counts

    @staticmethod
    def fill_array_with_poly(
        mask: np.ndarray,
        polys: Dict[str, List[List[Tuple[float, float]]]],
        group_to_value: Dict[str, int],
    ) -> np.ndarray:
        """
        Fill mask according to poly coordinates using values assigned by the group_to_value dictionary.

        Parameters
        ----------
        mask : np.ndarray
            Input mask array.

        polys : Dict[str, List[List[Tuple[float, float]]]]
            Dictionary with group types and lists of all found polygons and coordinates of their contours.

        group_to_value : Dict[str, int]
            Dictionary mapping group names to unique integer identifiers.

        Returns
        -------
        np.ndarray
            Updated mask array with polygons filled according to the group_to_value mapping.
        """
        for group, polygons in polys.items():
            value = group_to_value.get(group, 0)
            for polygon in polygons:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], value)

        return mask

    @staticmethod
    def convert_binary_to_rgb(img: np.ndarray) -> np.ndarray:
        """
        Convert binary array to rgb.

        Parameters
        ----------
        img : np.ndarray
            Input numpy array.

        Returns
        -------
        np.ndarray
            RGB image array.
        """
        height, width = img.shape
        rgba_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgba_image[img == 1, :] = [255, 255, 255]
        return rgba_image

    @staticmethod
    def prepare_metadata(image: pyvips.Image) -> Dict[any, any]:
        """
        Copy metadata from loaded image, change format from list to dictionary.

        Parameters
        ----------
        image : pyvips.Image
            Input image

        Returns
        -------
        Dict[any, any]
            Dictionary with image metadata.
        """
        image_copy = image.copy()
        properties = image_copy.get_fields()
        metadata = {}
        for field in properties:
            metadata[field] = image_copy.get(field)

        return metadata

    @staticmethod
    def save_mask_as_tiff(metadata: Dict, mask: np.ndarray, output_path: Path):
        """
        Save created mask in .tiff format using pyvips library.

        Parameters
        ----------
        metadata : Dict
            Metadata of input image.

        mask : np.ndarray
            Created mask to save.

        output_path : Path
            Path for file saving.
        """
        binary_mask = pyvips.Image.new_from_memory(
            mask.tobytes(), mask.shape[1], mask.shape[0], 3, "uchar"
        )
        binary_mask.tiffsave(
            output_path.with_suffix(".tiff"),
            tile=True,
            pyramid=True,
            compression="jpeg",
            Q=75,
        )

        binary_mask.set_type(
            pyvips.GValue.gstr_type,
            "image-description",
            metadata.get("openslide.comment", ""),
        )

    @staticmethod
    def save_labels_explanation_to_csv(
        group_to_value: Dict[str, int], csv_filepath: Path
    ) -> None:
        """
        Save the group_to_value dictionary to a CSV file.

        Parameters
        ----------
        group_to_value : Dict[str, int]
            Dictionary mapping group names to unique integer identifiers.

        csv_filepath : Path
            Path to the CSV file where the data will be saved.
        """
        with csv_filepath.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Group", "Value"])
            for group, value in group_to_value.items():
                writer.writerow([group, value])
