import os
import pyvips

################################################################
######################### WINDOWS
import platform

if platform.system() == "Windows":
    from paths.paths import OPENSLIDE_BIN_DIR

    os.add_dll_directory(OPENSLIDE_BIN_DIR)
import openslide

################################################################
import pandas as pd
from pathlib import Path
from typing import Tuple


class WSILoadingUtils:
    """
    Utility class for loading and processing Whole Slide Images (WSI).
    All methods are static and can be called without instantiating the class.
    """

    @staticmethod
    def open_slide(file_path: Path) -> tuple[openslide.OpenSlide, bool]:
        """
        Check if it is possible to open slide with use of OpenSlide library. If not, print exception.

        Parameters
        ----------
        file_path : Path
            Input image

        Returns
        -------
        tuple[openslide.OpenSlide, bool]
            Tuple containing the opened slide object and success boolean flag.

        """
        try:
            slide = openslide.open_slide(file_path)
            slide.read_region((0, 0), 0, (1, 1))
            return slide, True
        except openslide.OpenSlideError as e:
            print(f"Error opening slide: {e}")
            return None, False
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None, False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None, False

    @staticmethod
    def can_open_slide(file_path: Path) -> bool:
        """
        Check if it is possible to open slide.

        Parameters
        ----------
        file_path : Path
            Input image

        Returns
        -------
        bool
            True if the slide can be opened, False otherwise.
        """
        _, success = WSILoadingUtils.open_slide(file_path)
        return success

    @staticmethod
    def load_image(path: Path) -> pyvips.Image:
        """
        Load WSI image using PyVips library.

        Parameters
        ----------
        path : Path
            Path to the specific file.

        Returns
        -------
        pyvips.Image
            Loaded image object.
        """
        img = pyvips.Image.new_from_file(str(path), access="sequential")
        if img.bands > 3:
            img = img.flatten(background=255)
        return img

    @staticmethod
    def get_pixel_size_scalling_factor(img_path: Path) -> Tuple[float, float]:
        """
        Get pixel size at level 0.

        Parameters
        ----------
        img_path : Path
            Path to the image.

        Returns
        -------
        Tuple[float, float]
            Tuple of pixel size (mpp-x, mpp-y).

        Raises
        ------
        ValueError
            If the image cannot be opened.
        """
        slide, is_open = WSILoadingUtils.open_slide(img_path)
        if not is_open:
            raise ValueError(f"Image '{img_path}' can not be opended.")
        slide_properties = slide.properties
        factors = (
            float(slide_properties["openslide.mpp-x"]),
            float(slide_properties["openslide.mpp-y"]),
        )
        slide.close()
        return factors
