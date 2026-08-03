import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict
from parsers.utils.data_validation_utils import DataValidationUtils


class AnnotationParser:
    """
    Utility class for parsing XML annotations and extracting annotation data.
    """

    @staticmethod
    def parse_annotations(
        xml_filepath: Path, unit_scaling_factor: Tuple[float, float]
    ) -> Dict[str, List[List[Tuple[float, float]]]]:
        """
        Parse annotations coordinates from xml file.

        Parameters
        ----------
        xml_filepath : Path
            Path to the specific xml file.

        unit_scaling_factor : Tuple[float, float]
            Scalling factors, depending on unit in xml.

        Returns
        -------
        Dict[str, List[List[Tuple[float, float]]]]
            A dictionary where keys are the values of 'PartOfGroup' and values are lists of coordinates.
        """
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        categorized_coordinates = {}
        for annotation in root.findall(".//Annotation"):
            part_of_group = annotation.get("PartOfGroup")
            coords_temp = []
            coords = annotation.find("Coordinates")
            if coords is None:
                continue
            for coord in coords.findall("Coordinate"):
                x = float(coord.get("X")) * unit_scaling_factor[0]
                y = float(coord.get("Y")) * unit_scaling_factor[1]
                coords_temp.append((x, y))

            if part_of_group not in categorized_coordinates:
                categorized_coordinates[part_of_group] = []

            categorized_coordinates[part_of_group].append(coords_temp)
        return categorized_coordinates

    @staticmethod
    def find_unique_part_of_groups(xml_folder_path: Path) -> Dict[str, int]:
        """
        Find all unique 'PartOfGroup' values from XML files in a specified folder and
        map them to unique integer values starting from 1, with 0 reserved for background.

        Parameters
        ----------
        xml_folder_path : Path
            Path to the folder containing XML files.

        Returns
        -------
        Dict[str, int]
            A dictionary mapping unique 'PartOfGroup' values to unique integer identifiers.
        """
        unique_groups = set()

        for xml_file in xml_folder_path.glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for annotation in root.findall(".//Annotation"):
                part_of_group = annotation.get("PartOfGroup")
                if part_of_group:
                    unique_groups.add(part_of_group)

        unique_groups_dict = {
            group: idx + 1 for idx, group in enumerate(sorted(unique_groups))
        }
        unique_groups_dict["background"] = 0

        return unique_groups_dict

    @staticmethod
    def find_weak_label(df: pd.DataFrame, img_path: Path) -> List[int]:
        """
        Find weak label for the input image_path

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with information about weak labals, has to contains 2 columns: image_id and weak_labels.

        img_path : Path
            Input image path.

        Returns
        -------
        List[int]
            List of weak labels for the image.
        """
        filename = img_path.stem
        result = df[df["image_id"].apply(lambda x: filename in Path(x).stem)]
        if not result.empty:
            return DataValidationUtils.ensure_list_of_ints(
                result["weak_labels"].values[0]
            )
        else:
            return []
