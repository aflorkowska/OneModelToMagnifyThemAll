import os
from pathlib import Path
from typing import Dict


class FileSystemUtils:
    """
    Utility class for file system operations including finding files and mapping related files.
    """

    @staticmethod
    def find_files_with_extension(root_dir: Path, extensions: list[str]) -> list[Path]:
        """
        Search root dir to find files with declared extensions.

        Parameters
        ----------
        root_dir : Path
            Path to the main dir you want to be searched.
        extensions : list of string
            List of desirable files' extensions.

        Returns
        -------
        list[Path]
            List of all found files with corresponding extension.

        Raises
        ------
        FileNotFoundError: If rootDir was not found.
        """
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dir '{root_dir}' does not exist.")

        file_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                for extension in extensions:
                    if filename.endswith(extension):
                        file_paths.append(Path(os.path.join(dirpath, filename)))

        return file_paths

    @staticmethod
    def map_images_to_annotations(
        img_paths: list[Path], annotation_paths: list[Path]
    ) -> list[Dict[str, Path]]:
        """
        Map images and their corresponding annotations.

        Parameters
        ----------
        img_paths : list[Path]
            List of all found images' paths.
        annotation_paths : list[Path]
            List of all found annotations' paths.

        Returns
        -------
        list[dict]
            List of dictionaries containing all found pairs: paths to corresponding images and annotations.

        Raises
        ------
        FileNotFoundError: If at least one of the lists is empty.
        """
        if not img_paths or not annotation_paths:
            raise FileNotFoundError(f"Input list is empty")

        mappedData = []
        for img_path in img_paths:
            for annotation_path in annotation_paths:
                if str(img_path.stem) in str(annotation_path):
                    annotation_paths.remove(annotation_path)
                    mappedData.append(
                        {"image": img_path, "annotation": annotation_path}
                    )
                    break
        return mappedData

    @staticmethod
    def find_matching_file(reference_path: Path, files_to_search: list[Path]):
        """
        Find matching reference_path in the list of files_to_search.

        Parameters
        ----------
        reference_path : Path
            Reference path eg filename

        files_to_search : list[Path]
            List of paths to search.

        Returns
        -------
        Path | None
            Matching file path or None if not found.
        """
        filename_without_ext = reference_path.stem
        for file in files_to_search:
            if filename_without_ext in str(file):
                return file
        return None

    @staticmethod
    def check_if_required_data_for_dataset_summary_generation_exist(
        images_dir: Path | None,
        bg_mask_dir: Path | None,
        gt_mask_dir: Path | None,
        df_weak_labels: Path | None,
        df_all_labels_lst: Path | None,
        images_class_dist_csv: Path | None,
    ):
        """
        Check if data needed for dataset summary generation exist.

        Parameters
        ----------
        images_dir : Path | None
            Input image dir path.

        bg_mask_dir : Path | None
            Input bg mask dir path.

        gt_mask_dir : Path | None
            Input gt mask dir path.

        df_weak_labels : Path | None
            Input path to dataframe with weak labels.

        df_all_labels_lst : list[Path | None]
            Input path to dataframe with all labels explanation.

        images_class_dist_csv : Path | None
            Input path to dataframe with class distribiution for all the images.

        Returns
        -------
        bool | None
            None if validation fails.
        """
        if images_dir == None or bg_mask_dir == None:
            print(
                f"\nCheck required dirs. They may not be passed: images: {images_dir}, bg masks: {bg_mask_dir}"
            )
            return None

        if not images_dir.exists() or not bg_mask_dir.exists():
            print(
                f"\nCheck required dirs. They may not exist: images: {images_dir}, bg masks: {bg_mask_dir}"
            )
            return None

        if gt_mask_dir != None and not gt_mask_dir.exists():
            print(
                f"\nCheck! Path: {gt_mask_dir} does not exists. The dense masks were not processed."
            )
            return None

        if df_weak_labels != None and not df_weak_labels.exists():
            print(
                f"\nCheck! Path: {df_weak_labels} does not exists. The weak labels were not processed."
            )
            return None

        for path in df_all_labels_lst:
            if path != None and not path.exists():
                print(
                    f"\nCheck! Path: {path} does not exists. The weak labels were not processed."
                )
                return None

        if images_class_dist_csv != None and not images_class_dist_csv.exists():
            print(
                f"\nCheck! Path: {images_class_dist_csv} does not exists. The weak labels were not processed."
            )
            return None
