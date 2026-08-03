"""
Parse script for PANDA (Prostate cANcer graDe Assessment) dataset.

This module uses the PANDAParser to orchestrate the complete parsing pipeline
for the PANDA dataset, including dataset preparation, CSV creation, and k-fold splitting.

Usage:
    python parse_panda.py
"""

from pathlib import Path
from paths.paths import DATA_DIR
from parsers.datasets_parsers.panda_parser import PANDAParser

if __name__ == "__main__":
    """
    Main entry point for PANDA dataset parsing.

    Orchestrates the complete dataset parsing pipeline with k-fold splitting.
    """
    dataset_filename = Path(r"PANDAChallenge")
    kfold = 2
    seed = 42
    logger_filename = r"logger_parse_pandaALL.log"

    parser = PANDAParser(
        dataset_filename=dataset_filename,
        kfold=kfold,
        data_dir=Path(DATA_DIR),
        seed=seed,
    )
    parser.setup_logger(logger_filename)
    parser.logger.info("Starting PANDA dataset parsing")

    try:
        parser.run_dataset_setup()
        parser.logger.info("PANDA dataset parsing completed successfully")
        print("PANDA dataset parsing completed successfully")
    except Exception as e:
        parser.logger.error(
            f"Error during PANDA dataset parsing: {str(e)}", exc_info=True
        )
        print(f"Error during PANDA dataset parsing: {str(e)}")
        raise
