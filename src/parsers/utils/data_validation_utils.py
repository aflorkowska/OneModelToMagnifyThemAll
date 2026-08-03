import ast
from typing import List


class DataValidationUtils:
    """
    Utility class for data validation and conversion operations.
    """

    @staticmethod
    def ensure_list_of_ints(value: any) -> List[int]:
        """
        Transform input value to list of ints.

        Parameters
        ----------
        value : any
            Input value

        Returns
        -------
        List[int]
            List of integers.

        Raises
        ------
        ValueError: Input value must be an integer or a list of integers.
        """
        value = ast.literal_eval(value)
        if isinstance(value, int):
            return [value]
        elif isinstance(value, list):
            return [int(x) for x in value]
        else:
            raise ValueError("Input value must be an integer or a list of integers.")
