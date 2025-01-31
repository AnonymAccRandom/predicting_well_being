from typing import Any, Optional, Union

from src.utils.utilfuncs import NestedDict


class ConfigParser:
    """
    A utility class for parsing and retrieving entries from a multi-level configuration dictionary.

    This class provides static methods for:
    - Parsing configuration entries based on specific criteria (e.g., `var_type` or `name` values).
    - Searching for dictionaries containing a specified key within a nested structure.

    Methods:
        cfg_parser: Retrieves entries from the configuration matching a given `var_type` and optional `name` values.
        find_key_in_config: Finds all dictionaries in the configuration that contain a specific key.
    """

    @staticmethod
    def cfg_parser(
        cfg: Union[dict, NestedDict], var_type: Optional[str], *args: str
    ) -> list[Union[dict, NestedDict]]:
        """
        Parses a multi-level configuration dictionary and retrieves entries based on `var_type` and optional `name` values.

        - If `var_type` is provided, only entries where the "var_type" key matches the provided value are selected.
        - If `var_type` is `None`, all entries are selected.
        - If additional `name` values are provided as arguments, only entries with a "name" key matching any of these
          values are included in the result.

        Args:
            cfg: A multi-level configuration dictionary, potentially containing nested dictionaries and lists.
            var_type: The value that the "var_type" key must match. If `None`, all entries are matched.
            *args: A variable number of strings representing the "name" values to filter by. If not provided, entries are
                   matched only by `var_type`.

        Returns:
            list[dict]: A list of dictionaries meeting the specified criteria:
                        - `var_type` matches the provided value (or is `None`).
                        - If specified, the "name" key matches one of the provided arguments.
        """
        result = []

        def search_in_config(node: Any) -> None:
            """
            Recursively traverses the configuration to find matching entries.

            - If the node is a dictionary and contains a "var_type" key, checks for a match with the specified `var_type`.
            - If `args` are provided, further filters entries by matching the "name" key.
            - Continues searching within nested dictionaries and lists.

            Args:
                node: The current node being traversed, which can be a dictionary, list, or other data type.
            """
            if isinstance(node, dict):
                if "var_type" in node and (
                    var_type is None or node["var_type"] == var_type
                ):
                    if args:
                        if "name" in node and node["name"] in args:
                            result.append(node)
                    else:
                        result.append(node)

                for value in node.values():
                    search_in_config(value)

            elif isinstance(node, list):
                for item in node:
                    search_in_config(item)

        search_in_config(cfg)

        return result

    @staticmethod
    def find_key_in_config(
        cfg: Union[dict, NestedDict], key: str
    ) -> list[Union[dict, NestedDict]]:
        """
        Searches a multi-level configuration dictionary for entries containing a specified key.

        - This method traverses all levels of the configuration, including nested dictionaries and lists.
        - If the key is found in a dictionary, that dictionary is included in the result.

        Args:
            cfg: A multi-level configuration dictionary containing nested dictionaries and lists.
            key: The key to search for in the configuration.

        Returns:
            list[dict]: A list of dictionaries containing the specified key.
        """
        result = []

        def search_for_key(node: Any) -> None:
            """
            Recursively traverses the configuration to find dictionaries containing the specified key.

            - If the current node is a dictionary and contains the key, it is added to the result.
            - The method continues searching within nested dictionaries and lists.

            Args:
                node: The current node being traversed, which can be a dictionary, list, or other data type.
            """
            if isinstance(node, dict):
                if key in node:
                    result.append(node)

                for value in node.values():
                    search_for_key(value)

            elif isinstance(node, list):
                for item in node:
                    search_for_key(item)

        search_for_key(cfg)

        return result
