import csv
from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent


def save_list(list_name: list, file_path: str) -> None:
    """Save a list as a csv file

    Args:
        list_name (str): The list we want to save
        file_path (str): The file path we want to store the list in
    """
    with open(file_path, 'w') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(list_name)


def import_list(file_path: str) -> list:
    """Import a list from a csv file

    Args:
        file_path (str): The file path where the list is located

    Returns:
        list: The list stored in the csv file
    """
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        stored_list = list(reader)[0]
    return stored_list
