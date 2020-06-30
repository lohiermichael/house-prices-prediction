import csv


def save_list(list_name: list, file_path: str):
    """Save a list as a csv file

    Args:
        list_name (str): The list we want to save
        file_path (str): The file path we want to store the list in
    """
    with open(file_path, 'w') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(list_name)
