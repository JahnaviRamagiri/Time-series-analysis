import os


def get_file_path(file_name):
    dir = os.getcwd()
    print("Current work Directory", dir)
    file_path = dir + os.sep + file_name
    print("File Path is ", file_path)

    return file_path
