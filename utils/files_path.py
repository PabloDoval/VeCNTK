import os


def join_paths(path1, path2):
    return os.path.join(path1, path2)


def is_file(path):
    return os.path.isfile(path)
