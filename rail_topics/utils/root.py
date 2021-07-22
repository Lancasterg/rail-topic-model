import pathlib


def root():
    """
    Get the root path to the
    :return:
    """
    return pathlib.Path(__file__).parent.parent.resolve()
