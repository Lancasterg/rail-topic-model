from rail_topics.utils.manager_io import ManagerIO

"""
First step of processing pipeline -> preprocess the reviews
"""


def run_pipeline():
    ManagerIO().preprocess_train_reviews()


if __name__ == '__main__':
    run_pipeline()
