# Rail review topic modelling

This project requires Python version 8.

## How to run 

```bash

# create a new virtual environment
$ python -m virtualenv topic-model-venv

# Activate virtual environment
$ source topic-model-venv/bin/activate

# Install requirements
$ pip install -r requirements.txt

# Preprocess the reviews
$ python rail_topics/pipeline/preprocess_reviews.py

# Run the topic model
$ python rail_topics/pipeline/process_reviews.py
```