from os.path import join

import re

from gensim.models import LdaModel
from tqdm import tqdm
import string
import nltk
import pandas as pd
from nltk import PorterStemmer, NLTKWordTokenizer
from nltk.corpus import stopwords

from rail_topics.utils.root import root

# Constants for column names
COL_URL = 'url'
COL_TEXT = 'text'
COL_DATE = 'date'
COL_YEAR = 'year'
COL_TITLE = 'title'
COL_STARS = 'stars'
COL_TOKENS = 'tokens'
COL_COMPANY = 'company'
COL_FULLTEXT = 'full_text'

COLS_STR = [COL_TITLE, COL_TEXT]


class ManagerIO:
    DIR_RESOURCES = 'resources'
    FILE_TRAIN_REVIEWS = 'train_reviews.json'
    FILE_PROCESSED_REVIEWS = 'processed_train_reviews.json'
    FILE_MODEL = 'model'

    EXTRA_STOPWORDS = ['ive', 'im', 'train', 'trains']

    def __init__(self):
        nltk.download('wordnet')
        nltk.download('stopwords')

        self._punctuation_re = '[^0-9a-zA-Z]+'
        self._stemmer = PorterStemmer()
        self._tokeniser = NLTKWordTokenizer()
        self._stopwords = stopwords.words('english')
        self._stopwords.extend(self.EXTRA_STOPWORDS)

    def find_resource(self, file_name, *args):
        """
        Find a file from the resources folder
        :param file_name: The name of the file to find
        :return: 'rail_topics/resources/{file_name}'
        """
        return join(root(), self.DIR_RESOURCES, *args, file_name)

    def preprocess_train_reviews(self, **kwargs):
        df = pd.read_json(self.find_resource(self.FILE_TRAIN_REVIEWS), **kwargs).drop_duplicates().sort_values(
            by=COL_DATE)
        tqdm.pandas()
        df = df.progress_apply(self._format_columns, axis=1)

        # Remove anomalous year from data
        df = df[df[COL_YEAR] != 2011]
        df.to_json(self.find_resource(self.FILE_PROCESSED_REVIEWS), orient='records', indent=2)
        return df

    def _format_columns(self, df):
        """
        Format current DataFrame columns and return only those that are needed for processing
        :param df: The DataFrame to format
        :return: DataFrame: ['full_text', 'tokens', 'company', 'stars', 'year']
        """
        # Get integer from the stars column
        df[COL_STARS] = int(re.search('(\d+)', df[COL_STARS], re.IGNORECASE).group(1))

        # Get company name from url
        df[COL_COMPANY] = df[COL_URL].split('/')[-1].split('.')[1]

        # Get the year from the date
        df[COL_YEAR] = df[COL_DATE].year

        # Create token column
        df = self._create_token_col(df)

        # Keep only relevant columns
        return df[[COL_FULLTEXT, COL_TOKENS, COL_COMPANY, COL_STARS, COL_YEAR]]

    def _create_token_col(self, df):
        """
        Add two new columns to the DataFrame:
            * full_text: Processed title and text columns appended together
            * tokens: Bag of words created from full_text with stopwords removed
        :param df: The input DataFrame
        :return: The modified DataFrame
        """

        # Append Title and Text columns together
        df[COL_FULLTEXT] = df[COL_TITLE] + ' ' + df[COL_TEXT]

        # Convert to lowercase
        df[COL_FULLTEXT] = df[COL_FULLTEXT].lower()

        # Remove punctuation
        df[COL_FULLTEXT] = df[COL_FULLTEXT].translate(str.maketrans('', '', string.punctuation))

        # Remove unicode characters
        df[COL_FULLTEXT] = df[COL_FULLTEXT].encode('ascii', 'ignore').decode()

        # Tokenise the text
        df[COL_TOKENS] = [self._stemmer.stem(w) for w in self._tokeniser.tokenize(df[COL_FULLTEXT]) if
                          w not in self._stopwords and not w.isnumeric() and len(w) > 1]
        return df

    def read_processed_reviews(self):
        """
        Read the processed reviews file from resources dir
        :return: 'rail_topics/resources/processed_train_reviews.json'
        """
        return pd.read_json(self.find_resource(self.FILE_PROCESSED_REVIEWS))

    def save_model(self, model):
        """
        Save a model to resources
        :param model: The model to save
        :return: None
        """
        model.save(self.find_resource(self.FILE_MODEL))

    def load_model(self) -> LdaModel:
        """
        Load a model from resources
        :return: Loaded LDA model
        """
        return LdaModel.load(self.find_resource(self.FILE_MODEL))
