from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore, LdaModel
import pandas as pd

from rail_topics.utils.manager_io import COL_TOKENS, ManagerIO, COL_COMPANY, COL_YEAR
from rail_topics.utils.plot import Plotter

"""
Second step of processing pipeline -> fit a topic model and save it to disk
"""

found_topics = {
    'time': ['delay', 'late', 'hour', 'time', 'cancel'],
    'negative': ['stink', 'disgust', 'terribl', 'aw', 'frustrat'],
    'service': ['staff', 'servic', 'journey', 'compani'],
    'first class': ['first', 'class'],
    'booking': ['websit', 'book', 'refund', 'money', 'payment', 'buy', 'ticket']
}


def process_reviews(df, company=None, year=None, save_model=False):

    topic_scores = {key: 0 for key in found_topics}

    # Only process a specific company
    if company is not None:
        print(company)
        df = df[df[COL_COMPANY] == company]

    # Only process a specific year
    if year is not None:
        print(year)
        df = df[df[COL_YEAR] == year]

    # Create Gensim dictionary
    dictionary = Dictionary(df[COL_TOKENS])
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Create a bag of words from each document
    bow_corpus = [dictionary.doc2bow(doc) for doc in df[COL_TOKENS]]

    # Fit an implementation of a Latent Dirichlet Allocation model
    try:
        lda_model = LdaModel(bow_corpus,
                             eta='auto',
                             num_topics=5,
                             id2word=dictionary,
                             passes=10,
                             per_word_topics=True,
                             alpha='auto',
                             eval_every=1)
    except ValueError:
        return

    plotter = Plotter(df, lda_model, manager)

    if company is not None:
        plotter.word_cloud(company=company, year=year)
        return

    top_topics = lda_model.top_topics(bow_corpus)
    for topic, score in top_topics:
        for found_topic in found_topics:
            sum_scores = sum([num for num, word in topic if word in found_topics[found_topic]])
            if sum_scores > topic_scores[found_topic]:
                topic_scores[found_topic] = sum_scores

    if save_model:
        # Save the model to resources folder
        manager.save_model(lda_model)

    topic_scores[COL_YEAR] = year
    return topic_scores


if __name__ == '__main__':
    manager = ManagerIO()
    data_frame = manager.read_processed_reviews()

    companies = set(data_frame[COL_COMPANY])
    years = sorted(list(set(data_frame[COL_YEAR])))

    # Process wordcloud for each company
    pd.DataFrame([process_reviews(data_frame, company=comp) for comp in companies])

    # Process lineplot for each year
    topics_df = pd.DataFrame([process_reviews(data_frame, year=data_year) for data_year in years])

    plot = Plotter(topics_df)
    plot.plot_topics_over_time()


