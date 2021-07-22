import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from rail_topics.utils.manager_io import COL_DATE, ManagerIO, COL_YEAR
from copy import deepcopy


class Plotter:

    def __init__(self, df, model=None, manager=None):
        self.df = df
        self.model = model
        sns.set()
        if manager is None:
            self.manager = ManagerIO()
        else:
            self.manager = manager

    def copy_data(self):
        return deepcopy(self.df), deepcopy(self.model)

    def word_cloud(self, company='', year=None):
        df, model = self.copy_data()

        if year is None:
            year_str = ''
        else:
            year_str = str(year)

        for t in range(model.num_topics):
            save_string = f'{company}_{t}.png'

            plt.figure()
            plt.title(company)
            plt.imshow(WordCloud(background_color='white').fit_words(dict(model.show_topic(t))))
            plt.axis("off")
            plt.title(f"Company {company} topic #{t}")
            plt.savefig(self.manager.find_resource(save_string, 'plots'))

    def plot_sentiment_over_time(self):
        df = deepcopy(self.df)
        col_stars_7_day_avg = 'stars_7_day_avg'
        df = df[(df['date'] > '2015-01-01')]
        df[col_stars_7_day_avg] = df.stars.rolling(28 * 2).mean().shift(-12 * 2)
        sns.lineplot(x=COL_DATE, y=col_stars_7_day_avg, data=df, ci=None)
        plt.show()

    def plot_topics_over_time(self):
        df, _ = self.copy_data()
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        df.plot(x=COL_YEAR, style='.-', ax=ax)
        plt.xticks(df[COL_YEAR])
        plt.ylabel('Score')
        plt.savefig(self.manager.find_resource('topics_over_time.png', 'plots'))
