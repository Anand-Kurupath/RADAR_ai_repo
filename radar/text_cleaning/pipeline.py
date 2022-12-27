import radar.utils.config as config
import radar.utils.logger as logger
import radar.scrape_tweets.pipeline as scraper
import radar.text_cleaning.nodes as text_func
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



class text_cleaner():
    all = {}

    #LOAD FILES from interim path
    entries = [os.path.join(scraper.scrapper.config_catalog['paths']['temp_path'], entry) for entry in os.listdir(scraper.scrapper.config_catalog['paths']['temp_path']) if os.path.isfile(os.path.join(scraper.scrapper.config_catalog['paths']['temp_path'], entry))]

    def __init__(self,name,df):

        self.name = name
        self.df = df

        text_cleaner.all[self.name] = self.df 


    @classmethod
    def clean_text(cls):
        tweets_df = Parallel(n_jobs=num_cores)(delayed(text_func.clean_text_macd)(x) for x in text_cleaner.entries if '.csv' in x)
        tweets_df = pd.concat(tweets_df)
        logger.logger.info(f" name : twitter data cleaned, data shape: {tweets_df.shape} {tweets_df} ")
        tweets_df.to_csv(os.path.join(scraper.scrapper.output_path,'clean_tweets.csv'),index=False)
        return text_cleaner('clean_df',tweets_df)



    def __repr__(self):
        return f"{self.name}, {self.df}"
    