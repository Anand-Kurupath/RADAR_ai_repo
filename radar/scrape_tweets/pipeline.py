import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import snscrape
import snscrape.modules.twitter as sntwitter
import radar.utils.config as config
import radar.utils.logger as logger
import radar.scrape_tweets.nodes as scrape
import os



class scrapper():

    all = {}

    # Load congif
    config_catalog = config.load_catalog_params(os.getcwd(), 'conf/catalogs')
    config_parameters = config.load_catalog_params(os.getcwd(), 'conf/parameters')
    temp_path = os.path.join(os.getcwd(),config_catalog['paths']['temp_path'])
    output_path = os.path.join(os.getcwd(),config_catalog['paths']['output_path'])

    start_date = config_parameters['ScrapeTweets']['dates']['start_date']
    end_date = config_parameters['ScrapeTweets']['dates']['end_date']
    search_term = config_parameters['ScrapeTweets']['search_term']
    logger.logger.info(f"loaded config and parameters.")


    date_series = pd.Series(pd.date_range(start=start_date,end=end_date))

    def __init__(self,name,df):

        self.name = name
        self.df = df

        scrapper.all[self.name] = self.df 

    
    
    
    @classmethod
    def instantiate_from_twitter(cls):
        
        # Load input files
        tweets_df = Parallel(n_jobs=num_cores)(delayed(scrape.twitter_scrape)(x,scrapper.date_series,scrapper.temp_path) for x in scrapper.search_term)
        tweets_df = pd.concat(tweets_df)
        logger.logger.info(f" name : twitter scraped, data shape: {tweets_df.shape} {tweets_df} ")
        tweets_df.to_csv(os.path.join(scrapper.output_path,'tweets.csv'),index=False)
        return scrapper('tweets_df',tweets_df)

    
    def __repr__(self):
        return f"{self.name}, {self.df}"
    
