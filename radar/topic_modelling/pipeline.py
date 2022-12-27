import radar.utils.config as config
import radar.utils.logger as logger
import radar.scrape_tweets.pipeline as scraper
import radar.text_cleaning.nodes as text_func
import radar.topic_modelling.nodes as topic_mod

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



class topic_model():
    all = {}

    def __init__(self,name,df):

        self.name = name
        self.df = df

        topic_model.all[self.name] = self.df 

   
   
   
   
    #Daily Report Dashboard
    @classmethod
    def topic_modeller_daily_report(cls,df):
        
        topic_words = Parallel(n_jobs=num_cores)(delayed(topic_mod.topic_func)(x,'comments') for x in df.to_dict('records'))
        topic_words = pd.concat(topic_words)
        logger.logger.info(f" name : twitter data topic modelled for daily report, data shape: {topic_words.shape} {topic_words} ")
        

        final_tweet_topics = pd.merge(df,topic_words.drop('stock',axis = 1),on = 'entity',how='left')
        final_tweet_topics = final_tweet_topics[['date', 'entity', 'count', 'neg', 'pos', 'compound', 'ratio', 'stock',
                                                'macd', 'signal', 'flag1', 'macd_lag1', 'ews_social_media', 'social_media_risk',
                                                'comments', 'topic']]
        final_tweet_topics = final_tweet_topics.drop_duplicates(subset=['entity'], keep=False).reset_index(drop = True)
        

        dic = topic_mod.create_dic(final_tweet_topics,'comments')
        temp_df = pd.DataFrame(dic.items())
        temp_df.columns = ['entity','keywords']
        final_df = pd.merge(final_tweet_topics,temp_df,on='entity',how='left')
        final_df['keywords'] = final_df['keywords'].fillna(0).apply(lambda x : list(set(x)) if x !=0 else np.nan)
        final_df['keywords'] = final_df['keywords'].fillna(0).apply(lambda xs: " ".join(str(x) for x in xs) if xs !=0 else np.nan)


        entity = pd.read_csv(os.path.join(scraper.scrapper.config_catalog['paths']['custom_sentiment'].replace('sentiment_custom.csv','entity.csv')))
        entity = entity[['Type','Industry','Priority','stock']]
        subseg = entity[entity['Priority'].notna()]

        df2 = final_df.copy()
        df3 = pd.merge(df2,entity[['Type','Industry','stock']].drop_duplicates(), on ='stock',how='left')


        text_func.sentiment_custom.columns = ['keywords','polarity','mapping','polarity_scores','actual_word','len']
        text_func.sentiment_custom = text_func.sentiment_custom[['keywords','polarity','mapping','actual_word']]

        df4 = pd.merge(df3,text_func.sentiment_custom,on='keywords',how='left')
        df4['entity'] = df4['Type'] + df4['date']
        df4['class'] = 'Tweets'


        df4['radar_score'] = df4['ratio']*100
        df_dum = pd.DataFrame(columns = df4.columns.to_list())
        ty_lst = list(df4['Type'].unique())
        for i in ty_lst :
            df = df4[df4['Type'] == i]
            df['rolling_average_radar_score'] = df['radar_score'].rolling(window=7).mean()
            df_dum = pd.concat([df_dum,df])


        df_num = df_dum.drop(['count','comments'],axis=1)
        df_num.rename(columns={'stock':'entity_name'}, inplace=True)
        df_num['title'] = np.nan
        df_num['rank'] = np.nan
        df_num['news'] = np.nan
        df_num['article'] = np.nan
        df_num['url'] = np.nan
        df_num.to_csv(os.path.join(scraper.scrapper.config_catalog['paths']['output_path'], 
                                    'tweets_df_filtered_topic_cat_dailyreport.csv'), index=False)
        return  topic_model('final_tweet_topics',final_df)
        

    def __repr__(self):
        return f"{self.name}, {self.df}"









class entity_dashboard(topic_model):

    def __init__(self,name,df):

        super().__init__(
            name,
            df
        )
  
  
  
  
  
    #Entity Dashboard
    @classmethod
    def generate_topic_cat(cls,df):


        topic_words = Parallel(n_jobs=num_cores)(delayed(topic_mod.topic_func)(x,'lag_12_ews') for x in df.to_dict('records'))
        topic_words = pd.concat(topic_words)
        logger.logger.info(f" name : twitter data topic modelled for entity dashboard, data shape: {topic_words.shape} {topic_words} ")


        final_tweet_topics = pd.merge(df,topic_words.drop('stock',axis = 1),on = 'entity',how='left')
        final_tweet_topics = final_tweet_topics[['date', 'entity', 'count', 'neg', 'pos', 'compound', 'ratio', 'stock',
                                                 'macd', 'signal', 'flag1', 'macd_lag1', 'ews_social_media', 'social_media_risk', 
                                                 'lag_12_ews', 'topic']]
        final_tweet_topics_temp = final_tweet_topics.dropna(subset=['entity','lag_12_ews','topic']).reset_index(drop=True)
        final_tweet_topics_temp = final_tweet_topics_temp.drop_duplicates(subset=['entity'], keep=False).reset_index(drop = True)
        

        dic = topic_mod.create_dic(final_tweet_topics_temp,'lag_12_ews')
        temp_df = pd.DataFrame(dic.items())
        temp_df.columns = ['entity','keywords']
        final_df = pd.merge(final_tweet_topics,temp_df,on='entity',how='left')
        final_df['keywords'] = final_df['keywords'].fillna(0).apply(lambda xs: " ".join(str(x) for x in xs) if xs !=0 else np.nan)



        list_cols = {'keywords'}
        other_cols = list(set(final_df.columns) - set(list_cols))
        exploded = [final_df[col].str.split().explode() for col in list_cols]
        df2 = pd.DataFrame(dict(zip(list_cols, exploded)))
        df2 = final_df[other_cols].merge(df2, how="right", left_index=True, right_index=True)
        df2 = df2.drop(['lag_12_ews','topic'],axis=1)


        entity = pd.read_csv(os.path.join(scraper.scrapper.config_catalog['paths']['custom_sentiment'].replace('sentiment_custom.csv','entity.csv')))
        entity = entity[['Type','Industry','Priority','stock']]
        subseg = entity[entity['Priority'].notna()]


        df3 = pd.merge(df2,entity[['Type','Industry','stock']].drop_duplicates(), on ='stock',how='left')


        

        df4 = pd.merge(df3,text_func.sentiment_custom,on='keywords',how='left')
        df4['entity'] = df4['Type'] + df4['date']
        df4['class'] = 'Tweets'

        df_topweets = df4.copy()
        df_topweets.rename(columns={'stock':'entity_name'}, inplace=True)
        df_topweets = df_topweets.drop('count',axis=1)
        df_topweets['title'] = np.nan
        df_topweets['rank'] = np.nan
        df_topweets['news'] = np.nan
        df_topweets = df_topweets[['date', 'entity', 'signal', 'macd', 'compound', 'flag1', 'neg',
            'ews_social_media', 'social_media_risk', 'ratio', 'pos', 'macd_lag1', 'entity_name', 'keywords', 'Type', 'Industry', 'polarity', 'mapping',
            'actual_word', 'class', 'title', 'rank', 'news']]
       
        df_topweets.to_csv(os.path.join(scraper.scrapper.config_catalog['paths']['output_path'], 
                                    'tweets_df_filtered_topic_cat.csv'), index=False)
        logger.logger.info(f" name : twitter data topic modelled for entity dashboard generated, data shape: {df_topweets.shape} {df_topweets} ")
                            
        
        return topic_model('tweets_df_filtered_topic_cat',df_topweets)







    @classmethod
    def generate_neg_cat(cls,df):
        tweets_df_topic = df.dropna().reset_index(drop=True)


        final_df = tweets_df_topic.copy().reset_index(drop =True)
        final_df = topic_mod.get_keywords(final_df,'lag_12_ews')
        final_df['neg_key_words_lag_12_ews'] = final_df['neg_key_words_lag_12_ews'].fillna(0).apply(lambda xs: " ".join(str(x) for x in xs) if xs !=0 else np.nan)
        final_df['pos_key_words_lag_12_ews'] = final_df['pos_key_words_lag_12_ews'].fillna(0).apply(lambda xs: " ".join(str(x) for x in xs) if xs !=0 else np.nan)
        




        list_cols = {'neg_key_words_lag_12_ews'}
        other_cols = list(set(final_df.columns) - set(list_cols))
        exploded = [final_df[col].str.split().explode() for col in list_cols]
        df2_neg = pd.DataFrame(dict(zip(list_cols, exploded)))
        df2_neg = final_df[other_cols].merge(df2_neg, how="right", left_index=True, right_index=True)

        df2_neg = df2_neg.drop(['lag_12_ews'],axis=1)




        entity = pd.read_csv(os.path.join(scraper.scrapper.config_catalog['paths']['custom_sentiment'].replace('sentiment_custom.csv','entity.csv')))
        entity = entity[['Type','Industry','Priority','stock']]
        subseg = entity[entity['Priority'].notna()]

        df3 = pd.merge(df2_neg,entity[['Type','Industry','stock']].drop_duplicates(), on ='stock',how='left')
        df3 = df3.rename(columns = {'neg_key_words_lag_12_ews':'keywords'})
        df3.rename(columns={'stock':'entity_name'}, inplace=True)





        df4 = pd.merge(df3,text_func.sentiment_custom,on='keywords',how='left')
        df4 = df4.drop(['comments','pos_key_words_lag_12_ews'],axis=1)
        df4['entity'] = df4['Type'] + df4['date']
        df4['class'] = 'Tweets'

        df_negtweets = df4.drop('count',axis=1)
        df_negtweets['title'] = np.nan
        df_negtweets['rank'] = np.nan
        df_negtweets['news'] = np.nan

        df_negtweets = df_negtweets[['date', 'entity', 'signal', 'macd', 'compound', 'flag1', 'neg',
        'ews_social_media', 'social_media_risk', 'ratio', 'pos', 'macd_lag1', 'entity_name', 'keywords', 'Type', 'Industry', 'polarity', 'mapping',
        'actual_word', 'class', 'title', 'rank', 'news']]


        df_negtweets.to_csv(os.path.join(scraper.scrapper.config_catalog['paths']['output_path'], 
                                    'tweets_df_filtered_neg_cat.csv'), index=False)
        logger.logger.info(f" name : twitter data neg words for entity dashboard generated, data shape: {df_negtweets.shape} {df_negtweets} ")
         
        return topic_model('tweets_df_filtered_neg_cat',df_negtweets)






  