import radar.scrape_tweets.pipeline as scraper
import radar.text_cleaning.nodes as text_cleaning
import numpy as np
import pandas as pd

#TOPIC MODELLING FOR MODERATE & HIGH RISK
# latest all tweets_df_latest
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")



#Get Positive and Negative keywords for topics
def get_keywords(df,name):
  dic_neg = {}
  dic_pos = {}
  # dic_neu = {}
  
  for i in range(0,len(df)):
    try:
      ind_lst_neg = []
      # ind_lst_neu = []
      ind_lst_pos = []
      temp_lst = [text_cleaning.sid.polarity_scores(x)['compound'] for x in df[name].fillna('aeiou aeiou')[i].split(' ') if 'aeiou' not in x ]
      temp_arr = np.array(df[name].fillna('aeiou aeiou')[i].split(' '))
    
      for j,k in enumerate(temp_lst):
        if k<0:
          ind_lst_neg.append(j)
    
      for j,k in enumerate(temp_lst):
        if k>0:
          ind_lst_pos.append(j)

      # for j,k in enumerate(temp_lst):
      #   if k == 0:
      #     ind_lst_neu.append(j)
    
      neg_words = list(temp_arr[ind_lst_neg])
      pos_words = list(temp_arr[ind_lst_pos])
      # neu_words = list(temp_arr[ind_lst_neu])
    
      if df['entity'][i] not in dic_neg:
        dic_neg[df['entity'][i]] = neg_words
    
      if df['entity'][i] not in dic_pos:
        dic_pos[df['entity'][i]] = pos_words

      # if df['entity'][i] not in dic_neu:
      #   dic_neu[df['entity'][i]] = neu_words
   
    except:
      if df['entity'][i] not in dic_neg:
        dic_neg[df['entity'][i]] = np.nan
    
      if df['entity'][i] not in dic_pos:
        dic_pos[df['entity'][i]] = np.nan

      # if df['entity'][i] not in dic_neu:
      #   dic_neu[df['entity'][i]] = neu_words


  dic_neg = pd.DataFrame(pd.Series(dic_neg)).reset_index()
  dic_neg.columns = ['entity','neg_key_words_'+name]

  dic_pos = pd.DataFrame(pd.Series(dic_pos)).reset_index()
  dic_pos.columns = ['entity','pos_key_words_'+name]

  # dic_neu = pd.DataFrame(pd.Series(dic_neu)).reset_index()
  # dic_neu.columns = ['entity','neu_key_words_'+name]


  df = pd.merge(df,dic_neg,on='entity',how ='left')
  df = pd.merge(df,dic_pos,on='entity',how ='left')
  # df = pd.merge(df,dic_neu,on='entity',how ='left')
  
  return df


def show_topics(vectorizer, lda_model, n_words):
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords



def indices(lst, item):
  return [i for i, x in enumerate(lst) if x == item]


def topic_func(x,col):
  topic_words = pd.DataFrame(columns = ['topic','entity','stock'])
  tweets_df_topic = pd.DataFrame([x]).reset_index(drop=True)
  try:
    entity = tweets_df_topic['entity'][0]
    stock = tweets_df_topic['stock'][0]
    df_topic = tweets_df_topic
    vectorizer = CountVectorizer(analyzer='word',stop_words='english', lowercase=False,token_pattern='[a-zA-Z0-9]{3,}')
    df_vectorized = vectorizer.fit_transform(df_topic[col])
    lda_model = LatentDirichletAllocation(n_components=1,max_iter=10, learning_method='online', random_state=100,batch_size=28, evaluate_every = -1, n_jobs = -1)
    # Init Grid Search Class
    # model = GridSearchCV(lda_model, param_grid=search_params,cv=2)
    # Do the Grid Search
    lda_model.fit(df_vectorized)
    best_lda_model = lda_model
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)
    # Assign Column and Index
    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=10)        
    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords['topic'] = df_topic_keywords.apply(lambda row: ' '.join(row.values.astype(str)).split(" "), axis=1)
    df_topic_keywords['entity'] = entity
    df_topic_keywords['stock'] = stock
    df_topic_keywords = df_topic_keywords[['entity','stock','topic']]
    a = df_topic_keywords['topic'][0]
    b = df_topic_keywords['stock'][0].split(' ')
    lst_final = list(set(a)^set(b))
    df_topic_keywords['topic'][0] = lst_final
    topic_words = pd.concat([topic_words,df_topic_keywords])
  except:
    pass

  return topic_words



def create_dic(df,col):
  dic = {}
  for i in range(0, len(df)):
    entity = df['entity'][i]
    lst_ind = []
    df_temp = df.loc[i:i]
    a = df_temp['topic'][i]
    b = df_temp[col][i].split(" ")
    
    try:
      for j in a:
        lst_ind.extend(indices(b,j))
    except:
      continue
    
    
    words = [b[i] for i in lst_ind]

    if entity not in dic:
      dic[entity] = words
  return dic

  