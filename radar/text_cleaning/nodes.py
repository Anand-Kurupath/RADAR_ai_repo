#Generic Packages
import pandas as pd
import numpy as np
import heapq
from datetime import date
import os
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint


#Setting up text cleaning stack
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
words = set(nltk.corpus.words.words())
from nltk import word_tokenize
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer ,PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import words
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','https','in','to','xa',"co",'thank','bank','sir','of','india','yes','first','small','state'])
words = words.words() 
import re
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
import en_core_web_sm
nlp = nlp = spacy.load("en_core_web_sm")





#Relative import modules
import radar.utils.config as config
import radar.utils.logger as logger
import radar.scrape_tweets.pipeline as scraper









#Creating Custome Sentiments
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_custom = pd.read_csv(scraper.scrapper.config_catalog['paths']['custom_sentiment'])
df_sc = sentiment_custom.copy()
lst_len = list(df_sc['Keyword '].str.split().str.len())
sentiment_custom['polarity_scores'] = np.where(sentiment_custom.Polarity == 'Negative',-3,np.where(sentiment_custom.Polarity == 'Neutral',0,3))
sentiment_custom.columns = ['keyword','polarity',"mapping",'polarity_scores']
sentiment_custom["actual_word"] = sentiment_custom.keyword
sentiment_custom.keyword = sentiment_custom.keyword.str.replace(' ', '').str.lower()
new_words = sentiment_custom.drop('polarity',axis =1).set_index('keyword').T.reset_index(drop= True).to_dict('records')[1]
sentiment_custom['len'] = lst_len

sid = SentimentIntensityAnalyzer()
sid.lexicon.update(new_words)









#Preprocess function blocks for text cleaning
def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower() 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=cleantext
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_url)
    filtered_words = [lemmatizer.lemmatize(word) for word in tokens if not word in stop_words]
    return " ".join(filtered_words) 

def sent_to_words(sentences):
    for sentence in sentences:
        # yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        yield([preprocess(sentence)])  


def lemmatization(texts):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc ]))
    return texts_out












#Greedy search for custom words and sentiment scores
from nltk import everygrams
def substringSieve(string_list):
    out = []
    for s in string_list:
        if not any([s in r for r in string_list if s != r]):
            out.append(s)
    return out

allowed_bigrams = sentiment_custom[sentiment_custom['len'] > 1].drop(['polarity','polarity_scores'],axis =1).set_index('keyword').T.reset_index(drop= True).to_dict('records')[2]

def process_text(text):
    tokens = text.split() # list of tokens
    everygram = list(everygrams(tokens, max_len = max(allowed_bigrams.values())))
    everygram = list(map(''.join, everygram))
    
    #begin recreating the text
    final = ''
    for i in everygram:
        if len([x for x in allowed_bigrams if i == x])  == 0:
          join_word = [x for x in tokens if x == i]
          if len(join_word) > 0:
            join_word = join_word[0]
          else:
            continue
        else:
          join_word = [x for x in allowed_bigrams if i == x][0]
          
        final += join_word + " "
    words = final.split()
    words = substringSieve(words)
    final = " ".join(sorted(set(words), key=words.index))
    return final









#Text Cleaning and macd calculations
def clean_text_macd(x):
    tweets_df = pd.DataFrame(columns = ['date', 'entity', 'count', 'neg', 'pos', 'compound', 'ratio', 'stock', 'macd', 'signal', 'flag1', 'macd_lag1', 'ews_social_media',
                                    'social_media_risk', 'lag_12_ews'])
    try:
            empty_df = pd.read_csv(x,index_col=None, header=0)
    except:
            empty_df = pd.read_csv(x,index_col=None, header=0, encoding='unicode_escape')
    empty_df['comments'] = empty_df.renderedContent.apply(lambda x : list(sent_to_words([x]))[0])
    empty_df['comments'] = empty_df.comments.apply(lambda comment : [" ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ) for x in comment][0]) # need to find a faster method
    df_lemma = empty_df[['comments','stock','date']]
    df_lemma['entity'] = df_lemma['stock'] + df_lemma['date']
    df_lemma['comments'].replace('', np.nan, inplace=True)
    df_lemma = df_lemma.dropna().reset_index(drop = True)
    df_lemma['comments'] = df_lemma['comments'].apply( lambda x : process_text(x))
    df_vader = df_lemma.copy()

    df_vader['scores'] = df_vader['comments'].apply(lambda review: sid.polarity_scores(review))
    df_vader['compound']  = df_vader['scores'].apply(lambda score_dict: score_dict['compound'])
    df_vader['neg']  = df_vader['scores'].apply(lambda score_dict: score_dict['neg'])
    df_vader['pos'] = df_vader['scores'].apply(lambda score_dict: score_dict['pos'])
    df_vader['count'] = 1
    df_vader_agg = df_vader.groupby('entity').agg({'count':'sum', 'neg':'mean','pos':'mean','compound':"mean"}).reset_index()
    df_vader_agg['ratio'] = (df_vader_agg['count']*df_vader_agg['neg']) / (df_vader_agg['count']*df_vader_agg['pos'])
    df_vader_com_agg = df_vader.groupby((['stock','date','entity']))['comments'].apply(lambda x: ' '.join(x)).reset_index()
    df_vader_final = pd.merge(df_vader_agg,df_vader_com_agg,on='entity',how='left')
    exp12 = df_vader_final.ratio.ewm(span=12, adjust=False).mean()
    exp26 = df_vader_final.ratio.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    df_vader_final['macd'] = macd
    df_vader_final['signal'] = df_vader_final.macd.ewm(span=9, adjust=False).mean()
    df_vader_final = df_vader_final.replace([np.inf, -np.inf], np.nan)
    df_vader_final = df_vader_final.fillna(0)
    df_vader_final = df_vader_final.set_index(['date'])

    df_vader_final['flag1'] = np.where(df_vader_final.ratio>=.5,1,0)
    df_vader_final['macd_lag1'] = df_vader_final['macd'].shift(1)
    df_vader_final['ews_social_media'] = np.where(df_vader_final.flag1==0,0,np.where(df_vader_final.macd_lag1>=df_vader_final.signal,0,np.where(df_vader_final.macd>df_vader_final.signal,1,0)))
    df_vader_final['social_media_risk'] = np.where(df_vader_final.ews_social_media==0,'Low Risk',np.where(df_vader_final.ratio>.75,'High Risk',"Moderate Risk"))
    df_vader_final = df_vader_final.reset_index()

    lst_ind = df_vader_final[df_vader_final['ews_social_media'] == 1].index
    lst_words = []

    for i in lst_ind:
        df_te = df_vader_final.loc[i-12:i]
        df_te = df_te.groupby((['stock']))['comments'].apply(lambda x: ' '.join(x)).reset_index()
        k = df_te['comments'][0]
        lst_words.append(k)

    df_vader_final_ews_ind = df_vader_final[df_vader_final['ews_social_media'] == 1]
    df_vader_final_ews_ind['lag_12_ews'] = lst_words
    df_vader_final = pd.merge(df_vader_final,df_vader_final_ews_ind[['entity','lag_12_ews']],on='entity',how='left')

    # final_df = get_keywords(df_vader_final,'comments')
    # final_df['neg_key_words_comments'] = final_df['neg_key_words_comments'].apply(lambda xs:" ".join(str(x) for x in xs))
    # final_df['pos_key_words_comments'] = final_df['pos_key_words_comments'].apply(lambda xs:" ".join(str(x) for x in xs))
    # final_df['neu_key_words_comments'] = final_df['neu_key_words_comments'].apply(lambda xs:" ".join(str(x) for x in xs))


    # final_df = get_keywords(final_df,'lag_12_ews')
    # final_df['neg_key_words_lag_12_ews'] = final_df['neg_key_words_lag_12_ews'].fillna(0).apply(lambda xs: " ".join(str(x) for x in xs) if xs !=0 else np.nan)
    # final_df['pos_key_words_lag_12_ews'] = final_df['pos_key_words_lag_12_ews'].fillna(0).apply(lambda xs: " ".join(str(x) for x in xs) if xs !=0 else np.nan)
    # final_df['neu_key_words_lag_12_ews'] = final_df['neu_key_words_lag_12_ews'].fillna(0).apply(lambda xs: " ".join(str(x) for x in xs) if xs !=0 else np.nan)
    # final_df = final_df.drop('comments',axis=1)

    stock = df_vader_final['stock'][0]

    tweets_df = pd.concat([tweets_df,df_vader_final])
    tweets_df.to_csv(os.path.join(scraper.scrapper.output_path,stock+ '_cleaned.csv'),index=False)
    return tweets_df
 