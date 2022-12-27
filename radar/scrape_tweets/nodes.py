import pandas as pd
import snscrape
import snscrape.modules.twitter as sntwitter
import radar.utils.logger as logger
import os

def twitter_scrape(stock,date_series,path):
  tweets_df = pd.DataFrame(columns = ['renderedContent','replyCount','retweetCount','likeCount','quoteCount','lang','retweetedTweet','quotedTweet','hashtags','coordinates',
                                      'stock', 'date'])
  for i in range(0,len(date_series)-1):
    tweets_list = []
    start_date = date_series[i].strftime('%Y-%m-%d')
    end_date = date_series[i+1].strftime('%Y-%m-%d')
    try:
      for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f"{stock} since:{start_date} until:{end_date}").get_items()):
        renderedContent = tweet.renderedContent
        replyCount = tweet.replyCount
        retweetCount = tweet.retweetCount
        likeCount = tweet.likeCount
        quoteCount = tweet.quoteCount
        lang = tweet.lang
        retweetedTweet = tweet.retweetedTweet
        quotedTweet = tweet.quotedTweet
        hashtags = tweet.hashtags
        coordinates = tweet.coordinates

        tweets_list.append({
                "renderedContent":renderedContent,
                "replyCount":replyCount,
                "retweetCount":retweetCount,
                "likeCount":likeCount,
                "quoteCount":quoteCount,
                "lang":lang,
                "retweetedTweet":retweetedTweet,
                "quotedTweet":quotedTweet,
                "hashtags":hashtags,
                "coordinates":coordinates,
            })
      
      temp_df = pd.DataFrame(tweets_list, columns=['renderedContent','replyCount','retweetCount','likeCount','quoteCount',
                                                   'lang','retweetedTweet','quotedTweet','hashtags','coordinates'])
      temp_df['date'] = start_date
      temp_df['stock'] = stock
      tweets_df = pd.concat([tweets_df,temp_df])
      
    except:
      continue
  tweets_df.to_csv(os.path.join(path,stock + '.csv'),index = False)  
  return tweets_df
