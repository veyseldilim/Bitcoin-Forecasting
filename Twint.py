import twint
import nest_asyncio
nest_asyncio.apply()
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import sqlite3



since = datetime(2021,6,30)
until = datetime(2021,7,7)

#print(until.strftime("%m/%d/%Y, %H:%M:%S"))
con = sqlite3.connect('deneme.db')

#tweet_weight = pd.DataFrame(columns = ['Date', 'Weight_Sum'])


while(True):
    
    yesterday = until - timedelta(days = 1)
    if(until == since):
        break
    
    config = twint.Config()
    config.Lang = "en"
    config.Search = "#Bitcoin"
    config.Limit = 5
    config.Min_likes = 10
    config.Since = yesterday.strftime("%Y-%m-%d")
    config.Until = until.strftime("%Y-%m-%d")
    config.Custom["tweet"] = ["created_at","language", "tweet","replies_count","retweets_count","likes_count"] 
    config.Pandas = True
    config.Hide_output = True
    twint.run.Search(config)
    
    Tweets_df = twint.storage.panda.Tweets_df
    Tweets_df = Tweets_df[['date','tweet','language','nreplies','nretweets','nlikes']]
    Tweets_df = Tweets_df[Tweets_df.language == 'en']
    Tweets_df = Tweets_df[Tweets_df.nlikes >= 10]
    Tweets_df['date'] = pd.to_datetime(Tweets_df['date']) 
    Tweets_df.to_sql('Tweets',con,if_exists='append')
    until = until - timedelta(days = 1)



    #
    #list_of_tweets['tweet'] = list_of_tweets['tweet'].apply(cleanTweet)
    #list_of_tweets['compound'] = list_of_tweets['tweet'].apply(polarizeTweets)
   # list_of_tweets['polarity'] = list_of_tweets['polarity'].apply(polarizeTweets)[1]
    #list_of_tweets["weight"] = (list_of_tweets["nreplies"] + 1) * (list_of_tweets["nretweets"] + 1) *  (list_of_tweets["nlikes"] + 1) * list_of_tweets["compound"]
   
    
    #tweet_weight = tweet_weight.append({'Date' : until, 'Weight_Sum' : list_of_tweets['weight'].sum()}, 
    #            ignore_index = True)
    
    #tweet_weight.to_sql('Weight_Sum',con,if_exists='append')
    #list_of_tweets.to_sql('Tweets',con,if_exists='append')

#    df = pd.DataFrame(until, columns = ['Date'])
#    dff = pd.DataFrame(daily_sumOfWeight, columns = ['daily_sumOfWeight'])
#    frames = [df, dff]
#    result = pd.concat(frames)
#    result.to_csv(r'C:\Users\veyse\Desktop\bitirme projesi kodlar\weights.csv')
    
   
#    data = {'Date' : until, 'DailyWeight' : daily_sumOfWeight}
#    df = pd.DataFrame(data, columns= ['Date','DailyWeight'])
#    df = pd.DataFrame([until,daily_sumOfWeight, columns=list('AB'))
#    df.to_csv(r'C:\Users\veyse\Desktop\bitirme projesi kodlar\dailyWeight.csv')       
    
    #list_of_tweets.to_csv(r'C:\Users\veyse\Desktop\bitirme projesi kodlar\compound_Tweets.csv')
    
    # Store the tweet dataframe in database  

con.commit()
con.close()

"""
import twint
import nest_asyncio
nest_asyncio.apply()
import pandas as pd

# Configure
c = twint.Config()
c.Search = "covid vaccin"
c.Lang = "fr"
c.Geo = "48.880048,2.385939,5km"
c.Limit = 300
c.Output = "./test.json"
c.Store_json = True
# Run
twint.run.Search(c)

"""