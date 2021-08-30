# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:19:53 2021

@author: veyse
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import sqlite3
from datetime import datetime, timedelta
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json
import re

#START_DATE='2021-01-01'
#END_DATE=str(datetime.now().strftime('%Y-%m-%d'))

GOLD='GC=F'
USD = 'DX-Y.NYB'
BTC = 'BTC-USD'
ETH = 'ETH-USD'
DOGE = 'DOGE-USD'



#Compound Formula
def cleanTweet(tweet):
  
    tweet = re.sub(r'(@|https?)\S+|#', '', tweet)
    return tweet

def polarizeTweets(tweet):
    return analyser.polarity_scores(tweet)['compound']


def calculateDailyWeight():
    
   #tweets = pd.read_sql_query("SELECT date, tweet, nreplies, nlikes, nretweets FROM tweets  WHERE date BETWEEN date('2016-01-04') AND date('2021-06-12')", con)

    tweets = pd.read_sql_query("SELECT date, tweet, nreplies, nlikes, nretweets FROM tweets", con)
    tweets['date'] = pd.to_datetime(tweets['date'])
    tweets = tweets.sort_values(by = 'date',ascending = False)

    tweets['tweet'] = tweets['tweet'].apply(cleanTweet)
    tweets['compound'] = tweets['tweet'].apply(polarizeTweets)
    tweets["weight"] = ((tweets["nreplies"]) * (tweets["nretweets"]) *  (tweets["nlikes"])) * tweets["compound"]
    tweet_sums = pd.DataFrame(
    {'date': tweets['date'].dt.date.unique().tolist(),
     'sums': tweets.groupby(tweets['date'].dt.date)['weight'].sum()
    })
    tweet_sums.reset_index(drop=True, inplace=True) 
    tweet_sums['date'] = pd.to_datetime(tweet_sums['date'])
    return tweet_sums

def getBtcData(ticker, START_DATE, END_DATE):
    coinData = yf.download(ticker, start=(START_DATE + timedelta(days = 1)), end=(END_DATE +timedelta(days = 1)))
    coinData['Date'] = pd.to_datetime(coinData.index) #always format the date
    coinData.reset_index(drop=True, inplace=True)
    coinData = coinData[['Date','Open','High','Low','Volume','Adj Close','Close']]
    coinData =   coinData.sort_values(by = 'Date',ascending = False)
    return coinData

def getFinanceData(ticker, START_DATE, END_DATE):
    financeData = yf.download(ticker, start=(START_DATE + timedelta(days = 1)), end=(END_DATE +timedelta(days = 1)))
    financeData['Date'] = pd.to_datetime(financeData.index) #always format the date
    financeData.reset_index(drop=True, inplace=True)
    financeData = financeData[['Date','Close']]
    financeData =   financeData.sort_values(by = 'Date',ascending = False)
    return financeData

def getTweetCountsDaily(START_DATE, END_DATE):
    
    url =  'https://bitinfocharts.com/comparison/tweets-btc-eth-ltc.html'
    
    parsed_html = requests.get(url)
    soup = BeautifulSoup(parsed_html.content, 'html.parser')
    
    
    texts = str(soup).split('[')
    data = []
    
    for text in texts:
        if (text.startswith('new Date("')):
            data.append(text)
            
    #Remove last element from the list
    data = data[:-1]
    
    date = []
    btc = []
    eth = []
    ltc = []
    
    
    for x in data:
        x = x.replace(']','')
        elements = x.split(',')
        date.append((re.findall(r'"(.*?)"', elements[0]))[0]) 
        btc.append(elements[1]) 
        eth.append(elements[2])
        ltc.append(elements[3]) 
    
    tweet_count = pd.DataFrame(
        {'Date': date,
         'btcTweetCount': btc,
         'ethTweetCount': eth,
         'ltcTweetCount': ltc
        })
    
    #tweet_count['date'] = tweet_count['date'].str.replace('[','')
    tweet_count['Date'] = pd.to_datetime(tweet_count['Date'])
    
    tweet_count['btcTweetCount'] = pd.to_numeric(tweet_count['btcTweetCount'], errors='coerce')
    tweet_count['ethTweetCount'] = pd.to_numeric(tweet_count['ethTweetCount'], errors='coerce')
    tweet_count['ltcTweetCount'] = pd.to_numeric(tweet_count['ltcTweetCount'], errors='coerce')
    
    return tweet_count.loc[(tweet_count['Date'] > (START_DATE - timedelta(days = 1))) & (tweet_count['Date'] <= END_DATE)][['Date','btcTweetCount']]




analyser = SentimentIntensityAnalyzer()
con = sqlite3.connect('deneme.db')
conAllDF = sqlite3.connect('deneme2.db')

tweetSums = calculateDailyWeight()

tweetSums = tweetSums.rename(columns={'date': 'Date'})


START_DATE = min(tweetSums['Date'])
END_DATE = max(tweetSums['Date'])


btc = getBtcData(BTC,START_DATE,END_DATE)
btc = btc[['Date','Open','High','Low','Volume','Adj Close','Close']]
btc['Date'] = pd.to_datetime(btc['Date']) #always format the date


eth = getFinanceData(ETH, START_DATE, END_DATE)
eth = eth[['Date','Close']]
eth.columns = ['Date','ethClose']
eth['Date'] = pd.to_datetime(eth['Date']) #always format the date



gold = getFinanceData(GOLD, START_DATE, END_DATE)
gold = gold[['Date','Close']]
gold.columns = ['Date','goldPrice']


usd = getFinanceData(USD, START_DATE, END_DATE)
usd = usd[['Date','Close']]
usd.columns = ['Date','usdIndexPrice']


tweetCounts = getTweetCountsDaily(START_DATE, END_DATE)

k = pd.date_range(start=min(tweetSums['Date']), end = max(tweetSums['Date']))
k = k.to_frame()
k.columns = ['Date']
#kk = pd.date_range(start = min(btc['Date']),end = max(btc['Date']))

#df_merge = pd.merge(tweetSums,btc, how='inner', on = 'Date')

#btc = pd.read_csv('BTC-USD.csv')
#btc = btc[['Date','Open','High','Low','Volume','Adj Close','Close']]
#btc['Date'] = pd.to_datetime(btc['Date']) #always format the date


#eth = pd.read_csv('ETH-USD.csv')
#eth = eth[['Date','Close']]
#eth.columns = ['Date','ethClose']
#eth['Date'] = pd.to_datetime(eth['Date']) #always format the date


#usdCSV2 = pd.read_csv('US Dollar Index Futures Historical Data.csv',parse_dates=['Date'])
#usdCSV2 = usdCSV2[['Date','Price']]
#usdCSV2.columns = ['Date','usdIndexPrice']


#goldCSV2 = pd.read_csv('Gold Futures Historical Data.csv',parse_dates=['Date'])
#goldCSV2 = goldCSV2[['Date','Price']]
#goldCSV2.columns = ['Date','goldPrice']



goldData = pd.merge(k, gold, how='left', on = 'Date')
#goldData['goldPrice'] = goldData['goldPrice'].str.replace(',','')
goldData['goldPrice'] = goldData['goldPrice'].astype(float)
goldData = goldData.fillna(method='ffill')


usdData = pd.merge(k, usd, how='left', on = 'Date')

usdData = usdData.fillna(method='ffill')


df_AllData = pd.merge(usdData, goldData,how = 'inner', on = 'Date')
df_AllData = pd.merge(df_AllData, eth, how = 'inner', on = 'Date')
df_AllData = pd.merge(df_AllData, tweetCounts, how = 'inner', on = 'Date')
df_AllData = pd.merge(df_AllData, tweetSums, how = 'inner', on = 'Date')
df_AllData = pd.merge(df_AllData, btc, how = 'inner', on = 'Date')


df_AllData.isnull().sum(axis = 0)
df_AllData.isnull().sum(axis = 1)

df_DroppedAllData = df_AllData.dropna().copy()

df_AllData = df_AllData.fillna(method='ffill')

df_AllData.to_sql('FormulaallDataTable',con,if_exists='append')
#df_DroppedAllData.to_sql('droppedAllDataTable',conAllDF,if_exists='append')

con.commit()
con.close()

conAllDF.commit()
conAllDF.close()

#del df_AllData




#df_AllData = pd.merge(tweetSums,btc, how='inner', on = 'Date')

#Duplicated
#duplicateRowsDF = btc[btc.duplicated()]
#print("Duplicate Rows except first occurrence based on all columns are :")
#print(duplicateRowsDF)

#duplicateRowsDF = btc[btc.duplicated(['Date'])]
#print("Duplicate Rows based on a single column are:", duplicateRowsDF, sep='\n')

#btc = btc.drop_duplicates('Date', keep='last')








#Take all days from a date frame
#dates = tweets['date'].dt.date.unique().tolist()

#Check if some days are missing
#


#sums = tweets.groupby(tweets['date'].dt.date)['weight'].sum()




#d = s.groupby(lambda x: x.date()).aggregate(lambda x: sum(x) if len(x) >= 40 else np.nan)

#tweet_weight = tweet_weight.append({'Date' : until, 'Weight_Sum' : list_of_tweets['weight'].sum()},ignore_index = True)
                
    
#    tweet_weight.to_sql('Weight_Sum',con,if_exists='append')
#    list_of_tweets.to_sql('Tweets',con,if_exists='append')

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