# Bitcoin-Forecasting

# Table of Contents

* [Introduction](#Introduction)
* [Data Collection](#Data_Collection)
* [Data Preprocessing](#Data_Preprocessing) 
   * [Missing Values](#Missing_Values)
   * [Duplicated Rows](#Duplicated_Rows)
* [Models](#Models) 
   * [Naive Bayes](#Naive_Bayes)
   * [Random Forest](#Random_Forest)
   * [Decision Tree](#Decision_Tree)



# Introduction <a class="anchor" id="Introduction"></a>

A time series project to predict bitcoin prices based on historical data and tweet sentiments.
Tweets have been collected by using Twint library. To do sentiment analysis, VADER has been used.
Financial data has been taken from yahoo finance API.
After data sets has been combined, CNN model has been applied.

# Data Collecting <a class="anchor" id="Data_Collection"></a>

The data is very important for training and testing steps of the machine and deep learning algorithms. So, for a financial project, the historical data is essential. Because we know the importance of historical data, we tried to find a website for financial data. 
Twitter data is collected by using a library named Twint. Twint is an open-source program that allows you to scrape twitter data without using a Twitter API (Twint Project, 2018).   Unlike Twitter API, Twint has no restrictions on tweet scraping. Historical tweets can be collected easily. As it provides many filters, (language selection, date selection, number of likes selection etc.) tweet scraping becomes easier and faster.
After installing it, Twint library and pandas are imported. Tweet objects are stored as data frame in RAM for data preprocessing operations (text cleaning, sentiment analysis etc.).


![image](https://user-images.githubusercontent.com/50465232/185765589-a6ceb438-0761-47c3-835e-0868b5d83821.png)

Date interval is specified as global variables.  Tweets are collected and processed day by day. Unnecessary columns of twitter data frame are deleted to save resource and time. Tweets are stored in local database and upper bound date lowered by one day and it goes on doing same operations until upper bound date and lower bound date become equal.

To access historical data of Bitcoin (BTC) and Gold Price is very important for making good guess about their prices in the future. That is why, we collected the required data from famous finance site of Yahoo!. (https://finance.yahoo.com).

![image](https://user-images.githubusercontent.com/50465232/185765628-b87c7190-6e5d-4f87-b71d-62db491663c1.png)

In this code, primarily, we import pandas_datareader for getting data and datetime packages for define the date. We define a start date is ‘2005-01-01’ and end date of today. After that, we defined a variable for name of required data in yahoo finance site. For Bitcoin price in USD, the ‘BTC-USD’ ticker is used.
We used DataReader() function of pandas_datareader, this function takes parameters as ticker, data website name, start and end date. After you give that arguments, you can take all data in site easily.

![image](https://user-images.githubusercontent.com/50465232/185765658-b26ae652-3166-41e7-bd43-be31b45964b4.png)


# Data Preprocessing <a class="anchor" id="Data_Preprocessing"></a>



