# Bitcoin-Forecasting

# Table of Contents

* [Introduction](#Introduction)
* [Data Collection](#Data_Collection)
* [Data Preprocessing](#Data_Preprocessing) 
* [Data Storing](#Data_Storing)
* [Applying Deep Learning Model CNN](#CNN)
* [Results](#Results) 
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

Date preprocessing is needed to get good quality results from deep learning algorithms.   Some raw Twitter texts includes mentions, URLs, hashtags (#), and non-English alphabet characters etc. These must be cleaned to get more optimum polarity results from polarizeTweets function.

![image](https://user-images.githubusercontent.com/50465232/185765708-8da65080-6bff-4fbc-98be-00e63ebf0340.png)

The code fragment above deletes mentions, URLs, and hashtags. For example, the sentence below:
 “Interview: The philosophy of #bitcoin with Professor @CraigWarmke  https://t.co/iv19xzpN4d #BTC $BTC #crypto #cryptocurrency @YouTube #podcast”
After the function execution, it becomes:
 “Interview: The philosophy of bitcoin with Professor    BTC $BTC crypto cryptocurrency podcast”

To do sentiment analysis on twitter text data, VADER, an open-source sentiment analysis library for python, is used. VADER is a widely used library to classify social media text for it has ability to process slang and emojis too. After data preprocessing operations, data is stored in database.


# Data Storing <a class="anchor" id="Data_Storing"></a>

Collected data is stored by using a local database system. SQLite has been chosen as database management system. SQLite is an open-source DBMS project, and it is embedded into python language. One can use it by simply importing its library.

Unlike other DBMSs, SQLite does not require an internet connection to connect local or cloud server. It runs on local disk of computer. It is highly simple comparing to other DBMSs.
After importing the sqlite3 library, name of database has been declared. Then, to sql function of pandas has been used. This function writes data frame into database without even creating a table first. The database can be browsed by using DB Browser software for SQLite.

![image](https://user-images.githubusercontent.com/50465232/185765780-3d31d6a1-0225-4a3f-b0fd-858857c92478.png)


# Applying Deep Learning Model CNN <a class="anchor" id="CNN"></a>

Neural Networks are some algorithms and models based on the human brain. The human brain consists of billions of nerve cells called neurons. Also, these artificial neural networks consist of processors called as neurons like biological neurons.
Deep Learning is some powerful methods for learning in neural networks and generally, deep learning models are known as deep neural networks. ‘Deep’ term in the name of Deep Learning refers to the number of hidden layers in neural networks. Although normal neural networks can only include 2-3 hidden layers, deep neural networks can consist of 150 hidden layers.
Deep Learning methods can be chosen to process time series data. With enough resources and historical data to train model, neural networks can surpass traditional machine learning techniques in data science.
After data preprocessing operations, deep learning models CNN has been applied to dataset.

CNNs are a type of neural network that was designed to handle image data. They process raw data instead of handcrafted or featured data. Model automatically learns from the raw data. This feature of CNNs make it useful when dealing with time series data. A sequence of features can be treated as raw pixel data.
Time series data must be transformed to fit CNNs model. Specifically, two-dimensional data must be transformed three-dimensional data. Before the transforming operations, some other operations must be applied to analyze data such as autocorrelation and partial autocorrelation.
Autocorrelation and partial autocorrelation plots are used in time series analysis/forecasting. The main difference between partial autocorrelation and autocorrelation plots, partial autocorrelation excludes indirect correlations in the calculation. 

![image](https://user-images.githubusercontent.com/50465232/185765857-3e45f5c2-f4df-4ae6-81ba-945acecc38cb.png)

By looking at these plots, we can observe the relationships between instances/samples in time series data. 

![image](https://user-images.githubusercontent.com/50465232/185765867-7f9b4b14-0939-486f-8b31-12997a7fad9c.png)

The correlation scale is from -1 to +1, which represents a scale from zero correlation to full correlation. There are 20 lags in total and all of them have a correlation value bigger than 0.5. So, we can say statistically bitcoin future close price is dependent on past 20 days of close price.

![image](https://user-images.githubusercontent.com/50465232/185765878-c3e8b70d-746f-4eb9-95c1-7ab77350e0f5.png)

When indirect correlations are removed, we can see that bitcoin future close price is highly dependent on past 2 days close price.
For this reason, the dataset is shifted by 2. To lag the dataset, pandas shift function has been used for each column except for ‘date’ column.

![image](https://user-images.githubusercontent.com/50465232/185765888-5e6d38f0-943a-4016-b8ef-dfeedd55acc2.png)

Because of shift operation, the first two rows of the dataset now contain NA values. Hence, they are simply removed. Now ‘date’ and other non-lag columns are removed. Dataset now only contains lagged versions of variables and target feature ‘Close’.

Dataset is ready to split as train, validation, and test samples. After this, data is scaled to between 0 and +2. Normalization is always recommended as it provides equality among numeric variables.

![image](https://user-images.githubusercontent.com/50465232/185765906-4315f8c6-62fb-47f7-ac65-ddf0517b67f3.png)

Dataset must be break into independent variables and dependent variables to fit CNN model. Dependent variable is ‘Close’ price and independent variables are others.
Now the data must be split into samples. As 2 input time steps has been chosen, every sample contains past 2 values and dataset have a 3D shape. It is ready to fit.

![image](https://user-images.githubusercontent.com/50465232/185765918-571189ae-f635-4fcf-90c4-c27377f3fdfa.png)

CNN model is created like the figure below. Necessary inputs are given to the model.

![image](https://user-images.githubusercontent.com/50465232/185765923-e43672e5-9f00-4082-9c72-4b95900cce9a.png)

After fitting the model, it is ready to predict the outcome based on testing independent variables.


# Results <a class="anchor" id="Results"></a>

After model is trained with historical data, it is ready to predict the outcome.

![image](https://user-images.githubusercontent.com/50465232/185765990-652965f1-1566-453b-a69d-1cec34196ab8.png)

The independent variables are given to model predict function. By using these variables, the CNN model makes a prediction and returns an array.
Testing result plot for CNN model is given below.

![image](https://user-images.githubusercontent.com/50465232/185766002-460a1b5b-6f24-45bc-8218-2f9cde4aefdd.png)

To calculate error, mean squared error is used as it is a way to calculate error in regression problems. Mean squared error, as the name implies, the average squared difference between the predicted values and the actual values.
To measure the model’s accuracy, directions of bitcoin price have examined. For both of prediction and testing data, directions have calculated as 0 for down and 1 for up.
After compared the results, it is found that CNN model has a 55% accuracy. So, if model predicts price go up or go down it is true in 55%.
To investigate which descriptive features affect bitcoin price, each descriptive feature is removed from dataset and results are compared.

![image](https://user-images.githubusercontent.com/50465232/185766031-6d8b1899-b682-4e01-b9f4-f07797c3532b.png)

From this table it can be concluded that bitcoin close price is highly dependent on bitcoin data, gold, and USD index as they raise MSE when they are removed.

![image](https://user-images.githubusercontent.com/50465232/185766051-97cb2df4-6185-4b90-ae10-87a2e41fcf21.png)




