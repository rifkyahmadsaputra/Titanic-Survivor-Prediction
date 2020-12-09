# Titanic Survivor Prediction (Top 7%)

## Introduction
<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; in this time I will share my work which is a prediction titanic survivor from kaggle. This is beginner competition from kaggle, that is, using machine learning to create a model that predicts which passengers survived the Titanic shipwreck (Source : https://www.kaggle.com/c/titanic).
</p>

## Results
<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; I had done some experiment with several machine learning algorithms like Naive Bayes, Logistic Regression, XGBoost, K-Nearest Neighbors, etc. I used GridSearchCV to find best parameter and accuracy from each algorithms. The best hyperparameters I got is leaf_size = 1, metric = ‘minkowski’, n_neighbors = 12, p = 1, weights = ‘distance’.
</p>

<p align="center"> 
 <img src="images/gridsearchcv results.png" /> 
 <br></br>
 Gridsearchcv results
</p>

<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After that, I implemented on the data test , and then submitted the results prediction to kaggle .
</p>

<p align="center"> 
 <img src="images/prediction results.png" />
 <img src="images/score results.png" /> 
 <br></br>
 Results
</p>

<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Based on the results obtained, I got accuracy 0.79425 (top 7 %) with KNN algorithm.
</p>

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Summary
The following is a summary of what was done in this project:

#### - Exploratory Data Analysis
<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; First, I do exploratory data analysis for analysis purpose. In this section, I am going to look information from the data (type of columns, null value, etc.). After that I split the data into numerical and categorical data, then visualize that for easy to understand.
</p>

<p align="center"> 
 Pieces of data and informations on each column data train
 <br></br>
 <img src="images/pieces of data train.png" /> 
 <img src="images/informations of data train.png" />
 <br></br>
 There are first five and information on each column data test:
 <br></br>
 <img src="images/pieces of data test.png" /> 
 <img src="images/informations of data test.png" />
 <br></br>
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After that, I Split data into numeric and categories data. But first, we categorize the column ‘cabin’ with their first letter values. Then, the following are the distribution plots and heatmap of numerical data
</p>

<p align="center"> 
 <img src="images/distributions plot numeric data.png" /> 
 <img src="images/heatmap numeric data.png" />
 <br></br>
 Distributions plot and heatmap numeric data
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Then, I create a barplot of each categorical data. Here are the visualizations.
</p>

<p align="center"> 
 <img src="images/barplot categorical data.png" /> 
 <br></br>
 Barplot categorical data
</p>

#### - Preprocessing Data
<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Before input the data into a model prediction, firstly do preprocessing data to get a better data. Its important to do preprocessing data before build a model prediction, because no matter how good you model if the data is dirty or bad, the results will be less than optimal (like term “garbage in garbage out”). First, handle the missing data from data train and test.
</p>

<p align="center"> 
 <img src="images/missing data from data train and test.png" /> 
 <br></br>
 Missing data from data train and test
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; From the picture above, I will handle the columns ‘Age’, ‘Fare’, ‘Cabin, ‘Embarked’ from both datasets. In column ‘Age’ and ‘Fare’ we impute null value with median from each column ‘Age’ and ‘Fare’ from data train. After that, on column ‘Embarked’ I drop row of the data contains null values, because its only 2 rows with null values on ‘Embarked’ column on data train (if there are on data test, we can’t impute that with mode of values on data train (data test should have 418 row data, it’s the rules :)). And then I drop column ‘Cabin’ from both data train and test because it’s contained many null values (more than half of data).
</p>

<p align="center"> 
 <img src="images/missing data from data train and test after preprocessing data.png" /> 
 <br></br>
 Missing data from data train and test after preprocessing data
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After that, I created the columns ‘FamilySurvived’ & ‘FamilyDied’ from last name I got on column ‘Name’ values.
</p>

<p align="center"> 
 <img src="images/adds columns ‘FamilySurvived’ & ‘FamilyDied’ .png" /> 
 <br></br>
 Adds columns ‘FamilySurvived’ & ‘FamilyDied’ .png
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; And then, I identify and remove the outliers values, and then do log transform on column ‘Fare’ to make data more close into normal distribution.
</p>

<p align="center"> 
 <img src="images/distribution plot age & fare after preprocessing.png" /> 
 <br></br>
 Distribution plots age & fare after preprocessing
</p>

<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After that, I ecode categorical data into numeric category with LabelEncoder and OneHotEncoder, and then do standardize the numerical data, so that each numerical columns/features have the same scale. In the last,  I analyze and select the columns/features that used in the model prediction. Then, i split the data into predictor/input variable (X) and target/output variable (y). Here are the features and pieces of data that I used for prediction.
</p>

<p align="center"> 
 <img src="images/features for prediction.png" />
 <img src="images/pieces of data for prediction.png" /> 
 <br></br>
 Features and pieces of data for prediction
</p>

#### - Model Prediction and Results
<p align = "justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this section, I had done some experiment with several machine learning algorithms like Naive Bayes, Logistic Regression, XGBoost, K-Nearest Neighbors, etc. I used GridSearchCV to find best parameter and accuracy from each algorithms, after that i implemented on data test and submitted the results prediction to kaggle. The best score I have is 0.79425 (top 7%) with K-Nearest Neighbors algorithm (parameter : leaf_size = 1, metric = ‘minkowski’, n_neighbors = 12, p = 1, weights = ‘distance’)
</p>

<p align="center"> 
 <img src="images/model prediction codes.png" /> 
 <br></br>
 Model prediction code
</p>

<p align="center"> 
 <img src="images/gridsearchcv results.png" /> 
 <br></br>
 Gridsearchcv results
</p>

<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After that, I implemented on the data test , and then submitted the results prediction to kaggle .
</p>

<p align="center"> 
 <img src="images/prediction results.png" />
 <img src="images/score results.png" /> 
 <br></br>
 Results
</p>

<p align = "justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Based on the results obtained, I got accuracy 0.79425 (top 7 %) with KNN algorithm.
</p>

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

For details you can check my code : [Prediction Bitcoin Price with Gated Recurrent Unit (RNN).ipynb](https://github.com/rifkyahmadsaputra/Prediction-Bitcoin-Price-with-Gated-Recurrent-Unit-RNN/blob/master/Prediction%20Bitcoin%20Price%20with%20Gated%20Recurrent%20Unit%20%20(RNN).ipynb)
