##  Importing + Exploratory Data Analysis:
Firstly I imported the libraries/packages that I need to make it easier for me to write the code:

import pandas as pd 

import numpy as np 

import matplotlib as plt

import seaborn as sns

- Numpy is a python library and has a lot arrays and mathematical functions. This is something I will use in the project.
- Pandas is important because it helps me manipulate and analyse data.
- Matplotlin.pyplot is important because it is a plotting library for the Python programming language.
- Seaborn is important because it is a data visualization library built on top of matplotlib and closely integrated with pandas data structures in Python.

### EXPLORING THE DATA
Exploring the data helps form a big picture of the data that is being worked with.

COUNTING ROWS + COLUMNS OF DATA

I first counted the number of rows and columns in the data set to get familiar with what my dataset contains.

" df.shape " : This returns : (3910, 59) because there are 3910 rows and 59 columns of data. 

### Preprocessing 

Dataset does not have any missing values and already scaled so there is no need of using Normalization or Standardization.

## Preparing Data Before Model Training
### SPLITTING THE DATA

" from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101) "

NOW, I split from training and testing. The training set is the set my model learns from. This data is feeded into the model so that it can learn the relationships between the inputs and the outputs. A testing set is used to check the accuracy of your model after training. The model will never have seen the testing data during training.

I split 75% to training and 25% testing.

## Training and Evaluating the Model:

Now that I had workable data, I needed to determine a machine learning model that will determine the output after running it on the collected data.

I choose 3 different algorithms:

- Support Vector Machine:
  
  SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems.
  
- Random Forest Classifier:

  It contains multiple decision trees takes the average to improve the predictive accuracy of that dataset.

- Logistic Regression:

  It’s a supervised learning model technique. It is used for predicting the categorical dependent variable using a given set of independent variable.

After training the models

- Logistic Regression Classifier, has reached an accuracy of about **91.7%** on the testing data.

- Support Vector Machine Classifier has reached an accuracy of about **92.9%** on the testing data.

- Random Forest Classifier, has reached an accuracy of about **94.8%** on our testing data.

I have also code a classification report for each of the models, which measures the quality of the predictions made by the models.

From the accuracy and metrics above, the model that performed the best on the test data was the "Random Forest Classifier". It had an accuracy score of about **94.8%**.
It’s important to note that we can increase this accuracy with more input data and tweaking the parameters!

Random Forest Classifier Model had an Highest accuracy
So I'm choosing Random Forest Classifier model to make predictions on df_test dataset.
 
