
# coding: utf-8

# # Linear Regression [35pts (+5 bonus)]
# 
# ## Introduction
# One of the most widespread regression tools is the simple but powerful linear regression. In this notebook, you will engineer the Pittsburgh bus data into numerical features and use them to predict the number of minutes until the bus reaches the bus stop at Forbes and Morewood. 
# 
# Notebook restriction: you may not use scikit-learn for this notebook.  
# 
# ## Q1: Labeling the Dataset [8pts]
# 
# You may have noticed that the Pittsburgh bus data has a predictions table with the TrueTime predictions on arrival time, however it does not have the true label: the actual number of minutes until a bus reaches Forbes and Morewood. You will have to generate this yourself. 
# 
# Using the `all_trips` function that you implemented in homework 2, you can split the dataframe into separate trips. You will first process each trip into a form more natural for the regression setting. For each trip, you will need to locate the point at which a bus passes the bus stop to get the time at which the bus passes the bus stop. From here, you can calculate the true label for all prior datapoints, and throw out the rest. 
# 
# ### Importing functions from homework 2
# 
# Using the menu in Jupyter, you can import code from your notebook as a Python script using the following steps: 
# 1. Click File -> Download as -> Python (.py)
# 2. Save file (time_series.py) in the same directory as this notebook 
# 3. (optional) Remove all test code (i.e. lines between AUTOLAB_IGNORE macros) from the script for faster loading time
# 4. Import from the notebook with `from time_series import function_name`
# 
# ### Specifications
# 
# 1. To determine when the bus passes Morewood, we will use the Euclidean distance as a metric to determine how close the bus is to the bus stop. 
# 2. We will assume that the row entry with the smallest Euclidean distance to the bus stop is when the bus reaches the bus stop, and that you should truncate all rows that occur **after** this entry.  In the case where there are multiple entries with the exact same minimal distance, you should just consider the first one that occurs in the trip (so truncate everything after the first occurance of minimal distance). 
# 3. Assume that the row with the smallest Euclidean distance to the bus stop is also the true time at which the bus passes the bus stop. Using this, create a new column called `eta` that contains for each row, the number of minutes until the bus passes the bus stop (so the last row of every trip will have an `eta` of 0).
# 4. Make sure your `eta` is numerical and not a python timedelta object. 

# In[1]:


import pandas as pd
import numpy as np
import scipy.linalg as la
from collections import Counter
import math


# In[2]:


# AUTOLAB_IGNORE_START
from time_series import load_data, split_trips
vdf, _ = load_data('bus_train.db')
all_trips = split_trips(vdf)
# AUTOLAB_IGNORE_STOP


# In[3]:


def distance(busData, coord):
    return math.sqrt((busData["lat"]-coord[0])**2+(busData["lon"]-coord[1])**2)

def label_and_truncate(trip, bus_stop_coordinates):
    """ Given a dataframe of a trip following the specification in the previous homework assignment,
        generate the labels and throw away irrelevant rows. 
        
        Args: 
            trip (dataframe): a dataframe from the list outputted by split_trips from homework 2
            stop_coordinates ((float, float)): a pair of floats indicating the (latitude, longitude) 
                                               coordinates of the target bus stop. 
            
        Return:
            (dataframe): a labeled trip that is truncated at Forbes and Morewood and contains a new column 
                         called `eta` which contains the number of minutes until it reaches the bus stop. 
    """
    i = 0
    minind = 0 
    mindist =  100000000000;
    newtripwithresetindex = trip.reset_index()
    for ind,row in trip.iterrows():
        dist = distance(row, bus_stop_coordinates)
        if (dist < mindist) : 
            mindist = dist
            minind = i
        i+=1
    d = newtripwithresetindex.truncate(after = minind)
    d = d.set_index("tmstmp")
    arrivalTime = d.index[-1]
    times = map(lambda ind: (arrivalTime-ind).seconds//60 ,d.index.tolist())
    d["eta"] = pd.Series(list(times), index = d.index)
    return d
    pass
    
# AUTOLAB_IGNORE_START
morewood_coordinates = (40.444671114203, -79.94356058465502) # (lat, lon)
labeled_trips = [label_and_truncate(trip, morewood_coordinates) for trip in all_trips]
labeled_vdf = pd.concat(labeled_trips).reset_index()
# We remove datapoints that make no sense (ETA more than 10 hours)
labeled_vdf = labeled_vdf[labeled_vdf["eta"] < 10*60].reset_index(drop=True)
print(Counter([len(t) for t in labeled_trips]))
print(labeled_vdf.head())
# AUTOLAB_IGNORE_STOP


# For our implementation, this returns the following output
# ```python
# >>> Counter([len(t) for t in labeled_trips])
# Counter({1: 506, 21: 200, 18: 190, 20: 184, 19: 163, 16: 162, 22: 159, 17: 151, 23: 139, 31: 132, 15: 128, 2: 125, 34: 112, 32: 111, 33: 101, 28: 98, 14: 97, 30: 95, 35: 95, 29: 93, 24: 90, 25: 89, 37: 86, 27: 83, 39: 83, 38: 82, 36: 77, 26: 75, 40: 70, 13: 62, 41: 53, 44: 52, 42: 47, 6: 44, 5: 39, 12: 39, 46: 39, 7: 38, 3: 36, 45: 33, 47: 33, 43: 31, 48: 27, 4: 26, 49: 26, 11: 25, 50: 25, 10: 23, 51: 23, 8: 19, 9: 18, 53: 16, 54: 15, 52: 14, 55: 14, 56: 8, 57: 3, 58: 3, 59: 3, 60: 3, 61: 1, 62: 1, 67: 1}) 
# >>> labeled_vdf.head()
#                tmstmp   vid        lat        lon  hdg   pid   rt        des  \
# 0 2016-08-11 10:56:00  5549  40.439504 -79.996981  114  4521  61A  Swissvale   
# 1 2016-08-11 10:57:00  5549  40.439504 -79.996981  114  4521  61A  Swissvale   
# 2 2016-08-11 10:58:00  5549  40.438842 -79.994733  124  4521  61A  Swissvale   
# 3 2016-08-11 10:59:00  5549  40.437938 -79.991213   94  4521  61A  Swissvale   
# 4 2016-08-11 10:59:00  5549  40.437938 -79.991213   94  4521  61A  Swissvale   
# 
#    pdist  spd tablockid  tatripid  eta  
# 0   1106    0  061A-164      6691   16  
# 1   1106    0  061A-164      6691   15  
# 2   1778    8  061A-164      6691   14  
# 3   2934    7  061A-164      6691   13  
# 4   2934    7  061A-164      6691   13 
# ```

# ## Q2: Generating Basic Features [8pts]
# In order to perform linear regression, we need to have numerical features. However, not everything in the bus database is a number, and not all of the numbers even make sense as numerical features. If you use the data as is, it is highly unlikely that you'll achieve anything meaningful.
# 
# Consequently, you will perform some basic feature engineering. Feature engineering is extracting "features" or statistics from your data, and hopefully improve the performance of your learning algorithm (in this case, linear regression). Good features can often make up for poor model selection and improve your overall predictive ability on unseen data. In essence, you want to turn your data into something your algorithm understands. 
# 
# ### Specifications
# 1. The input to your function will be a concatenation of the trip dataframes generated in Q1 with the index dropped (so same structure as the original dataframe, but with an extra column and less rows). 
# 2. Linear models typically have a constant bias term. We will encode this as a column of 1s in the dataframe. Call this column 'bias'. 
# 2. We will keep the following columns as is, since they are already numerical:  pdist, spd, lat, lon, and eta 
# 3. Time is a cyclic variable. To encode this as a numerical feature, we can use a sine/cosine transformation. Suppose we have a feature of value f that ranges from 0 to N. Then, the sine and cosine transformation would be $\sin\left(2\pi \frac{f}{N}\right)$ and $\cos\left(2\pi \frac{f}{N}\right)$. For example, the sine transformation of 6 hours would be $\sin\left(2\pi \frac{6}{24}\right)$, since there are 24 hours in a cycle. You should create sine/cosine features for the following:
#     * day of week (cycles every week, 0=Monday)
#     * hour of day (cycles every 24 hours, 0=midnight)
#     * time of day represented by total number of minutes elapsed in the day (cycles every 60*24 minutes, 0=midnight).
# 4. Heading is also a cyclic variable, as it is the ordinal direction in degrees (so cycles every 360 degrees). 
# 4. Buses run on different schedules on the weekday as opposed to the weekend. Create a binary indicator feature `weekday` that is 1 if the day is a weekday, and 0 otherwise. 
# 5. Route and destination are both categorical variables. We can encode these as indicator vectors, where each column represents a possible category and a 1 in the column indicates that the row belongs to that category. This is also known as a one hot encoding. Make a set of indicator features for the route, and another set of indicator features for the destination. 
# 6. The names of your indicator columns for your categorical variables should be exactly the value of the categorical variable. The pandas function `pd.DataFrame.get_dummies` will be useful. 

# In[4]:


# def create_dummies(data):
#     return pd.get_dummies(data, columns = ["des", "rt"], prefix = '', prefix_sep = '')

def create_features(vdf):
    """ Given a dataframe of labeled and truncated bus data, generate features for linear regression. 
    
        Args:
            df (dataframe) : dataframe of bus data with the eta column and truncated rows
        Return: 
            (dataframe) : dataframe of features for each example
        """
    vdf["bias"] = pd.Series([1]*vdf.shape[0])
    vdf["sin_day_of_week"] = pd.Series(list(map(lambda ind: math.sin(2*math.pi*ind.weekday()/7),vdf["tmstmp"].tolist())))
    vdf["sin_hour_of_day"] = pd.Series(list(map(lambda ind: math.sin(2*math.pi*ind.hour/24),vdf["tmstmp"].tolist())))
    vdf["cos_day_of_week"] = pd.Series(list(map(lambda ind: math.cos(2*math.pi*ind.weekday()/7),vdf["tmstmp"].tolist())))
    vdf["cos_hour_of_day"] = pd.Series(list(map(lambda ind: math.cos(2*math.pi*ind.hour/24),vdf["tmstmp"].tolist())))
    vdf["sin_time_of_day"] = pd.Series(list(map(lambda ind: math.sin(2*math.pi*(ind.hour*60+ind.minute)/(60*24)),vdf["tmstmp"].tolist())))
    vdf["cos_time_of_day"] = pd.Series(list(map(lambda ind: math.cos(2*math.pi*(ind.hour*60+ind.minute)/(60*24)),vdf["tmstmp"].tolist())))
    vdf["sin_hdg"] = pd.Series(list(map(lambda ind: math.sin(2*math.pi*ind/360),vdf["hdg"])))
    vdf["cos_hdg"] = pd.Series(list(map(lambda ind: math.cos(2*math.pi*ind/360),vdf["hdg"])))
    vdf["weekday"] = pd.Series(list(map(lambda ind: 1 if ind.weekday()<5 else 0 ,vdf["tmstmp"].tolist())))
    vdf["Downtown"] = pd.Series(list(map(lambda ind: 1 if ind == "Downtown" else 0 ,vdf["des"])))
    vdf["Swissvale"] = pd.Series(list(map(lambda ind: 1 if ind == "Swissvale" else 0 ,vdf["des"])))
    vdf["Murray-Waterfront"] = pd.Series(list(map(lambda ind: 1 if ind == "Murray-Waterfront" else 0 ,vdf["des"])))
    vdf["McKeesport "] = pd.Series(list(map(lambda ind: 1 if ind == "McKeesport " else 0 ,vdf["des"])))
    vdf["Greenfield Only"] = pd.Series(list(map(lambda ind: 1 if ind == "Greenfield Only" else 0 ,vdf["des"])))
    vdf["Braddock "] = pd.Series(list(map(lambda ind: 1 if ind == "Braddock " else 0 ,vdf["des"])))
    vdf["61A"] = pd.Series(list(map(lambda ind: 1 if ind == "61A" else 0 ,vdf["rt"])))
    vdf["61B"] = pd.Series(list(map(lambda ind: 1 if ind == "61B" else 0 ,vdf["rt"])))
    vdf["61C"] = pd.Series(list(map(lambda ind: 1 if ind == "61C" else 0 ,vdf["rt"])))
    vdf["61D"] = pd.Series(list(map(lambda ind: 1 if ind == "61D" else 0 ,vdf["rt"])))
    vdf = vdf.drop(["tmstmp","rt","des","hdg","vid","tablockid","tatripid","pid"], axis = 1)
    return vdf
    pass

# AUTOLAB_IGNORE_START
vdf_features = create_features(labeled_vdf)
# AUTOLAB_IGNORE_STOP


# In[5]:


# AUTOLAB_IGNORE_START
with pd.option_context('display.max_columns', 26):
    print(vdf_features.columns)
    print(vdf_features.head())
# AUTOLAB_IGNORE_STOP


# Our implementation has the following output. Verify that your code has the following columns (order doesn't matter): 
# ```python
# >>> vdf_features.columns
# Index([             u'bias',             u'pdist',               u'spd',
#                      u'lat',               u'lon',               u'eta',
#                  u'sin_hdg',           u'cos_hdg',   u'sin_day_of_week',
#          u'cos_day_of_week',   u'sin_hour_of_day',   u'cos_hour_of_day',
#          u'sin_time_of_day',   u'cos_time_of_day',           u'weekday',
#                u'Braddock ',          u'Downtown',   u'Greenfield Only',
#              u'McKeesport ', u'Murray-Waterfront',         u'Swissvale',
#                      u'61A',               u'61B',               u'61C',
#                      u'61D'],
#       dtype='object')
#    bias  pdist  spd        lat        lon  eta   sin_hdg   cos_hdg  \
# 0   1.0   1106    0  40.439504 -79.996981   16  0.913545 -0.406737   
# 1   1.0   1106    0  40.439504 -79.996981   15  0.913545 -0.406737   
# 2   1.0   1778    8  40.438842 -79.994733   14  0.829038 -0.559193   
# 3   1.0   2934    7  40.437938 -79.991213   13  0.997564 -0.069756   
# 4   1.0   2934    7  40.437938 -79.991213   13  0.997564 -0.069756   
# 
#    sin_day_of_week  cos_day_of_week ...   Braddock   Downtown  \
# 0         0.433884        -0.900969 ...         0.0       0.0   
# 1         0.433884        -0.900969 ...         0.0       0.0   
# 2         0.433884        -0.900969 ...         0.0       0.0   
# 3         0.433884        -0.900969 ...         0.0       0.0   
# 4         0.433884        -0.900969 ...         0.0       0.0   
# 
#    Greenfield Only  McKeesport   Murray-Waterfront  Swissvale  61A  61B  61C  \
# 0              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
# 1              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
# 2              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
# 3              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
# 4              0.0          0.0                0.0        1.0  1.0  0.0  0.0   
# 
#    61D  
# 0  0.0  
# 1  0.0  
# 2  0.0  
# 3  0.0  
# 4  0.0  
# 
# [5 rows x 25 columns]
# ```

# ## Q3 Linear Regression using Ordinary Least Squares [10 + 4pts]
# Now you will finally implement a linear regression. As a reminder, linear regression models the data as
# 
# $$\mathbf y = \mathbf X\mathbf \beta + \mathbf \epsilon$$
# 
# where $\mathbf y$ is a vector of outputs, $\mathbf X$ is also known as the design matrix, $\mathbf \beta$ is a vector of parameters, and $\mathbf \epsilon$ is noise. We will be estimating $\mathbf \beta$ using Ordinary Least Squares, and we recommending following the matrix notation for this problem (https://en.wikipedia.org/wiki/Ordinary_least_squares). 
# 
# ### Specification
# 1. We use the numpy term array-like to refer to array like types that numpy can operate on (like Pandas DataFrames). 
# 1. Regress the output (eta) on all other features
# 2. Return the predicted output for the inputs in X_test
# 3. Calculating the inverse $(X^TX)^{-1}$ is unstable and prone to numerical inaccuracies. Furthermore, the assumptions of Ordinary Least Squares require it to be positive definite and invertible, which may not be true if you have redundant features. Thus, you should instead use $(X^TX + \lambda*I)^{-1}$ for identity matrix $I$ and $\lambda = 10^{-4}$, which for now acts as a numerical "hack" to ensure this is always invertible. Furthermore, instead of computing the direct inverse, you should utilize the Cholesky decomposition which is much more stable when solving linear systems. 

# In[6]:


class LR_model():
    """ Perform linear regression and predict the output on unseen examples. 
        Attributes: 
            beta (array_like) : vector containing parameters for the features """
    
    def __init__(self, X, y):
        """ Initialize the linear regression model by computing the estimate of the weights parameter
            Args: 
                X (array-like) : feature matrix of training data where each row corresponds to an example
                y (array like) : vector of training data outputs 
            """
        shape = X.shape[1]
        otherintermediatevar = np.dot(X.T,y)
        intermediatevar = la.cho_factor(np.dot(X.T, X) + (1/(10 ** 4) * np.identity(shape)))
        self.beta = la.cho_solve(intermediatevar,otherintermediatevar)
        pass
        
    def predict(self, X_p): 
        """ Predict the output of X_p using this linear model. 
            Args: 
                X_p (array_like) feature matrix of predictive data where each row corresponds to an example
            Return: 
                (array_like) vector of predicted outputs for the X_p
            """
        return np.dot(X_p, self.beta)


# We have provided some validation data for you, which is another scrape of the Pittsburgh bus data (but for a different time span). You will need to do the same processing to generate labels and features to your validation dataset. Calculate the mean squared error of the output of your linear regression on both this dataset and the original training dataset. 
# 
# How does it perform? One simple baseline is to make sure that it at least predicts as well as predicting the mean of what you have seen so far. Does it do better than predicting the mean? Compare the mean squared error of a predictor that predicts the mean vs your linear classifier. 
# 
# ### Specifications
# 1. Build your linear model using only the training data
# 2. Compute the mean squared error of the predictions on both the training and validation data. 
# 3. Compute the mean squared error of predicting the mean of the **training outputs** for all inputs. 
# 4. You will need to process the validation dataset in the same way you processed the training dataset.
# 5. You will need to split your features from your output (eta) prior to calling compute_mse

# In[7]:


# Calculate mean squared error on both the training and validation set
def compute_mse(LR, X, y, X_v, y_v):
    """ Given a linear regression model, calculate the mean squared error for the 
        training dataset, the validation dataset, and for a mean prediction
        Args:
            LR (LR_model) : Linear model
            X (array-like) : feature matrix of training data where each row corresponds to an example
            y (array like) : vector of training data outputs 
            X_v (array-like) : feature matrix of validation data where each row corresponds to an example
            y_v (array like) : vector of validation data outputs 
        Return: 
            (train_mse, train_mean_mse, 
             valid_mse, valid_mean_mse) : a 4-tuple of mean squared errors
                                             1. MSE of linear regression on the training set
                                             2. MSE of predicting the mean on the training set
                                             3. MSE of linear regression on the validation set
                                             4. MSE of predicting the mean on the validation set
                         
            
    """
    predictedy = LR.predict(X)
    predictedy_v = LR.predict(X_v)
    train_mse = np.mean((y-predictedy)**2)
    train_mean_mse = np.mean((y - predictedy.mean())**2)
    valid_mse = np.mean((y_v - predictedy_v)**2)
    valid_mean_mse = np.mean((y_v - predictedy.mean())**2)
    return (train_mse, train_mean_mse, valid_mse, valid_mean_mse)
    pass



# In[1]:


# AUTOLAB_IGNORE_START
# First you should replicate the same processing pipeline as we did to the training set
vdf_valid, pdf_valid = load_data('bus_valid.db')
all_trips_valid = split_trips(vdf_valid)
labeled_trips_valid = [label_and_truncate(trip, morewood_coordinates) for trip in all_trips_valid]
labeled_vdf_valid = pd.concat(labeled_trips_valid).reset_index()
labeled_vdf_valid = labeled_vdf_valid[labeled_vdf_valid["eta"] < 10*60].reset_index(drop=True)
vdf_features_valid = create_features(labeled_vdf_valid)

# Separate the features from the output and pass it into your linear regression model.
X_df = vdf_features.drop("eta", axis = 1)
y_df = vdf_features["eta"]
X_valid_df = vdf_features_valid.drop("eta", axis = 1)
y_valid_df = vdf_features_valid["eta"]
LR = LR_model(X_df, y_df)
print(compute_mse(LR, X_df, y_df, X_valid_df, y_valid_df))

# AUTOLAB_IGNORE_STOP


# As a quick check, our training data MSE is approximately 38.99. 
# (38.9938263046819, 130.5804831652153, 49.03443982863896, 150.20916602723423)
# 

# ## Q4 TrueTime Predictions [5pts]
# How do you fare against the Pittsburgh Truetime predictions? In this last problem, you will match predictions to their corresponding vehicles to build a dataset that is labeled by TrueTime. Remember that we only evaluate performance on the validation set (never the training set). How did you do?
# 
# ### Specification
# 1. You should use the pd.DataFrame.merge function to combine your vehicle dataframe and predictions dataframe into a single dataframe. You should drop any rows that have no predictions (see the how parameter). (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html)
# 2. You can compute the TrueTime ETA by taking their predicted arrival time and subtracting the timestamp, and converting that into an integer representing the number of minutes. 
# 3. Compute the mean squared error for linear regression only on the rows that have predictions (so only the rows that remain after the merge). 

# In[9]:


def compare_truetime(LR, labeled_vdf, pdf):
    """ Compute the mse of the truetime predictions and the linear regression mse on entries that have predictions.
        Args:
            LR (LR_model) : an already trained linear model
            labeled_vdf (pd.DataFrame): a dataframe of the truncated and labeled bus data (same as the input to create_features)
            pdf (pd.DataFrame): a dataframe of TrueTime predictions
        Return: 
            (tt_mse, lr_mse): a tuple of the TrueTime MSE, and the linear regression MSE
        """
    vdf = create_features(labeled_vdf)
    y_predict = LR.predict(vdf.drop(["eta"],axis = 1))
    labeled_vdf["eta_lr"] = y_predict
    df = labeled_vdf.merge(pdf, how = "inner")
    eta_tt = list(map(lambda ind: (ind[1]-ind[0]).seconds//60 ,zip(df["tmstmp"].tolist(),df["prdtm"].tolist())))
    lr_mse = np.mean((eta_tt - df.eta)**2)
    tt_mse = np.mean((df.eta_lr - df.eta)**2)
    return lr_mse, tt_mse
    pass
    
# AUTOLAB_IGNORE_START
print(compare_truetime(LR, labeled_vdf_valid, pdf_valid))
# AUTOLAB_IGNORE_STOP


# As a sanity check, your linear regression MSE should be approximately 50.20. 
# (50.20239900730732, 60.40782041336532)

# ## Q5 Feature Engineering contest (bonus)
# 
# You may be wondering "why did we pick the above features?" Some of the above features may be entirely useless, or you may have ideas on how to construct better features. Sometimes, choosing good features can be the entirety of a data science problem. 
# 
# In this question, you are given complete freedom to choose what and how many features you want to generate. Upon submission to Autolab, we will run linear regression on your generated features and maintain a scoreboard of best regression accuracy (measured by mean squared error). 
# 
# The top scoring students will receive a bonus of 5 points. 
# 
# ### Tips:
# * Test your features locally by building your model using the training data, and predicting on the validation data. Compute the mean squared error on the **validation dataset** as a metric for how well your features generalize. This helps avoid overfitting to the training dataset, and you'll have faster turnaround time than resubmitting to autolab. 
# * The linear regression model will be trained on your chosen features of the same training examples we provide in this notebook. 
# * We test your regression on a different dataset from the training and validation set that we provide for you, so the MSE you get locally may not match how your features work on the Autolab dataset. 
# * We will solve the linear regression using Ordinary Least Squares with regularization $\lambda=10^{-4}$ and a Cholesky factorization, exactly as done earlier in this notebook. 
# * Note that the argument contains **UNlabeled** data: you cannot build features off the output labels (there is no ETA column). This is in contrast to before, where we kept everything inside the same dataframe for convenience. You can produce the sample input by removing the "eta" column, which we provide code for below. 
# * Make sure your features are all numeric. Try everything!

# In[10]:


def contest_features(vdf, vdf_train):
    """ Given a dataframe of UNlabeled and truncated bus data, generate ANY features you'd like for linear regression. 
        Args:
            vdf (dataframe) : dataframe of bus data with truncated rows but unlabeled (no eta column )
                              for which you should produce features
            vdf_train (dataframe) : dataframe of training bus data, truncated and labeled 
        Return: 
            (dataframe) : dataframe of features for each example in vdf
        """
    # create your own engineered features
    pass
    
# AUTOLAB_IGNORE_START
# contest_cols = list(labeled_vdf.columns)
# contest_cols.remove("eta")
# contest_features(labeled_vdf_valid[contest_cols], labeled_vdf).head()
# AUTOLAB_IGNORE_STOP

