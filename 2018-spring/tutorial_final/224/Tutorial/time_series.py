
# coding: utf-8

# # Introduction
# 
# In this problem you will be analysing time series data. Specifically, the data that you will be working with has been collecting via Pittsburgh's TrueTime system which is available publicly here http://truetime.portauthority.org/bustime/login.jsp. If you're interested, you can request an API key and collect the data yourself (the process is just submitting a form), however we've already collected some smaller subset of the available data for the purposes of this assignment. 

# ## Part 1 TrueTime dataset
# 
# The bus data has been collected by querying the TrueTime API every minute. Each time, we make two requests: 
# 
# 1. We request vehicle information for every bus running on the 61A, 61B, 61C, and 61D bus routes. 
# 2. We request all available time predictions for the CMU / Morewood bus stop in both outbound and inbound directions. 
# 
# The results are given as XML, which are consequently parsed and stored within a sqlite database with two tables, one for vehicles and one for predictions. The table for the vehicles is organized in the following manner.  
# 
# | | **vehicles**             | 
# |----------|-------------|
# | vid      | vehicle identifier |
# | tmstmp | date and time of the last positional update of the vehicle |
# | lat | latitude position of the vehicle in decimal degrees |
# | lon | longitude position of the vehicle in decimal degrees |
# | hdg | heading of vehicle as a 360 degree value (0 is north, 90 is east, 180 is south, and 270 is west |
# | pid | pattern ID of trip currently being executed | 
# | rt | route that is currently being execute | 
# | des | destination of the current trip | 
# | pdist | linear distance (feet) vehicle has traveled into the current pattern |
# |  spd | speed as reported from the vehicle in miles per hour | 
# | tablockid | TA's version of the scheduled block identifier for work currently behind performed |
# | tatripid | TA's version of the scheduled trip identifier for the vehicle's current trip |
# 
# The table for the predictions is organized in the following manner
# 
# | | **predictions** | 
# |---|---|
# | tmstmp | date and time the prediction was generated |
# | typ | type of prediction (A for arrival, D for a departure) | 
# | stpnm | display name of the stop for which this prediction was generated |
# | stpid | unique identifier representing the stop for which this prediction was generated |
# | vid | unique ID of the vehicle for which this prediction was generated |
# | dstp | linear distance (feet) left to be traveled by the vehicle before it reaches the stop for which this prediction was generated |
# | rt | route for which this prediction was generated | 
# | rtdd | language-specific route designator meant for display |
# | rtdir | direction of travel of the route associated with this prediction |
# | des | final destination of the vehicle associated with this prediction |
# | prdtm | predicted date and time of a vehicle's arrival or departure to the stop associated with this prediction | 
# | dly | true if the vehicle is delayed, only present if the vehicle that generated this prediction is delayed | 
# | tablockid | TA's version of the scheduled block identifier for work currently behind performed |
# | tatripid | TA's version of the scheduled trip identifier for the vehicle's current trip |
#     

# First you will need to read in the data. We have dumped the raw form of the data into a sqlite database, which you can read directly into a pandas dataframe. 
# 
# Since this data has not been processed at all by the course staff, you will need to fix and canonicalize a few things. 
# 
# ### Specification
# 
# 1. Sometimes the TrueTime API returns a bogus result that has all the attributes but empty strings for all the values. You should inspect the data for clearly useless entries and remove all offending rows. 
# 
# 2. If you check the datatype of each column, you'll notice that most columns are stored as objects. However, some of these columns are in fact integers or floats, and if you wish to run numerical functions on them (like with numpy) you'll need to convert the columns to the correct type. Note that strings show up as objects. This is because the underlying implementation of Pandas uses numpy arrays, which need fixed-size entries, so they store pointers to strings. Your dataframe datatypes should match the following order and types (your datatypes may be 32bit instead of 64bit depending on your platform): 
# 
#    ```python
#    >>> vdf.dtypes
#    vid                   int64
#    tmstmp       datetime64[ns]
#    lat                 float64
#    lon                 float64
#    hdg                   int64
#    pid                   int64
#    rt                   object
#    des                  object
#    pdist                 int64
#    spd                   int64
#    tablockid            object
#    tatripid              int64
#    dtype: object
# 
#    >>> pdf.dtypes
#    tmstmp       datetime64[ns]
#    typ                  object
#    stpnm                object
#    stpid                 int64
#    vid                   int64
#    dstp                  int64
#    rt                   object
#    rtdd                 object
#    rtdir                object
#    des                  object
#    prdtm        datetime64[ns]
#    dly                    bool
#    tablockid            object
#    tatripid              int64
#    dtype: object
#    ```
# 
# 3. As you may have noticed from the above data types, you should convert all timestamps to Pandas datetime objects. 

# In[5]:


import pandas as pd
import numpy as np
import sqlite3


# In[6]:


def load_data(fname):
    """ Read the given database into two pandas dataframes. 
    
    Args: 
        fname (string): filename of sqlite3 database to read
        
    Returns:
        (pd.DataFrame, pd.DataFrame): a tuple of two dataframes, the first for the vehicle data and the 
                                      second for the prediction data. 
    """
    vdf = pd.read_sql_query("Select * from vehicles",sqlite3.connect(fname), parse_dates = {"tmstmp" : "ns"})
    vdf = vdf[vdf["vid"] != ""]
    #vdf.replace('', np.nan,inplace = True)
    #vdf = vdf.dropna(axis = 0, how = "any")
    for col in ["vid", "pdist", "pid", "hdg", "spd", "tatripid"]:
        vdf[col] = vdf[col].astype(int)
    for col in ["lat", "lon"]:
        vdf[col] = vdf[col].astype(float)
    pdf = pd.read_sql("Select * from predictions",sqlite3.connect(fname),parse_dates = {"tmstmp" : "ns", "prdtm": "ns"})
    pdf["dly"] = pdf.dly.astype(bool)
    pdf = pdf[pdf["vid"] != ""]
    #pdf.replace('', np.nan,inplace = True)
    #pdf = pdf.dropna(axis = 0, how = "any")
    for col in ["stpid", "vid", "dstp", "tatripid"]:
        pdf[col] = pdf[col].astype(int)
    return (vdf, pdf)
    pass
    


# In[7]:


# # AUTOLAB_IGNORE_START
# vdf, pdf = load_data('bus_aug23.db')


# # Inspect the datatypes of the dataframe
# print(vdf.dtypes)
# print(pdf.dtypes)

# print(len(vdf), len(pdf))

# # Inspect the first five entries of the dataframe
# print(vdf.head())
# print(pdf.head())
# # AUTOLAB_IGNORE_STOP


# ## Part 2 Splitting Trips
# 
# For this assignment, we will focus on the vehicles dataframe and come back to the predictions later. The next thing we will do is take the dataframe of vehicles and and split it into individual trips. Specifically, a trip is the sequence of rows corresponding to a single bus, typically at one minute intervals, from the start of its route to the end of its route. We will represent each trip as an individual dataframe, and create a list of trip dataframes to represent all the trips.
# 
# ### Specification
# 1. All entries in a trip should belong to a single route, destination, pattern, and vehicle.
# 
# 2. The entries in a trip should have (not strictly) monotonically increasing timestamps and distance traveled. 
# 
# 3. Each trip should be of maximal size. I.e. you should sort first by time, and secondarily by pdist, and use a drop in a pdist as an indication that a new trip has started. 
# 
# 3. Each trip should have the timestamp set as the index, named `tmstmp`

# In[27]:


def split_trips(df):
    """ Splits the dataframe of vehicle data into a list of dataframes for each individual trip. 
    
    Args: 
        df (pd.DataFrame): A dataframe containing vehicle data
        
    Returns: 
        (list): A list of dataframes, where each dataFrame contains vehicle data for a single trip
    """
    trips = []
    for g, d in df.groupby(["vid","des", "rt"]):
        d = d.sort_values(["tmstmp","pdist"])
        d = d.set_index("tmstmp")
        prevpdist = -1
        previndex = 0
        i = 0
        for index,row in d.iterrows():
            if row["pdist"] < prevpdist:
                trips.append(d[previndex:i])
                previndex = i
            prevpdist = row["pdist"]
            i+=1
        trips.append(d[previndex:])
    #print(trips[0])
    return trips
    

# # AUTOLAB_IGNORE_START    
# all_trips = { rt : split_trips(vdf[vdf["rt"]==rt]) for rt in ["61A", "61B", "61C", "61D"] }
# #print(all_trips.keys())
# sums = 0
# for trip in all_trips["61A"]:
#     sums+=len(trip.index)
# print(sums,len(vdf[vdf["rt"]=="61A"].index))
# # AUTOLAB_IGNORE_STOP


# You should verify that your code meets the specification above. Use this cell to write tests that check the validity of your resulting output. For example,you should test that the total number of datapoints in the list of dataframes is the same as the number of datapoints in the original dataframe. 

# In[ ]:


# Test the validity of your code here



# ## Part 3 Sliding Averages
# 
# Let's compute a basic statistic for time series / sequential data, which is the sliding average. Sliding averages are typically used to smooth out short-term fluctuations to see the long-term patterns. 
# 
# While it would be fairly simple to directly construct a list of all the sliding averages from the existing dataset, in reality, new TrueTime bus data is constantly becoming available. We want to avoid recomputing all the sliding averages every time a new data point comes in. Thus, instead of storing an unbounded list of datapoints, we will construct a class which does constant time updates as new data comes in. 
# 
# ### Specifications
# 1. Your function should not use more than O(k) memory, where k is the number of elements to average. 
# 2. Each update should do O(1) work. 
# 3. We will use a centered sliding average: we will average the k values both before and after the center point, averaging a total of 2k+1 elements. Note that k=0 will just return the stream without any averaging. 
# 4. Since the average depends on both past and future elements, the `update` function will not be able to output anything useful for the first k elements. You should output `None` during these iterations. We suggest you signify the end of the stream by calling `update(None)` k times, during which you should output the last k sliding averages. 
# 4. When at the beginning or end of a list, just compute the average of elements that exist. 
# 5. As usual, you should test the correctness of your code. You can do this in the same cell or make a new cell.
# 
# Note: you may find the `collections.deque` data structure to be helpful. 
# 
# Example: 
# ```python
# >>> compute_sliding_averages(pd.Series([1,2,3,4,5]),1)
# pdf.Series([1.5, 2.0, 3.0, 4.0, 4.5])
# ```

# In[28]:


from collections import deque

class SlidingAverage:
    def __init__(self,k):
        """ Initializes a sliding average calculator which keeps track of the average of the last k seen elements. 
        
        Args: 
            k (int): the number of elements to average (the half-width of the sliding average window)
        """
        self.queue = deque([])
        self.k = k
        self.sum = 0
        self.len = 0
        self.Nonelen = 0
        pass
        
        
    def update(self,x):
        """ Computes the sliding average after having seen element x 
        
        Args:
            x (float): the next element in the stream to view
            
        Returns: 
            (float): the new sliding average after having seen element x, if it can be calculated
        """
        if (x == None):
            self.Nonelen+= 1
        if (x != None):
            self.queue.append(x)
            self.sum += x
            self.len += 1
            self.Nonelen+= 1
        if (self.Nonelen>(self.k*2+1)):#or (x == None and self.Nonelen>(self.k*2+1))):
            num = self.queue.popleft()
            self.sum -= num
            self.Nonelen -= 1
            self.len -= 1
        if (self.Nonelen > self.k):
            return self.sum/self.len
        else: return None
        pass
    

def compute_sliding_averages(s, k):
    """ Computes the sliding averages for a given Pandas series using the SlidingAverage class. 
    
    Args:
        s (pd.Series): a Pandas series for which the sliding average needs to be calculated
        k (int): the half-width of the sliding average window 
        
    Returns:
        (pd.Series): a Pandas series of the sliding averages
    
    """
    res = []
    slide = SlidingAverage(k)
    for num in s.values:
        slidingavg = slide.update(num)
        if (slidingavg != None):
            res.append(slidingavg)
    for i in range(k):
        slidingavg = slide.update(None)
        if (slidingavg != None):
            res.append(slidingavg)
    return pd.Series(res)
    pass
    

# AUTOLAB_IGNORE_START    
# print(compute_sliding_averages(pd.Series([1,2,3,4,5]),1))
# AUTOLAB_IGNORE_STOP


# ## Part 4 Time Series Visualizations
# 
# Time series data is typically displayed as signals over time. For example, this could be the speed of the bus over time, or the number of minutes behind or ahead of schedule a bus is. 

# In[20]:


# import matplotlib
# # Use svg backend for better quality
# # AUTOLAB_IGNORE_START
# matplotlib.use("svg")
# # AUTOLAB_IGNORE_STOP
# import matplotlib
# import matplotlib.pyplot as plt
# # AUTOLAB_IGNORE_START
# %matplotlib inline
# plt.style.use('ggplot')
# matplotlib.rcParams['figure.figsize'] = (10.0, 5.0) # you should adjust this to fit your screen
# # AUTOLAB_IGNORE_STOP


# As the first example, you'll plot the speed of the bus as a function of time. Here, we'll overlay multiple routes on a single plot. Can you determine the direction of the bus (to or away from downtown) from the signal? 
# 
# ### Specification:
# 1. Plot the sliding average speed of each bus, using a new line for each bus. 
# 2. Return a list of the resulting `Line2D` objects plot. The order of the line objects should correspond with the order of the trips. 
# 3. Do not call `plt.show()` inside the function. Autolab will not X out of any plotted images. 
# 4. We want you to plot the bus as a function of time, and not of datetime (these are different python types) which is the index of your dataframe. You can get the time with df.index.time. 

# In[136]:


def plot_trip(trips, k):
    """ Plots the sliding average speed as a function of time 
    
    Args: 
        trip (list): list of trip DataFrames to plot
        k (int): the half-width of the sliding average window
    """
    res = []
    for trip in trips:
        ydata = compute_sliding_averages(trip["spd"],k)
        xdata = trip.index.time
        res.append(matplotlib.lines.Line2D(xdata,ydata))
    return res


# Play around with these values. Can you differentiate the buses going towards downtown from the buses going away from downtown?
# AUTOLAB_IGNORE_START
# lines = plot_trip(all_trips['61A'][:20], 15)
# AUTOLAB_IGNORE_STOP


# We can also gain information from overall trends from averaging many data points. In the following function, you will plot the average speed of all buses at regular time intervals throughout the day. 
# 
# ### Specification
# 1. You should group the rows of the dataframe by taking the timestamp modulo t and ignoring the day/month/year (since the data was collected at 1 minute intervals, this means that t=1 corresponds to averaging one entry per day recorded).
# 2. Return the PathCollection object of your plot. For example, if you create the plot using the matplotlib command `scatter(...)`, return the result of this function call. 
# 3. Do not call `plt.show()` inside the function. Autolab will not X out of any plotted images. 

# In[36]:


import datetime




def plot_avg_spd(df, t):
    """ Plot the average speed of all recorded buses within t minute intervals 
    Args: 
        df (pd.DataFrame): dataframe of bus data
        t (int): the granularity of each time period (in minutes) for which an average is speed is calculated
    """
    timeseries = [df["tmstmp"].apply(lambda x: ((x.hour*60+x.minute)//t *t))]
    d = df.groupby(timeseries)
    x = []
    y = []
    for group,dp in d:
        time = pd.to_datetime(dp.tmstmp.values[0]).time()
        xval = datetime.time(time.hour,((time.hour*60+time.minute)//t)*t-time.hour*60)
        print(xval)
        x.append(xval) 
        y.append(dp.spd.mean())
    for i in range(len(x)):
        pass
        #print (type(x))
    return plt.scatter(x,y)
    pass

# AUTOLAB_IGNORE_START
# s = plot_avg_spd(vdf, 10)
# print(s)
# plt.show()
# AUTOLAB_IGNORE_STOP

