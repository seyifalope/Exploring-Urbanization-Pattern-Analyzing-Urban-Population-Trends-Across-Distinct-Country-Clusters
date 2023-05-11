import pandas as pd
import numpy as np

import sklearn.cluster as cluster
import sklearn.metrics as skmet

import seaborn as sns
import matplotlib.pyplot as plt
import cluster_tools as ct
import scipy.optimize as opt
import itertools as iter

"""
Creating a def function to read in our datasets and skiprows the first 4 rows
"""


def read_data(filename, **others):
    """
    A function that reads in climate change data and returns the skip the first 4 rows
        filename: the name of the world bank data that will be read for analysis 
        and manupulation

        **others: other arguments to pass into the functions as need be, such
        as skipping the first 4 rows

    Returns: 
        The  dataset that has been read in with its first 4 rows skipped
    """

    # Reading in the climate dataset for to be used for analysis with first 4 rows skipped
    world_data = pd.read_csv(filename, skiprows=4)

    return world_data


# for this analysis, i will be reading in  'Urban population (% of total population',
urban = read_data(
    'API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_5359159.csv', skiprows=4)
# Checking decriptive statistics

print(urban.describe())

# Selecting  the countries the needed
urban = urban[urban['Country Name'].isin(['China', 'Ghana', 'Kenya', 'United States',
                                         'United Kingdom', 'Zambia', 'France',  'Angola', 'Cuba', 'Sudan'])]

# Dropping the columns not needed
urban = urban.drop(
    ['Indicator Name', 'Country Code', 'Indicator Code'], axis=1)
