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

# reseting the index
urban.reset_index(drop=True, inplace=True)

# Extracting  years from our urban out dataset

urban_s = urban[['Country Name', '1990', '2000', '2010', '2019']]
print(urban_s.describe())

# Checking for missing values
print(urban_s.isna().sum())

# Transposing the data
urban_t = urban_s.T

urban_t.columns = urban_t.iloc[0]  # setting the columns
urban_t = urban_t.iloc[1:]  # filtering for only years
print(urban_t.describe())
urban_t = urban_t.apply(pd.to_numeric)  # converting the datatype to numeric
urban_t.plot()


# Extracting 30 years of data at an interval of 10 years from out dataset
urban_year = urban_s[['1990', '2000', '2010', '2019']]
urban_year.describe()

# Checking for missing values
urban_year.isna().sum()

# Checking for any missing values
urban_year.isna().sum()

# dropping the missing values
urban_year.dropna(inplace=True)

# Checking for correlation between our years choosen

# Correlation
corr = urban_year.corr()
corr
# heatmap
ct.map_corr(urban_year)

print(corr)

# scatter plot
pd.plotting.scatter_matrix(urban_year, figsize=(9.0, 9.0))
plt.tight_layout()    # helps to avoid overlap of labels
plt.show()

urban_cluster = urban_year[['1990', '2019']].copy()

print(urban_cluster)

# Normalizing the data and storing minimum and maximum value
urban_norm, urban_min, urban_max = ct.scaler(urban_cluster)
print(urban_norm.describe())


# Calculating the best clustering number
for i in range(2, 9):
    # creating  kmeans and fit
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(urban_cluster)

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(i, skmet.silhouette_score(urban_cluster, labels))

"""    
2 and 3 has the highest silhoutte score respectively , 
so i will be plotting for 2 and 3 clusters and choosing the best
"""

# Plotting for 3 clusters
nclusters = 3 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nclusters)
kmeans.fit(urban_norm)     

# extract labels and cluster centres
labels = kmeans.labels_

# extracting the estimated number of cluster
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(urban_norm["1990"], urban_norm["2019"], c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# show cluster centres
xcen = cen[:,0]
ycen = cen[:,1]
plt.scatter(xcen, ycen, c="k", marker="d", s=80)
# c = colour, s = size

plt.xlabel("Urban(1990)")
plt.ylabel("Urban(2019)")
plt.title("3clusters")
plt.show()

# Scaling back to the original data and creating a plot it on the original scale

plt.style.use('default')
plt.figure(dpi=300)

# now using the original dataframe
plt.scatter(urban_s["1990"], urban_s["2019"], c=labels, cmap="tab10")


# rescale and show cluster centres
scen = ct.backscale(cen, urban_min, urban_max)
xc = scen[:,0]
yc = scen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=80)

plt.xlabel("1990")
plt.ylabel("2020")
plt.title("Urban population (% of total population)")
plt.show()
