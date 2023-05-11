import pandas as pd
import numpy as np

import sklearn.cluster as cluster
import sklearn.metrics as skmet

import seaborn as sns
import matplotlib.pyplot as plt
import cluster_tools as ct
import scipy.optimize as opt
from scipy.optimize import curve_fit
import errors as err
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
nclusters = 3  # number of cluster centres

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
xcen = cen[:, 0]
ycen = cen[:, 1]
plt.scatter(xcen, ycen, c="k", marker="d", s=80)
# c = colour, s = size

plt.xlabel("Urban(1990)")
plt.ylabel("Urban(2019)")
plt.title("3clusters")
plt.show()

# Scaling back to the original data and creating a plot it on the original scale

plt.style.use('seaborn')
plt.figure(dpi=300)

# now using the original dataframe
plt.scatter(urban_s["1990"], urban_s["2019"], c=labels, cmap="tab10")


# rescale and show cluster centres
scen = ct.backscale(cen, urban_min, urban_max)
xc = scen[:, 0]
yc = scen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=80)

plt.xlabel("1990")
plt.ylabel("2020")
plt.title("Urban population (% of total population)")
plt.show()

"""
Curve fitting solution
"""
# calling in the urban population data
urban
# Transposing the data
urban = urban.T

# Making the country name the colums
print(urban)

# Making the country name the colums
urban.columns = urban.iloc[0]

# Selecting only the years
urban = urban.iloc[1:]


# resetting the index
#urban.reset_index(drop=True, inplace=True)

urban.reset_index(inplace=True)

urban.rename(columns={'index': 'Year'}, inplace=True)

#renaming the country name to years
urban.rename(columns={'Country Name': 'Year'}, inplace=True)

urban=urban.apply(pd.to_numeric) # converting to numeric
print(urban.dtypes) # checking the types

# Fitting for China

def exponential(t, a, b):
    """Computes exponential growth of urban population
    
    Parameters:
        t: The current time
        a: The initial population
        b: The growth rate
        
    Returns:
        The population at the given time
    """
    f = a * np.exp(b * t)
    return f

def err_range(x, func, param, sigma):
    """Calculates the error range for a given function and its parameters
    
    Parameters:
        x: The input value for the function
        func: The function for which the error ranges will be calculated
        param: The parameters for the function
        sigma: The standard deviation of the data
        
    Returns:
        The lower and upper error ranges
    """
    lower = func(x, *param)
    upper = lower
    
    for i, p in enumerate(param):
        pmin = p - sigma[i]
        pmax = p + sigma[i]
        y = func(x, *param[:i], pmin, *param[i+1:])
        lower = np.minimum(lower, y)
        y = func(x, *param[:i], pmax, *param[i+1:])
        upper = np.maximum(upper, y)
        
    return lower, upper

years = urban['Year'].values
population = urban['China'].values

# Provide initial guess for exponential function
initial_guess = [min(population), 0.01]  # You can adjust the initial guess if needed

popt, pcov = curve_fit(exponential, years, population, p0=initial_guess, maxfev=5000)  # Increase maxfev

prediction_2030 = exponential(2030, *popt)
prediction_2040 = exponential(2040, *popt)

print("Urban population prediction for 2030:", prediction_2030)
print("Urban population prediction for 2040:", prediction_2040)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
predicted_years = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended = np.concatenate((curve_years, predicted_years))

curve_population = exponential(curve_years_extended, *popt)

# Calculate error range
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_range(curve_years_extended, exponential, popt, sigma)

# Plot the data, fitted curve, and error range
plt.figure(dpi=300)
plt.plot(years, population, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended, lower, upper, color='yellow', alpha=0.3, label='Error Range')
plt.xlabel('Year')
plt.ylabel('Urban Population')
plt.title('Exponential Growth Fit for Urban Population in China')
#plt.axvline(x=2030, color='g', linestyle='--', label='Projection Year')
plt.legend()
plt.grid(True)
plt.show()


# Fitting for France

# Polynomial function
def polynomial(t, *coefficients):
    """Computes a polynomial function
    
    Parameters:
        t: The time
        coefficients: Coefficients of the polynomial
        
    Returns:
        The population at the given time
    """
    return np.polyval(coefficients, t)

# Data
years = urban['Year'].values
population = urban['France'].values

# Degree of the polynomial
degree = 3  # Adjust the degree as needed

# Fitting the polynomial model
coefficients = np.polyfit(years, population, degree)

# Predictions for 2030 and 2040
prediction_2030 = polynomial(2030, *coefficients)
prediction_2040 = polynomial(2040, *coefficients)

print("Urban population prediction for 2030:", prediction_2030)
print("Urban population prediction for 2040:", prediction_2040)

# Generating points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
predicted_years = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended = np.concatenate((curve_years, predicted_years))

curve_population = polynomial(curve_years_extended, *coefficients)

# Error range
residuals = population - polynomial(years, *coefficients)
sigma = np.std(residuals)
lower = curve_population - sigma
upper = curve_population + sigma

# Plotting the data, fitted curve, and error range
plt.figure(dpi=300)
plt.plot(years, population, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended, lower, upper, color='yellow', alpha=0.9, label='Error Range')
plt.xlabel('Year')
plt.ylabel('Urban Population')
plt.title('Polynomial Fit for Urban Population in France')
#plt.axvline(x=2030, color='g', linestyle='--', label='Projection Year')
plt.legend()
plt.grid(True)
plt.show()


# Fitting for Kenya
# Data
years = urban['Year'].values
population = urban['Kenya'].values

# Degree of the polynomial
degree = 3  # Adjust the degree as needed

# Fitting the polynomial model
coefficients = np.polyfit(years, population, degree)

# Predictions for 2030 and 2040
prediction_2030 = polynomial(2030, *coefficients)
prediction_2040 = polynomial(2040, *coefficients)

print("Urban population prediction for 2030:", prediction_2030)
print("Urban population prediction for 2040:", prediction_2040)

# Generating points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
predicted_years = np.array([2030, 2040])  # Add the predicted years here
curve_years_extended = np.concatenate((curve_years, predicted_years))

curve_population = polynomial(curve_years_extended, *coefficients)

# Error range
residuals = population - polynomial(years, *coefficients)
sigma = np.std(residuals)
lower = curve_population - sigma
upper = curve_population + sigma

# Plotting the data, fitted curve, and error range
plt.figure(dpi=300)
plt.plot(years, population, 'ro', label='Data')
plt.plot(curve_years_extended, curve_population, 'b-', label='Fitted Curve')
plt.fill_between(curve_years_extended, lower, upper, color='yellow', alpha=1, label='Error Range')
plt.xlabel('Year')
plt.ylabel('Urban Population')
plt.title('Polynomial Fit for Urban Population in Kenya')
#plt.axvline(x=2030, color='g', linestyle='--', label='Projection Year')
plt.legend()
plt.grid(True)
plt.show()
