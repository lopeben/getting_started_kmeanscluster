import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn import preprocessing
from statistics import mean, stdev
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Columns meanings in the csv files
REFERENCE = 3
AIRSPEED  = 5
AIRTEMPER = 7
CTA_VOLTS = 9


# CSV files
labels    = ['5cmbhi', '5cmbme', '5cmblo', '4cmbhi', '4cmbme', '4cmblo', '3cmbhi', '3cmbme', '3cmblo', '2cmbhi', '2cmbme', '2cmblo', '1cmbhi', '1cmbme', '1cmblo', '0cmbhi', '0cmbme', '0cmblo', 'xxOOxx']
filenames = ['2023-11-30_d7_5cm_block_hii.csv', '2023-11-30_d7_5cm_block_med.csv', '2023-11-30_d7_5cm_block_low.csv', '2023-11-30_d7_4cm_block_hii.csv', '2023-11-30_d7_4cm_block_med.csv', '2023-11-30_d7_4cm_block_low.csv', '2023-11-24_d7_3cm_block_hii.csv', '2023-11-24_d7_3cm_block_med.csv', '2023-11-24_d7_3cm_block_low.csv', '2023-11-24_d7_2cm_block_hii.csv', '2023-11-24_d7_2cm_block_med.csv', '2023-11-24_d7_2cm_block_low.csv', '2023-11-22_d7_1cm_block_hii.csv', '2023-11-22_d7_1cm_block_med.csv', '2023-11-22_d7_1cm_block_low.csv', '2023-11-22_d7_hii.csv', '2023-11-22_d7_med.csv', '2023-11-22_d7_low.csv', '2023-11-22_d7_off.csv']

data = []
for filename in filenames:
    with open(filename) as csvfile:
        reader = pd.read_csv(csvfile, sep=':')
        reader.columns = reader.columns.str.strip()
        data.append(reader)

## Airflow Data
flow5cmbhi = data[labels.index('5cmbhi')].iloc[:, AIRSPEED].values
flow5cmbme = data[labels.index('5cmbme')].iloc[:, AIRSPEED].values
flow5cmblo = data[labels.index('5cmblo')].iloc[:, AIRSPEED].values
flow4cmbhi = data[labels.index('4cmbhi')].iloc[:, AIRSPEED].values
flow4cmbme = data[labels.index('4cmbme')].iloc[:, AIRSPEED].values
flow4cmblo = data[labels.index('4cmblo')].iloc[:, AIRSPEED].values
flow3cmbhi = data[labels.index('3cmbhi')].iloc[:, AIRSPEED].values
flow3cmbme = data[labels.index('3cmbme')].iloc[:, AIRSPEED].values
flow3cmblo = data[labels.index('3cmblo')].iloc[:, AIRSPEED].values
flow2cmbhi = data[labels.index('2cmbhi')].iloc[:, AIRSPEED].values
flow2cmbme = data[labels.index('2cmbme')].iloc[:, AIRSPEED].values
flow2cmblo = data[labels.index('2cmblo')].iloc[:, AIRSPEED].values
flow1cmbhi = data[labels.index('1cmbhi')].iloc[:, AIRSPEED].values
flow1cmbme = data[labels.index('1cmbme')].iloc[:, AIRSPEED].values
flow1cmblo = data[labels.index('1cmblo')].iloc[:, AIRSPEED].values
flow0cmbhi = data[labels.index('0cmbhi')].iloc[:, AIRSPEED].values
flow0cmbme = data[labels.index('0cmbme')].iloc[:, AIRSPEED].values
flow0cmblo = data[labels.index('0cmblo')].iloc[:, AIRSPEED].values
flowxxOOxx = data[labels.index('xxOOxx')].iloc[:, AIRSPEED].values


## Temperature Data
temp5cmbhi = data[labels.index('5cmbhi')].iloc[:, AIRTEMPER].values
temp5cmbme = data[labels.index('5cmbme')].iloc[:, AIRTEMPER].values
temp5cmblo = data[labels.index('5cmblo')].iloc[:, AIRTEMPER].values
temp4cmbhi = data[labels.index('4cmbhi')].iloc[:, AIRTEMPER].values
temp4cmbme = data[labels.index('4cmbme')].iloc[:, AIRTEMPER].values
temp4cmblo = data[labels.index('4cmblo')].iloc[:, AIRTEMPER].values
temp3cmbhi = data[labels.index('3cmbhi')].iloc[:, AIRTEMPER].values
temp3cmbme = data[labels.index('3cmbme')].iloc[:, AIRTEMPER].values
temp3cmblo = data[labels.index('3cmblo')].iloc[:, AIRTEMPER].values
temp2cmbhi = data[labels.index('2cmbhi')].iloc[:, AIRTEMPER].values
temp2cmbme = data[labels.index('2cmbme')].iloc[:, AIRTEMPER].values
temp2cmblo = data[labels.index('2cmblo')].iloc[:, AIRTEMPER].values
temp1cmbhi = data[labels.index('1cmbhi')].iloc[:, AIRTEMPER].values
temp1cmbme = data[labels.index('1cmbme')].iloc[:, AIRTEMPER].values
temp1cmblo = data[labels.index('1cmblo')].iloc[:, AIRTEMPER].values
temp0cmbhi = data[labels.index('0cmbhi')].iloc[:, AIRTEMPER].values
temp0cmbme = data[labels.index('0cmbme')].iloc[:, AIRTEMPER].values
temp0cmblo = data[labels.index('0cmblo')].iloc[:, AIRTEMPER].values
tempxxOOxx = data[labels.index('xxOOxx')].iloc[:, AIRTEMPER].values

## Consolidated Airflow Data
airflow = np.concatenate((flow5cmbhi, flow5cmbme, flow5cmblo, 
                          flow4cmbhi, flow4cmbme, flow4cmblo, 
                          flow3cmbhi, flow3cmbme, flow3cmblo, 
                          flow2cmbhi, flow2cmbme, flow2cmblo, 
                          flow1cmbhi, flow1cmbme, flow1cmblo, 
                          flow0cmbhi, flow0cmbme, flow0cmblo, 
                          flowxxOOxx, flowxxOOxx, flowxxOOxx))
 
# print(type(airflow))
# print(airflow.dtype)
# airflow_arr = np.array(airflow_txt, dtype=int)
# print(type(airflow_arr))
# print(airflow_arr.dtype)

# print(airflow)
# print(len(airflow))

normalized_airflow = preprocessing.normalize([airflow])
print("Normalized shape: ", normalized_airflow.shape)

#reshaped_airflow = normalized_airflow.reshape(11097, 7)
reshaped_airflow = normalized_airflow.reshape(77679, 1) #single column array
print("Reshaped: ", reshaped_airflow.shape)


scaler = StandardScaler()
scaled_features = scaler.fit_transform(reshaped_airflow)

# print("Is contiguous?", normalized_airflow.flags['C_CONTIGUOUS'])

# print(type(normalized_airflow))
# print(normalized_airflow.shape)
# print(normalized_airflow.dtype)
# print(max(normalized_airflow))
# print(min(normalized_airflow))

kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=1)

dat = np.ascontiguousarray(reshaped_airflow)
#dat = np.ascontiguousarray(scaled_features)
print("Array ", dat.shape)
print(dat.dtype)

result = kmeans.fit(dat)
print("Fit results ", result)

result = kmeans.inertia_ 
print("Inertia ", result)

result = kmeans.cluster_centers_
print("Cluster centers\n", result)

result = kmeans.n_iter_
print("Iterations ", result)

result = kmeans.labels_[:3]
print("Labels ", result)








kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(reshaped_airflow)
    sse.append(kmeans.inertia_)


# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 20), sse)
# plt.xticks(range(1, 20))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")



kl = KneeLocator(
    range(1, 20), sse, curve="convex", direction="decreasing"
)

print("Elbow: ", kl.elbow)




# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(reshaped_airflow)
    score = silhouette_score(reshaped_airflow, kmeans.labels_)
    silhouette_coefficients.append(score)


plt.style.use("fivethirtyeight")
plt.plot(range(2, 20), silhouette_coefficients)
plt.xticks(range(2, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")








plt.show()

## Data series
# airspeed_high = data[labels.index('high')].iloc[:, AIRSPEED].values
# airtemper_high = data[labels.index('high')].iloc[:, AIRTEMPER].values

# airspeed_medium = data[labels.index('medium')].iloc[:, AIRSPEED].values
# airtemper_medium = data[labels.index('medium')].iloc[:, AIRTEMPER].values

# airspeed_low = data[labels.index('low')].iloc[:, AIRSPEED].values
# airtemper_low = data[labels.index('low')].iloc[:, AIRTEMPER].values


# ## 2D Scatter plot
# # plt.scatter(time_high, airtemper_high, color = 'magenta')
# # plt.scatter(time_high, airspeed_high, color = 'yellow')
# # plt.scatter(airtemper_high, airspeed_high, color = 'green')

# fig, axs = plt.subplots(3, sharex=False, sharey=False, tight_layout=True)

# time      =      time_medium
# airspeed  =  airspeed_medium
# airtemper = airtemper_medium


# axs[0].set_title("ScatterPlot ")
# axs[0].scatter(time, airtemper, color = 'orchid')
# axs[0].set_ylabel('Temperature (C)')
# axs[0].set_xlabel('Time (s)')

# axs[1].scatter(time, airspeed, color = 'steelblue')
# axs[1].set_ylabel('Wind Speed (m/s)')
# axs[1].set_xlabel('Time (s)')

# axs[2].scatter(airtemper, airspeed, color = 'green')
# axs[2].set_ylabel('Wind Speed (m/s)')
# axs[2].set_xlabel('Temperature (C)')

# bfig, baxs = plt.subplots(1, sharex=False, sharey=False, tight_layout=True)
# baxs.boxplot( [airspeed_high, airspeed_medium, airspeed_low], labels=['high', 'medium', 'low'] )


# def linear_model(x, m, c):
    # return (x*m + c)


# popt, pcov = curve_fit(linear_model, airtemper, airspeed)
# # print parameter values
# m, c = popt
# print(f'Slope = {m:.3f}, Intercept = {c:.3f}')
# print(pcov)

# # # Create a new array of x-values that includes zero
# x_new = np.linspace(round(min(airtemper))-1, round(max(airtemper))+1, len(airtemper))
# # # # Use the predict method to generate the corresponding y-values
# y_new = [ m*x_new[i] + c for i in range(len(airtemper))]

# axs[2].plot(x_new, y_new, color = 'red')


# print(f'Mean speed - high: {mean(airspeed_high)}')
# print(f'Std dev speed - high: {stdev(airspeed_high)}')
# print(f'Mean speed - medium: {mean(airspeed_medium)}')
# print(f'Std dev speed - medium: {stdev(airspeed_medium)}')
# print(f'Mean speed - low: {mean(airspeed_low)}')
# print(f'Std dev speed - low: {stdev(airspeed_low)}')




# hfig, haxs = plt.subplots(3, sharex=True, sharey=True, tight_layout=True)

# sns.histplot(ax=haxs[0], data=airspeed_low, bins=32, kde=True, color='skyblue', edgecolor='red')
# #haxs[0].hist(airspeed_low, bins=32, color='skyblue', edgecolor='black')
# haxs[0].set_xlabel('Lowflow Values')
# haxs[0].set_ylabel('Frequency')

# sns.histplot(ax=haxs[1], data=airspeed_medium, bins=32, kde=True, color='lightgreen', edgecolor='red')
# #haxs[1].hist(airspeed_medium, bins=32, color='lightgreen', edgecolor='black')
# haxs[1].set_xlabel('Mediumflow Values')
# haxs[1].set_ylabel('Frequency')

# sns.histplot(ax=haxs[2], data=airspeed_high, bins=32, kde=True, color='aquamarine', edgecolor='red')
# #haxs[2].hist(airspeed_high, bins=32, color='aquamarine', edgecolor='black')
# haxs[2].set_xlabel('Highflow Values')
# haxs[2].set_ylabel('Frequency')



# plt.show()

