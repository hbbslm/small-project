import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

sns.set(style='dark')

# Hide PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# read data
data = pd.read_csv('D:/MBKM/analysis_data_project/data/day.csv')

# dashboard title
st.title("Bike Sharing Data Analysis")

# dashboard header
st.write("**This is a dashboard for analyzing bike sharing data.**")

# daily user 2011
st.subheader("Daily user in 2011")
data_2011 = data.loc[data['yr'] == 0]
plt.figure(figsize=(12, 6))
data_2011.plot(x='dteday', y=['casual', 'registered'], kind='line')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('Daily Count')
st.pyplot()

# daily user 2012
st.subheader("Daily user in 2012")
data_2012 = data.loc[data['yr'] == 1]
plt.figure(figsize=(12, 6))
data_2012.plot(x='dteday', y=['casual', 'registered'], kind='line')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('Daily Count')
st.pyplot()

st.markdown("Whether in 2011 or 2012, registered users used more bicycles than casual users, although there has been a few significant declines.")

# the most popular day in spring among regular users.
st.subheader("The most popular day in spring among regular users.")
data_day_spring = data.loc[data['season'] == 1]
day_grouped = data_day_spring.groupby('weekday')
casual_by_weekday = day_grouped['casual'].sum()

plt.figure(figsize=(12, 6))
casual_by_weekday.plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.xlabel('Day')
plt.ylabel('Amount')
plt.title('Casual Users by Weekday in Spring')
plt.grid(axis='y', linestyle='--', alpha=0.5) 
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot()

# the most popular day in winter among registered users
st.subheader("The most popular day in winter among registered users")
data_day_winter = data.loc[data['season'] == 4]
day_grouped = data_day_winter.groupby('weekday')
casual_by_weekday = day_grouped['registered'].sum()

casual_by_weekday.plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.xlabel('Day')
plt.ylabel('Amount')
plt.title('Registered Users by Weekday in Winter')
plt.grid(axis='y', linestyle='--', alpha=0.5) 
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
st.pyplot()

# cluster methode with k = 3
st.subheader("K-Means Clustering")
X = data[['cnt', 'temp', 'season']]
n_clusters = 3
kmeans = KMeans(n_clusters = n_clusters)
kmeans.fit(X)
labels = kmeans.labels_
data['cluster'] = labels

# define colors for each cluster
colors = ['red', 'green', 'blue']

plt.scatter(data['temp'], data['cnt'], c=[colors[i] for i in labels])
plt.xlabel('Temperature')
plt.ylabel('Count of Total Rental Bikes')
plt.title('K-Means Clustering')
st.pyplot()
st.markdown("There are 3 cluster formed by using column temperature, count of total rental bikes, and season.")