import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn
seaborn.set_theme()

raw_data = pd.read_csv('Countries-exercise.xls')
print(raw_data)

plt.scatter(raw_data['Longitude'],raw_data['Latitude'], c=raw_data['Longitude']*raw_data['Longitude'],cmap='rainbow')
plt.show()

location = raw_data.loc[:,['Longitude', 'Latitude']]
kmeans = KMeans(8)
kmeans.fit(location)

clusters = kmeans.fit_predict(location)
print(clusters)
location['Cluster'] = clusters
print(location)
plt.scatter(location['Longitude'], location['Latitude'], c=location['Cluster'],cmap='rainbow',alpha=0.4)
plt.show()

centroids = pd.DataFrame(kmeans.cluster_centers_)
new_columns=['Longitude','Latitude']
centroids.columns=new_columns
centroids['Cluster'] = centroids.shape[0]
print(centroids)

location=pd.concat([location,centroids]).reset_index(drop=True)
plt.scatter(location['Longitude'], location['Latitude'], c=location['Cluster'],cmap='rainbow',alpha=0.4)
plt.show()
