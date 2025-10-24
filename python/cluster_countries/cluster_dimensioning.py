import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn
seaborn.set_theme()

raw_data = pd.read_csv('Countries-exercise.csv')
print(raw_data)

x = raw_data.loc[:,['Longitude','Latitude']]
print(x)

wcss=[]
keep_going = 1
while keep_going > 0:
    print(f"KMeans({keep_going})")
    kmeans = KMeans(keep_going)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    if len(wcss)>1 and (wcss[len(wcss)-2]/wcss[len(wcss)-1] < 1.25):
        keep_going = 0
    else:
        keep_going += 1

print(wcss)
number_of_clusters=range(1,len(wcss)+1)

f,axs = plt.subplots(1,4, figsize=(20,4))
axs[3].plot(number_of_clusters,wcss)
for i in range(len(wcss)-3,len(wcss)):
    kmeans = KMeans(i)
    kmeans.fit(x)
    clusters = kmeans.fit_predict(x)
    data_w_clusters = raw_data.copy()
    data_w_clusters['Cluster']=clusters
    axs[i - len(wcss)+3].set_title(str(i))
    axs[i - len(wcss)+3].scatter(data_w_clusters['Longitude'],data_w_clusters['Latitude'],c=data_w_clusters['Cluster'],cmap='rainbow',alpha=0.4)
    #plt.scatter(data_w_clusters['Longitude'],data_w_clusters['Latitude'],c=data_w_clusters['Cluster'],cmap='rainbow')

plt.show()


