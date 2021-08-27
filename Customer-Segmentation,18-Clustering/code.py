# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# Load Offers
offers=pd.read_excel(path, sheet_name=0)

# Load Transactions
transactions=pd.read_excel(path,sheet_name=1)

# Merge dataframes
transactions['n'] =1

# Look at the first 5 rows
df = pd.merge(offers, transactions, on='Offer #')

df.head()




# --------------
# Code starts here

# create pivot table
matrix=pd.pivot_table(df,index='Customer Last Name',columns='Offer #',values='n')

# replace missing values with 0
matrix.fillna(0,inplace=True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
print(matrix.head())

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster=KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10,random_state=0)

# create 'cluster' column

matrix['cluster'] = cluster.fit_predict(matrix.iloc[:, 1:])

print(matrix['cluster'].head())
# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca=PCA(n_components=2, random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:, 0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:, 1]


# dataframe to visualize clusters by customer names
clusters = matrix.iloc[:, 33:36]
clusters['Customer Last Name'] = matrix.iloc[:, 0]
print(clusters.head())

# visualize clusters
plt.scatter(x=clusters.x, y=clusters.y, cmap='viridis')
plt.show()
# Code ends here


# --------------
# Code starts here



# merge 'clusters' and 'transactions'
data = pd.merge(transactions, clusters, on='Customer Last Name')

# merge `data` and `offers`
data = pd.merge(offers,data)
# initialzie empty dictionary
champagne = {}

# iterate over every cluster
for cluster in range(1,5):
    # observation falls in that cluster
    new_df = data[data['cluster'] == cluster] 
    # sort cluster according to type of 'Varietal'
    counts = new_df['Varietal'].value_counts(ascending=False)
    # check if 'Champagne' is ordered mostly
    if counts.index[0] == "Champagne":
        # add it to 'champagne'
        champagne[cluster] = counts[0]

# get cluster with maximum orders of 'Champagne' 
import operator
cluster_champagne = max(champagne.items(), key=operator.itemgetter(1))[0]

# print out cluster number
print(cluster_champagne)



# --------------
# Code starts here

# empty dictionary
discount = {}

# iterate over cluster numbers
for cluster in range(0, 5):
    # dataframe for every cluster
    new_df = data[data['cluster'] == cluster]
    # average discount for cluster
    counts = new_df['Discount (%)'].sum()/len(new_df['Discount (%)'])
    # adding cluster number as key and average discount as value 
    discount[cluster] = counts

# cluster with maximum average discount
cluster_discount  = max(discount.items(), key=operator.itemgetter(1))[0]
print(cluster_discount)

# Code ends here



