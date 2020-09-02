import pandas as pd
# Pandas is a Python library that provides data structures and functions
# for working with structured data, primarily tabular data.
# Pandas is built on top of NumPy and some familiarity with NumPy
# makes Pandas easier to use and understand.


print("########## Series #############")
x = pd.Series([6,3,8,6])
print(type(x))
print(x)
print(x[1])

x = pd.Series([6,3,8,6], index=["q","w","e","r"])
print(type(x))
print(x)
print(x["w"], x["r"])
print(x.index)

xi = sorted(x.index)
x2 = x.reindex(sorted(x.index))
print(x2)

y = pd.Series([7,3,5,2], index=["e","q","r","t"])
print(y)
xplusy = x+y
print(xplusy)


age = {"Tim":29,"Jim":31,"Pam":27,"Sam":35}
x = pd.Series(age)
print(type(x))
print(x)

#############
print("########## DataFrame #############")
data = {'name': ['Tim','Jim','Pam','Sam'],
        'age': [29,31,27,35],
        'Zip': ['02115','02130','67700','00100']}

print(type(data))
print(data)
x = pd.DataFrame(data, columns = ["name","Zip","age"])
print(type(x))
print(x)
print(x["name"])
print(x.name)

########################################################################

import numpy as np
whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")


print(whisky)
# print(whisky.head)
# print(whisky.tail)

# first 10 rows
print(whisky.iloc[1:10,:])

# 5:10 rows & 0:5 columns
print(whisky.iloc[5:10,0:5])

#  : rows & 2:14 columns --- extract all flavor information for the dataframe into another dataframe
flavors = whisky.iloc[:,2:14]
print(flavors)

#############
## There are many different kinds of correlations,
# and by default, the function uses what is
# called Pearson correlation which estimates
# linear correlations in the data.


corr_flavors = pd.DataFrame.corr(flavors)
print(corr_flavors)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("corr_flavors.pdf")


corr_whisky = pd.DataFrame.corr(flavors.transpose())
print(corr_whisky)
plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
plt.axis("tight")
plt.colorbar()
plt.savefig("corr_whisky.pdf")


########################
# Next we're going to cluster whiskeys based on their flavor profiles.
# We'll do this using a clustering method from the scikit-learn machine learning
# module.
# The specific method we'll be using is called spectral co-clustering.
# One way to think about spectral co-clustering method
# is to consider a list of words and a list of documents,
# which is the context in which the method was first introduced.

# If you'd like to learn more about eigenvalues and eigenvectors,

# 6 Region
x = np.array(whisky["Region"])
print(np.unique(x))

from sklearn.cluster.bicluster import SpectralCoclustering
# 6 Region --- so the clusters are set to 6
x = np.array(whisky["Region"])
print(np.unique(x))

model = SpectralCoclustering(n_clusters = 6, random_state=0)
model.fit(corr_whisky)
# 6 clusters with rows as individual whisky & Treu mean whisky belong to this cluster.
print(model.rows_)
print(model)

print(np.sum(model.rows_,axis =1))
print(np.sum(model.rows_,axis =0))
print(model.row_labels_)

#######
# Let's draw the clusters as groups that we just
# discovered in our whisky DataFrame.
# Let's also rename the indices to match the sorting.

whisky['Groups'] = pd.Series(model.row_labels_,index=whisky.index)
print(whisky)
print(np.argsort(model.row_labels_))
whisky = whisky.ix[np.argsort(model.row_labels_)]
print(whisky)
whisky = whisky.reset_index(drop=True)
print(whisky)

correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)
print(correlations)

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")
plt.colorbar()
plt.savefig("correlations.pdf")

####################
import pandas as pd
data = pd.Series([1,2,3,4])
data = data.ix[[3,0,1,2]]
print(data[0])