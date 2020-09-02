import numpy as np, random, scipy.stats as ss

def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode

def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]


import pandas as pd
data = pd.read_csv("wine.csv")
print(type(data))
print(data)
# print(data.head(10))
data["is_red"] = data.apply(lambda row: 1 if (row.color == "red") else 0 , axis = 1)
#  1 is for column
numeric_data = data.drop('color',1)

red_count = sum(numeric_data["is_red"])
print(numeric_data)
print(red_count)

import sklearn.preprocessing
import numpy as np
# nnumeric_data = np.array(numeric_data)
numeric_data = (numeric_data - np.mean(numeric_data, axis=0)) / np.std(numeric_data, ddof=0 )
# scaled_data = (numeric_data.values - numeric_data.mean(axis = 1))/numeric_data.std(axis = 1)
columns = numeric_data.columns
print(numeric_data)
print(columns)

# mean = np.mean(numeric_data, axis=0)
# print(mean)
# std = np.std(numeric_data, ddof=0 )
# print(std)
# std = np.std(numeric_data, axis=0, ddof=0 )
# print(std)

import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=2)
print(type(pca))
print(pca)
principal_components = pca.fit(numeric_data).transform(numeric_data)
print(type(principal_components))
print(principal_components)
print(principal_components.shape)

##################################################


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2, c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
# plt.show()

####################################

def accuracy(predictions, outcomes):
    return (100*np.mean(predictions == outcomes))

x = np.array([1,2,3])
y = np.array([1,2,4])
print (accuracy(x, y))


import numpy as np
np.random.seed(1) # do not change this!

x = np.random.randint(0, 2, 1000)
y = np.random.randint(0 ,2, 1000)
print (accuracy(x, y))

#############################
print (accuracy(0, data["high_quality"]))

##################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
# Enter your code here!
library_predictions = knn.predict(numeric_data)

print (accuracy (library_predictions, data["high_quality"] ))

###################

n_rows = data.shape[0]
random.seed(123)
selection = random.sample(range(n_rows), 10)
print(selection)

########################
predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])

#my_predictions = # Enter your code here!
my_predictions = np.array([knn_predict(p, predictors[training_indices,:], outcomes, k=5) for p in predictors[selection]])
#percentage = # Enter your code here!
percentage = accuracy(my_predictions, data.high_quality[selection])
print (percentage)