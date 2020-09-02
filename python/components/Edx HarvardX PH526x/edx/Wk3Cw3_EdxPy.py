import numpy as np
import random

def distance(p1, p2):
    """Find the distance between p1 & p2"""
    return np.sqrt(np.sum(np.power(p2 -p1, 2)))

p1 = np.array([1, 1])
p2 = np.array([4, 4])

z = distance(p2, p1)
print(z)


def majority_vote(votes):
    """ """
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    # print (vote_counts)
    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)
    # print(winners)
    return random.choice(winners)


import scipy.stats as ss
def majority_vote_short(votes):
    """ Return the most common element in votes"""
    mode, count = ss.mstats.mode(votes)
    return mode



votes = [1,2,3,2,2,3,3,2,1,2,3,1,2,1,3,3]
print(votes)

winner = majority_vote(votes)
print(winner)
winner = majority_vote_short(votes)
print(winner, winner[0])


dict = {1: 'hindi', 2: 'sanskrit', 3: 'english'}
print(dict.items())

#########

def find_nearest_neighbors(p, points, k):
    """Find the k nearest neighbors of point p and return their indices."""
    distances = np.zeros(points.shape[0])
    # print(distance)
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    # print(distances)
    # print (ind)

    return ind[:k]

def knn_predict(p, points, outomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])


outcomes = np.array([0,0,0,0,1,1,1,1,1])
points = np.array([[1,1], [1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])



p = np.array([2.5,2.7])
p2 = np.array([1 ,1.7])
x1 = knn_predict(p, points, outcomes, 3)
print(x1)
x2 = knn_predict(p2, points, outcomes, 3)
print(x2)

import matplotlib.pyplot as plt
print(points[:,0], points[:,1])
plt.plot(points[:,0], points[:,1], "ro")
plt.plot(p[0], p[1], "bo")
plt.plot(p2[0], p2[1], "go")
plt.axis([0.5,3.5,0.5,3.5])
# plt.show()
print("point shape: ", points.shape)


###############
def generate_synth_data(n=50):
    """Create two sets of points from bivariante normal distribution.
    mean= 0, std = 1 create n rows & 2 columns      - dataset 1
    mean= 1, std = 1 create n rows & 2 columns      - dataset 2
    concatenate along axis 0 i.e. rows

    For out come vector
    repeate 0 n times
    repeate 1 n times
    """
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))),axis=0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return (points,outcomes)

n=20
(points,outcomes) = generate_synth_data(n)
print(points)
print(outcomes)

plt.figure()
plt.plot(points[:n,0], points[:n,1], "ro")
plt.plot(points[n:,0], points[n:,1], "bo")
plt.savefig("bivardata.pdf")

x = ss.norm(0,1).rvs((n,2))
print(type(x))

##################################################


def make_pridiction_grid(predictors, outcomes, limits, h, k):
    """Classify each point on the pridiction grid.

    """
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array(([x,y]))
            prediction_grid[j,i] = knn_predict(p,predictors,outcomes,k)
    return(xx,yy, prediction_grid)

#####
seasons = ["spring","summer", "fasll", "winter"]
e = enumerate(seasons)
list_e = list(e)
print(e)
print(list_e)
for ind,season in enumerate(seasons):
    print(ind, season)

# 0.5 = min, 10=max, 0,75=step size for alist
xs = np.arange(0.5, 10, 0.75)
print(xs)
print(type(xs))

####################
def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

#Belos is the valid code commented it just to save time for running other code line.
# (predictors, outcomes) = generate_synth_data()
#
# k=5; filename="knn_synth_5.pdf"; limits=(-3,4,-3,4); h=0.1
# (xx,yy,prediction_grid) = make_pridiction_grid(predictors, outcomes, limits, h, k)
# plot_prediction_grid(xx,yy, prediction_grid, filename)
#
#
# k=50; filename="knn_synth_50.pdf"; limits=(-3,4,-3,4); h=0.1
# (xx,yy,prediction_grid) = make_pridiction_grid(predictors, outcomes, limits, h, k)
# plot_prediction_grid(xx,yy, prediction_grid, filename)


#############################
print("############ SKLearn ###########")
from sklearn import datasets

# It consists of 150 different iris flowers.
# 50 from each of three different species.
# For each flower, we have the following covariates: sepal length, sepal width,
# petal length, and petal width.
iris = datasets.load_iris()



print (type(iris))
print(iris)
predictors = iris.data[:,0:2]
outcomes = iris.target
print (type(predictors))
print(predictors.shape)
print(predictors)
print (type(outcomes))
print(outcomes.shape)
print(outcomes)

# all ppridictors where the outcome is 0 -- all rows from column 0 (x)
# all ppridictors where the outcome is 0 -- all rows from column 1 (y)
plt.plot(predictors[outcomes==0][:,0], predictors[outcomes==0][:,1], "ro")
plt.plot(predictors[outcomes==1][:,0], predictors[outcomes==1][:,1], "go")
plt.plot(predictors[outcomes==2][:,0], predictors[outcomes==2][:,1], "bo")
plt.savefig("iris.pdf")

k=5; filename="iris_grid.pdf"; limits=(4,8,1.5,4.5); h=0.1
(xx,yy,prediction_grid) = make_pridiction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx,yy, prediction_grid, filename)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)

print(sk_predictions.shape)
print(sk_predictions[0:10])

my_predictions = np.array([knn_predict(p, predictors,outcomes, 5)for p in predictors])
print(100*np.mean(sk_predictions == my_predictions),"% match sk_predictions == my_predictions")

print(100*np.mean(sk_predictions == outcomes),"% match sk_predictions == outcomes")
print(100*np.mean(my_predictions == outcomes),"% match my_predictions == outcomes")
