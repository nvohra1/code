def isVowel(letter):
#    if(letter in ['a','e','i','o','u']):
    if (letter in ["aeiouy"]):
        return(True)
    else:
        return(False)

vo = isVowel('a')
print(vo)
vo = isVowel('A')
print(vo)
vo = isVowel(4)
print(vo)

#(letter in ['a','e','i','o','u','y','A','E','I','O','U','Y'])


import string

alphabet = string.ascii_letters
print(alphabet)

#sentence = 'Jim quickly realized that the beautiful gowns are expensive'
def counter(sentence):
    count_letters = {}  # empty dictionary
    for letter in sentence:  # iterates through every letter in sentence
        if letter in alphabet:  # checks whether letter in in alphabet
            if letter in count_letters:  # if letter is in the alphabet, it further checks whether it is already in the dictionary
                count_letters[letter] += 1  # If so, it's count value is incremented by 1
            else:
                count_letters[letter] = 1  # if the letter is not in the dictionary, its value is set to 1

    print(count_letters)
    return count_letters

address = """Four score and seven years ago our fathers brought forth on this continent, a new nation, 
conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a 
great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. 
We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final 
resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper 
that we should do this. But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- 
this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add 
or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. 
It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so 
nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored 
dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here 
highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of 
freedom -- and that government of the people, by the people, for the people, shall not perish from the earth."""

counter(address)

from math import pi
print(pi/4)

import random
random.seed(1)  # This line fixes the value called by your function,
# and is used for answer-checking.
def rand():
    # define `rand` here!
    return random.uniform(-1, 1)
x=rand()
print(x)

import math
def distance(x, y):
    # define your function here!
    return math.sqrt((y[1] - x[1]) ** 2 + (y[0] - x[0]) ** 2)
print(distance((0, 0), (1, 1)))

# -*- coding: utf-8 -*-
# Created on 2017-08-18 14:17
# @author Sergii Panchenko
# The ratio of the areas of a circle and the square inscribing it is pi / 4.
# In this six-part exercise, we will find a way to approximate this value.
import random, math
def rand():
    return random.uniform(-1, 1)
def distance(x, y):
    return math.hypot(x[0] - y[0], x[1] - y[1])
def in_circle(x, origin=[0] * 2):
    return distance(x, origin) < 1
R = 100000
x = []
inside = []
for i in range(R):
    point = [rand(), rand()]
    x.append(point)
    inside.append(in_circle(point))
print(abs(inside.count(True) / R - math.pi / 4))

print(abs(inside.count(True) / R ))
cal = abs(inside.count(True) / R )
pi4 = pi/4
dif = cal-pi4
print(cal)
print(pi4)
print(dif)



def moving_window_average(x, n_neighbors=1):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[-1]]*n_neighbors
    # To complete the function,
    # return a list of the mean of values from i to i+width for all values i from 0 to n-1.
    return [sum(x[i: i + width]) / width for i in range(n)]

x = [0,10,5,3,1,5]
print(moving_window_average(x, 1))
print(sum(moving_window_average(x, 1)))

# -*- coding: utf-8 -*-
# Created on 2017-08-18 23:09
# @author Sergii Panchenko
# Compute and store R=1000 random values from 0-1 as x.
# moving_window_average(x, n_neighbors) is pre-loaded into memory from 3a.
# Compute the moving window average for x for values of n_neighbors ranging from 1 to 9 inclusive.
# Store x as well as each of these averages as consecutive lists in a list called Y.
import random
random.seed(1)  # This line fixes the value called by your function,
# and is used for answer-checking.
# write your code here!
Y = []
R = 1000
x = [random.uniform(0, 1) for i in range(R)]
Y.append(x)
for i in range(1, 10):
    Y.append(moving_window_average(x, i))
    if(i==5):
        print(sum(moving_window_average(x, i)))
print(Y)


R = 1000
x = [random.uniform(0, 1) for i in range(1000)]
#print(x)
Y = [x] + [moving_window_average(x, n_neighbors) for n_neighbors in range(1, 10)]
#print(Y)


import sys
def update(n,x,y):
    func_name = sys._getframe().f_code.co_name
    print(str(func_name),'n =', n, " id(n):", id(n), '; list x:',x," id(x):", id(x), '; tupple y:',y," id(y):", id(y) )
    n=9 # New memory location is created.
    x.append(4) #Reference to caller's variable will be used
    #x=[0,1,2,3,4] # New memory location is created.
    y=(10,11,22,33,34,35) # New memory location is created.
    print(str(func_name),'n =', n, " id(n):", id(n), '; list x:',x," id(x):", id(x), '; tupple y:',y," id(y):", id(y) )

def caller():
    func_name = sys._getframe().f_code.co_name
    n=1
    x=[0,1,2,3]
    y=(10,11,22,33)
    print(str(func_name),'n =', n, " id(n):", id(n), '; list x:',x," id(x):", id(x), '; tupple y:',y," id(y):", id(y) )
    update(n,x,y)
    #p=[4,5,6,7,8]
    #x = [5,4, 5, 6, 7, 8] # New memory location is created.
    print(str(func_name),'n =', n, " id(n):", id(n), '; list x:',x," id(x):", id(x), '; tupple y:',y," id(y):", id(y) )

caller()
# caller n = 1  id(n): 140708038297248 ; list x: [0, 1, 2, 3]  id(x): 2559296949696 ; tupple y: (10, 11, 22, 33)  id(y): 2559274193040
# update n = 1  id(n): 140708038297248 ; list x: [0, 1, 2, 3]  id(x): 2559296949696 ; tupple y: (10, 11, 22, 33)  id(y): 2559274193040
# update n = 9  id(n): 140708038297504 ; list x: [0, 1, 2, 3, 4]  id(x): 2559296949696 ; tupple y: (10, 11, 22, 33, 34, 35)  id(y): 2559274099520
# caller n = 1  id(n): 140708038297248 ; list x: [0, 1, 2, 3, 4]  id(x): 2559296949696 ; tupple y: (10, 11, 22, 33)  id(y): 2559274193040

print("############## numpy #########")
import numpy as np
zero_vector = np.zeros(5)
zero_matrix = np.zeros((2,3))
one_matrix = np.ones((3,4))
np_array = np.array([[1,2,3],[11,22,33]])
transpose_array = np_array.transpose()

print(zero_vector)
print(zero_matrix)
print(one_matrix)
print(np_array)
print(transpose_array)


###
a = np.array([1,2])
b = np.array([3,4,5])
c = b[1:]
x = b[a] is c
print(x)

z1 = np.array([1,3,5,7,9])
w = z1[0:3]
w[0] = 3
print(id(w[0]))
print(id(z1[0]))
xx= z1[0:3] is w
print(xx, '  ',id(z1[0:3]), '   ',id(w)) # False
print(w)
print(z1)

############
z1 = np.array([1,3,5,7,9])
ind = np.array([0,1,2])
w = z1[ind]
print(w, '  ', id(w))
w[0] = 3
print(w, '  ', id(w))
print(z1, '  ', id(z1))
#################
z1 = np.array([1,3,5,7,9])
z2 = np.array([2,4,6,8,10])
ind = z1>6
print(ind)
print(z1[ind])
print(z2[ind])

########################
z1 = np.linspace(0,100,10)
print(z1)
z2 = np.logspace(1,2,10) # 1=log(10), 2 = log(100), 10 elements with log spacing
print(z2)
z3 = np.logspace(np.log10(250), np.log10(500), 10) # 1=log(250), 2 = log(500), 10 elements with log spacing
print(z3)
print('###')
X = np.array([[1,2,3],[4,5,6]])
s = X.shape
size = X.size
print(s)
print(size)


x = np.random.random(10)
x9 = np.any(x > 0.9)
x1 = np.all(x >= 0.1)
print(x)
print(x9)
print(x1)

print('###')
x = 13
t = not np.any([x%i == 0 for i in range(2,x)])
print(t)

print('##################### Pyplot ################################')
import matplotlib.pyplot as plt
#m = plt.plot([0,1,4,9,16])
# legend()
# axis()
# xlabel()
# ylabel()
# savefig()
#plt.show() # Todisplay the graph
# x = np.linspace(0,10,20)  --- 1 Regular Plot
x = np.logspace(-1,1,40) # log to the base 10 --- # 2 log log plot
y1 = x**2.0
y2 = x**1.5
print(x)
print(y1)
print(y2)

# 1 - Regular Plot
# plt.plot(x, y1, 'rd-', linewidth=2, markersize=5, label="First")
# plt.plot(x, y2, 'gs-', linewidth=2, markersize=5, label="Second")

# 2 - log log plot
# plt.loglog(x, y1, 'rd-', linewidth=2, markersize=5, label="First")
# plt.loglog(x, y2, 'gs-', linewidth=2, markersize=5, label="Second")
#
# plt.xlabel("$X$")
# plt.ylabel("$Y$")
# #plt.axis([xmin,xmax,ymin,ymax])
# plt.axis([-0.5,10.5,-5,105])
# plt.legend(loc="upper left")
# plt.savefig("myplot.pdf")
# plt.show()
#################################################################################
######### Regular Plot
x = np.linspace(0,10,20)
y1 = x**2.0
y2 = x**1.5
print(x)
print(y1)
print(y2)
plt.subplot(331) # 4= rows, 3-columns, 1=figure number
plt.plot(x, y1, 'rd-', linewidth=2, markersize=5, label="First")
plt.plot(x, y2, 'gs-', linewidth=2, markersize=5, label="Second")
plt.xlabel("$X$")
plt.ylabel("$Y$")
#plt.axis([xmin,xmax,ymin,ymax])
plt.axis([-0.5,10.5,-5,105])
plt.legend(loc="upper left")

####################log log plot ##################
x = np.logspace(-1,1,40)
y1 = x**2.0
y2 = x**1.5
print(x)
print(y1)
print(y2)
plt.subplot(332)
plt.loglog(x, y1, 'rd-', linewidth=2, markersize=5, label="First")
plt.loglog(x, y2, 'gs-', linewidth=2, markersize=5, label="Second")
plt.xlabel("$X$")
plt.ylabel("$Y$")
#plt.axis([xmin,xmax,ymin,ymax])
plt.axis([-0.5,10.5,-5,105])
plt.legend(loc="upper left")

#################### Histogram plot - Normal distribution ##################
print("##### Histogram ########")
x = np.random.normal(size=1000) # normal Distribution
plt.subplot(333)
plt.hist(x, normed=True, bins=np.linspace(-5,5,21)) # histogram, default bin = 10, normed = True change number of observation to proportion of observation, 20 bins betwen -5 & 5
# plt.show()

print("##### Gamma distribution in Histogram ########")
#subplot() # 2X3 mean two rows & 3 columns
x = np.random.gamma(2,3,100000) # gamma Distribution with 100000 points
plt.subplot(334)
plt.hist(x, bins=30)
plt.subplot(335)
plt.hist(x, normed=True, bins=30)
plt.subplot(336)
plt.hist(x, bins=30, cumulative = 30)
plt.subplot(337)
#plt.subplot(4,3,7)
plt.hist(x, normed=True, bins=30, cumulative = True, histtype="step")
# plt.show()
##############################################
############ Rendomness ###########################

import random
x = random.choice(["H","T"])
y = random.choice([0,1])
dice = random.choice([1,2,3,4,5,6]) # use list
dice2 = random.choice(range(1,7)) # use range
op = random.choice(random.choice([range(1,7), range(1,9), range(1,11)])) # there are 3 dice 1-6 faces, 2-8 faces, 3-110 faces, Randomly pick one dice and role it to get an output number
print(op)

rolls = []
for k in range(1000):
    rolls.append(random.choice(range(1,7)))
#print(rolls)
plt.subplot(338)
plt.hist(rolls)
#plt.show()
plt.cla()
plt.close()

############# Central Limit Theorm lead to normal distribution --- sum of large number of random variable done large number of times will get close to normal distribution#########
ys = []
for rep in range(100000):
    y = sum(random.choice(range(1, 7)) for i in range(10))
    ys.append(y)

plt.hist(ys,normed=True,bins=61)
#plt.show()
plt.close()


#######################



#Standard uniform distribution for number between 0 & 1
print("####")
x = np.random.random()
X1 = np.random.random(5)
X2 = np.random.random((5,4))
print("x: ",x)
print("X1: ",X1)
print("X2: ",X2)

# Normal distribution for number between 0 & 1
print("####")
x= np.random.normal(0,1) # mean = 0, std.dev. = 1
X1= np.random.normal(0,1,5) # mean = 0, std.dev. = 1, Array of 5 numbers
X2= np.random.normal(0,1,(4,5)) # mean = 0, std.dev. = 1, matrix of $X5 numbers

print("x: ",x)
print("X1: ",X1)
print("X2: ",X2)

###############################
print("####")
X = np.random.randint(1,7,(2,3,4)) # Create a 3-DMatrix of 2X3X4
print(X)
Y = np.sum(X,axis=1) # Creates a 2-D Matrix of 2X4
print(Y)

X = np.random.randint(1,7,(1000000,10))
Y = np.sum(X,axis=1)
plt.hist(Y, bins=101)
#plt.show()
plt.close()


z=np.sum(np.random.randint(1,7,(100,10)), axis=0)
print("z: ",z)

########### Measuring Time ####################
import time

print("       ")
print("PurePython implementation")
#start_time = time.clock() # not working
start_time = time.time()
ys = []
for rep in range(100000):
    y=0
    for k in range(10):
        x = random.choice([1,2,3,4,5,6])
        y=y+x
    ys.append(y)
end_time=time.time()
elapse_time = end_time-start_time
print(elapse_time, " sec")


print("numpy implementation")
start_time=time.time()

X=np.random.randint(1,7,(100000, 10))
Y=np.sum(X,axis=1)

end_time=time.time()
elapse_time = end_time-start_time
print(elapse_time, " sec")

############## Random Walk ##############
# Start point , Step size,  location at time t
# x(t=0) = X0
# x(t=1) = x0 + delta(t=1) = x1
# x(t=2) = x1 + delta(t=2) = X2

# x(t=k) = x0 + delta(t=1) + delta(t=2).... delta(t=k)


delta_X = np.random.normal(0,1,(2,5)) # mean, std dev, 2X5 matrix
#plt.plot(delta_X[0],delta_X[1], "go")
# plt.show()

X = np.cumsum(delta_X, axis = 1)
print(delta_X)
print(X)

# plt.plot(X[0], X[1])
# plt.show()

# initial locaiton is 0,0
X_0 = np.array([[0],[0]])
Xn = np.concatenate((X_0, X), axis = 1)
print(Xn)

plt.plot(Xn[0], Xn[1], "ro-")
plt.show()
plt.close()

###############
np.zeros()