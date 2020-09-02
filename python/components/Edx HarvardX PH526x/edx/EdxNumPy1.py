# What happens when import statement executes
# Create the new namespace
# the module in this newly created name space.
# creates a name or aliase for the pkg like np for numpy
import numpy as np
import math
import random
from math import pi # import only module pi
import gensim
from nltk.tokenize import word_tokenize
import naiveBayesClassifier


x = np.array([1,3,5])
y = np.array([1,5,9])

print(x.mean())

print(x.shape)

##########################
print (' import math examples')
print (math.pi)
print(math.sqrt(10000))
print(math.sin(math.pi/2))

print(np.sqrt(1000))
print(np.sqrt([1000,4,64,400])) # MATH DON HAVE THIS FUNCTIONALITY



a = "A String...."
print(type(a)) # shows type of object
print(dir(a)) # shows list of functions associated with this object a which is of type str
print(dir(str)) # shows list of functions associated with this object type str
print(help(a.upper)) # provide usage of upper() method of str object

##########################
# One of the interesting things about Python integers is that they have unlimited precision.

# integer division
a = 15//7
print('integer division result : ' + str(a))

#float division
a = 15/7
print('float division result : ' + str(a))

#exponent
a = 5**13
print('exponent result : ' + str(a))

# _ underscore operator --- used in interactive mode
# the value of the underscore operator is always the last object that Python has returned to you.
# below in script dosnot work
#a = 15/2.3
#b = _ *300
#print(b)

# factorial
b = math.factorial(4)
print ('factoria : ' + str(b))

#In this list, I'm going to just enter a few numbers-- 2, 44, 55, and 66.
#And then when I run the random choice, Python returns one of these numbers back to me.
#Random works just the same way for any type of object.

print('Ramdom.choice')
a = random.choice([2,44,55,66])
print('random.choice : ' + str(a))
a = random.choice([2,"sdsd",55,66,"swretf"])
print('random.choice : ' + str(a))

#Video 1.1.6: Expressions and Booleans
#Objects of the boolean type have only two values. These are called True and False.
#There are only three boolean operations, which are "or", "and", and "not".
a = type(True)
print('boolean : ' + str(a))


# There are a total of eight different comparison operations in Python.
# Although these are commonly used for numeric types,   we can actually apply them to other types as well.# For example, if you're comparing two sequences,
# the comparison is carried out element-wise. So you're comparing the first element of your first sequence to your first element in your second sequence, and so on.
#There are a total of eight different comparison operations in Python.
# <= >= == !=
print(' < > <= >= == != is  is not ')
if([2,3] == [2,3]):
    print(True)
else:
    print(False)

if([2,3] is [2,3]):
    print(True)
else:
    print(False)

if([2,3] is not [2,3]):
    print(True)
else:
    print(False)

if(2.0 == 2):
    print(True)
else:
    print(False)

    #Video 1.2.1: Sequences
# there are three basic sequences,
# which are lists, tuples, and so-called "range objects".
#Python also has additional sequence types for representing things like strings.
#  sequences is that indexing starts at 0.
# Sequence object can be accessed from right to left & left to right
# left most (first) object is indexed as 0
# right most (last) object is indexed as -1
a=[1,2,3,4,5,6,7,8,9,0]
print(a[0])
print(a[2])
print(a[-1])
print(a[-3])
print(a[3:-3])  # start:to location
print(a[3:]) # from 3 index obj to last
print(a[:-3]) # from 0 index obj to -3 indexed object

a=(1,2,3)[-2] # return -2 indexed obj from tuple
print(a)

#Video 1.2.2: Lists : Lists are mutable sequences of objects of any type.
# And they're typically used to store homogeneous items.
# Lists are one type of sequence, just like strings but they do have their differences.
# strings are immutable, whereas lists are mutable.

alist = [1,2,3,4,5,6,7,8,9,0]
alist.append(1000)
print(alist)
blist = [66.88,999,33,44,55,66]
clist = alist + blist
print(clist)
clist.reverse()
dlist = sorted(clist)

print(dlist)
print(clist)

clist.sort()
print(clist)
clistLen = len(clist)
print(clistLen)


#######################1.2.3: Tuples
#Tuples are immutable sequences typically used to store heterogeneous data.
T = (1,3,5,7,3)
l = len(T)

T + ('a',2,'s')
print(type(T))
print(T[1])
print(T)

cnt=T.count(3)
print(cnt)
sm = sum(T)
print(sm)

#Packing of tuple
x=12.23
y=23.34
coordinate=(x,y)
coType = type(coordinate)
print(coordinate)
print(coType)

# Un-Packing of tuple
(c1,c2) = coordinate
print(c1)
print(c2)

# Creating tuple with single object
t1 = (2,)
print(t1)
print(type(t1))
print(type((5,)))


t2 = (2)
print(t2)
print(type(t2))

##############Reange
# Ranges are immutable sequences of integers,
# and they are commonly used in for loops.

# Default step up
x = range(0,7)
l = list(x)
print(x)
print(l)

# custom step up
y = range(1,17,3)
ly = list(y)
print(y)
print(ly)

##########1.2.5: Strings
#Strings are immutable sequences of characters.
#In Python, you can enclose strings in either single quotes, in quotation marks, or in triple quotes.

s = 'Pythonn'
l = len(s)
f = s[0]
la = s[-1]
subs = s[0:3]
subs2 = s[-3:]

print(s,f,la,subs,subs2)

t = 'y' in s
print(t)

ThreeTimes_s = 10*s
print(ThreeTimes_s)

k = ThreeTimes_s + str(1000)
print(k)

### Help: Use Following on python prompt
#   dir(str)
#   help(str.replace)

name = 'Ram Bhaiya'
newName = name.replace('y',"Y")
print(name, newName)

ss = name.split(' ')
print(ss)
print(type(ss))

fss=ss[0].upper()
print(fss)

nnn = "125,000"
print(nnn.isnumeric())
print(nnn.isdecimal())
nnn = "125000"
print(nnn.isnumeric())
print(nnn.isdecimal())

##############################
print("@@@@@@@@@@@@@@!")
if False:
    print("False!")
elif True:
    print("Now True!")
else:
    print("Finally True!")

n = 200
if n % 2 == 0:
    print("even")
else:
     print("odd")

###########
n=100
number_of_times = 0
while n >= 1:
   n //= 2
   print(n)
   number_of_times += 1
print(number_of_times)

###################
print('####Doc Similarity#############')

print(dir(gensim))
raw_documents = ["I'm taking the show on the road.",
                 "My socks are a force multiplier.",
             "I am the barber who cuts everyone's hair who doesn't cut their own.",
             "Legend has it that the mind is a mad monkey.",
            "I make my own fun."]
print("Number of documents:",len(raw_documents))

gen_docs = [[w.lower() for w in word_tokenize(text)]
            for text in raw_documents]
print(gen_docs)
dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary[5])
print(dictionary.token2id['road'])
print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)
tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
print(s)
sims = gensim.similarities.Similarity('/usr/workdir/',tf_idf[corpus],
                                      num_features=len(dictionary))
print(sims)
print(type(sims))
query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)
sims[query_doc_tf_idf]