import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt


#Prepare data for training
trainDataNegative = ['It is learnt from unknown sources that Logan is planning to file for Bankruptcy.',
        'There was a raid at a Chicago location where a close partner of Logan was found meeting with drug cartel.',
        'Logan had been evaluating bankruptcy option for the last 12 months and he is expected to make a decision this week.',
        'After one year of struggle bankruptcy is the only option for Emma.',
        'Drug addiction is becoming a big issue for Emma and her behavior at shareholders meeting exposed her drug issue.,'
        'Through investigation by IRS concluded that Emma has been doing tax fraud for last ten years.']
trainDataPositive = ['Since the last 15 months Logan is flying internationally on a vacation with his family.',
        'NFL fan club invited Logan as a spokesperson to their annual member meeting in Chicago.',
        'Emma will be on vacation for next 2 months.',
        'For some reason Emma is always in the news.']
data = trainDataNegative + trainDataPositive
print(data)