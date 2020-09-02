import os
import pandas as pd
import numpy as np
from collections import Counter
def count_words_fast(text):
    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"', "\n", "!", "?", "(", ")"]
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))
    return word_counts
def read_book(title_path):
    text   = pd.read_csv(title_path, sep = "\n", engine='python', encoding="utf8")
    text = text.to_string(index = False)
    return text
def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)


#title_path="Wk3CS2_HW\hamlets.csv"
title_path="GutenbergBooks\Books_EngFr\English\shakespeare\Hamlet.txt"
# title_path="GutenbergBooks\Books_GerPort\German\shakespeare\Hamlet.txt"


text = read_book(title_path)
# print(text)
counted_text = count_words_fast(text)
print(counted_text)

data = pd.DataFrame({"word": list(counted_text.keys()), "count": list(counted_text.values())})



print(counted_text['hamlet'])
#
# title_num = 1
# data = pd.DataFrame(columns= ("word", "count"))
# for itr in counted_text:
#     data.loc[title_num] = itr, counted_text[itr]
#     title_num += 1
#     # print(itr)
print(data)
subset = data[data.word == "hamlet"]
print(subset)
