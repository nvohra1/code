text = "This is my test text. We're keeping this text short to keep things managable."

def count_words(text):
    """ count each occurance of work"""

    text = text.lower()
    skips = [".", ",", ";", ":","'", '"']
    for ch in skips:
        text = text.replace(ch, "")

    word_counts = {}
    for word in text.split(" "):
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts

x1 = count_words(text)
print(x1)

#####################
from collections import Counter

def count_words_fast(text):
    """ count each occurance of work"""

    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"']
    for ch in skips:
        text = text.replace(ch, "")

    word_counts = Counter(text.split(" "))
    return word_counts

x2 = count_words_fast(text)
print(x2)

print(x1 == x2)
print(len(count_words("This comprehension check is to check for comprehension.")))

###3.2.3############################################
def read_book(title_path):
    """Read a book & retuen the string"""
    with open(title_path, "r", encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n","").replace("\r","")
    return text

title_path = "GutenbergBooks\Books_EngFr\English\shakespeare\Romeo and Juliet.txt"
text = read_book(title_path)
print(len(text))
ind = text.find("What's in a name?")
print(ind)
sample_text = text[ind : ind+1000]
print(sample_text)

################

def word_stats(word_counts):
    """Return number of unique words and word frequencies"""
    num_unique = len(word_counts)
    counts = word_counts.values()
    return(num_unique, counts)

title_path = "GutenbergBooks\Books_EngFr\English\shakespeare\Romeo and Juliet.txt"
text = read_book(title_path)
word_counts = count_words(text)
(num_unique, counts) = word_stats(word_counts)
print("English " , num_unique,sum(counts))

# both below will work
#title_path = "GutenbergBooks\Books_GerPort\German\shakespeare\Romeo und Julia.txt"
title_path = "GutenbergBooks/Books_GerPort/German/shakespeare/Romeo und Julia.txt"
text = read_book(title_path)
word_counts = count_words(text)
(num_unique, counts) = word_stats(word_counts)
print("German " , num_unique,sum(counts))

######3.2.5###################

import os
import pandas as pd
book_dir = "./GutenbergBooks/Books_EngFr"
#book_dir = "./GutenbergBooks/Books_GerPort"
stats = pd.DataFrame(columns= ("language", "author", "title", "length", "unique"))
title_num = 1

for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + "/" + language):
        for title in os.listdir(book_dir + "/" + language +"/" + author):
            inputfile = book_dir + "/" + language +"/" + author + "/" + title
            # print(inputfile)
            text = read_book(inputfile)
            # print(text)
            (num_unique, counts) = word_stats(count_words(text))
            stats.loc[title_num] = language, author.capitalize(), title.replace(".txt", ""), sum(counts), num_unique
            title_num += 1
# print(stats)
print(stats.head(1))
print(stats.tail(1))

# print(stats.length)
# print(stats.unique)
# print(stats["length"])


import matplotlib.pyplot as plt
#plt.plot(stats.length, stats.unique, "bo")
# plt.loglog(stats.length, stats.unique, "bo")
# plt.show()
#
# print(stats[stats.language == "English"])
# print(stats[stats.language == "French"])
plt.figure(figsize=(10,10))
subset = stats[stats.language == "English"]
plt.loglog(subset.length, subset.unique, "o", label = "English", color = "crimson")
subset = stats[stats.language == "French"]
plt.loglog(subset.length, subset.unique, "o", label = "French", color = "forestgreen")

plt.legend()
plt.xlabel("Book length")
plt.ylabel("number of unique words")

plt.savefig("lang_Plot.pdf")