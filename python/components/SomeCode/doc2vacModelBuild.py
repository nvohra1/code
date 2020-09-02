#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

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
# It is learnt from unknown sources that Logan is planning to file for Bankruptcy.
# Since the last 15 months Logan is flying internationally on a vacation with his family.
# There was a raid at a Chicago location where a close partner of Logan was found meeting with drug cartel.
# NFL fan club invited Logan as a spokesperson to their annual member meeting in Chicago.
# This is just a regular news related to none of the person of interest that we are looking for.
# Logan had been evaluating bankruptcy option for the last 12 months and he is expected to make a decision this week.
#
# After one year of struggle bankruptcy is the only option for Emma.
# Emma will be on vacation for next 2 months.
# Drug addiction is becoming a big issue for Emma and her behavior at shareholders meeting exposed her drug issue.
# For some reason Emma is always in the news.
# Invention of new energy store is a long-awaited breakthrough that can shape up energy usage pattern of society.
# Through investigation by IRS concluded that Emma has been doing tax fraud for last ten years.

# Prepare data for training
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

# Training of the model
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")