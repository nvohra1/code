from .abc_model import AbcModel

import logging
import random
import os
import inspect

import numpy as np
from gensim.models import doc2vec
import sys

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
base_path = os.path.dirname(os.path.abspath(base_file_path))
project_dir_path = os.path.dirname(os.path.abspath(base_path))
classifiers_path = os.path.join(project_dir_path, 'classifiers')


class d2vModel(AbcModel):

    def __init__(self):
        super().__init__()

    def init_model(self, corpus):
        func_name = sys._getframe().f_code.co_name
        logging.info("d2vModel :: " + str(func_name))

        logging.info("Building Doc2Vec vocabulary")
        self.corpus = corpus
        self.model = doc2vec.Doc2Vec(min_count=1, # Ignores all words with total frequency lower than this
                                     window=10, # The maximum distance between the current and predicted word within a sentence
                                     vector_size=300,  # Dimensionality of the  generated feature vectors
                                     workers=5,  # Number of worker threads to  train the model
                                     alpha=0.025,  # The initial learning rate
                                     min_alpha=0.00025, # Learning rate will linearly drop to min_alpha as training progresses
                                     dm=1)
        self.model.build_vocab(self.corpus)

    def train_model(self):
        func_name = sys._getframe().f_code.co_name
        logging.info("d2vModel :: " + str(func_name))

        # for epoch in range(9):
        for epoch in range(9):
            logging.info('Training iteration #{0}'.format(epoch))
            self.model.train(
                self.corpus, total_examples=self.model.corpus_count,
                epochs=self.model.epochs)
            logging.info('Before shuffle -- Training iteration #{0}'.format(epoch))
            # shuffle the corpus
            random.shuffle(self.corpus)
            # decrease the learning rate
            self.model.alpha -= 0.0002
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha

    def get_vectors(self, corpus_size, vectors_size, vectors_type):
        func_name = sys._getframe().f_code.co_name
        logging.info("d2vModel :: " + str(func_name))
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = self.model.docvecs[prefix]
            # logging.info( "corpus_size: " + str(corpus_size) + '    vector_size: ' + str(vectors_size))
            # logging.info("prefix: " + prefix)
            # logging.info("vector: " + str(vectors[i]))
        return vectors

    def new_vectors(self, corpus, corpus_size, vectors_size, vectors_type):
        func_name = sys._getframe().f_code.co_name
        logging.info("d2vModel :: " + str(func_name))
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            # logging.info(str(corpus[i][0]))
            vectors[i] = self.model.infer_vector(corpus[i][0])
            # logging.info( "corpus_size: " + str(corpus_size) + '    vector_size: ' + str(vectors_size))
            # logging.info("vector: " + str(vectors[i]))
        return vectors

    def label_text(corpus, label_type):
        func_name = sys._getframe().f_code.co_name
        logging.info("d2vModel :: " + str(func_name))
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
        return labeled
