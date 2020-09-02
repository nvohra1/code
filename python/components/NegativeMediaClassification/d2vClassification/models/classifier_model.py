from .abc_model import AbcModel
from .d2v_model import d2vModel

import logging
import os
import inspect

import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import sys

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
base_path = os.path.dirname(os.path.abspath(base_file_path))
project_dir_path = os.path.dirname(os.path.abspath(base_path))
classifiers_path = os.path.join(project_dir_path, 'classifiers')

print(base_file_path)

class classifierModel(AbcModel):
    def __init__(self):
        super().__init__()

    def init_model(self):
        func_name = sys._getframe().f_code.co_name
        logging.info("Classifier :: " + str(func_name))
        #self.model = LogisticRegression()
        #self.model = DecisionTreeClassifier()
        self.model = RandomForestClassifier()

    def train_model(self, d2v, training_vectors, training_labels):
        func_name = sys._getframe().f_code.co_name
        logging.info("Classifier :: " + str(func_name))

        # logging.info("Classifier training ::   training_vector : " + str(len(training_vectors)) + '   & training_labels  : ' + str(len(training_labels)))
        # doc2Vac vectors of train data
        train_vectors = d2vModel.get_vectors(
            d2v, len(training_vectors), 300, 'Train')

        ##############################
        self.model.fit(train_vectors, np.array(training_labels))
        training_predictions = self.model.predict(train_vectors)
        logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
        logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
        logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions,average='weighted')))

    def test_model(self, d2v, testing_vectors, testing_labels):
        func_name = sys._getframe().f_code.co_name
        logging.info("Classifier :: " + str(func_name))
        #logging.info("Classifier testing ::   testing_vector : " + str(len(testing_vectors)) + '   & testing_labels  : ' + str(len(testing_labels)))
        test_vectors = d2vModel.get_vectors(
            d2v, len(testing_vectors), 300, 'Test')
        testing_predictions = self.model.predict(test_vectors)
        logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
        logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
        logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions,average='weighted')))

    def predict(self, d2v, testing_vectors):
        func_name = sys._getframe().f_code.co_name
        logging.info ("Classifier :: " + str(func_name))
        test_vectors = d2vModel.new_vectors(
            d2v, testing_vectors, len(testing_vectors), 300, 'Test')
        testing_predictions = self.model.predict(test_vectors)
        logging.info(testing_predictions)

