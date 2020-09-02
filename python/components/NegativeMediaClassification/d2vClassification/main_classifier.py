from models.d2v_model import d2vModel
from models.classifier_model import classifierModel

import os
import logging
import inspect

import pandas as pd
from numpy import sort
from sklearn.model_selection import train_test_split
import sys

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
project_dir_path = os.path.dirname(os.path.abspath(base_file_path))
data_path = os.path.join(project_dir_path, 'data')

class MainClassifier():

    def __init__(self):
        super().__init__()
        self.d2v = d2vModel()
        self.classifier = classifierModel()
        self.dataset = None

    def read_file(self, filename):
        filename = os.path.join(data_path, filename)
        self.dataset = pd.read_csv(filename, header=0, delimiter="\t")

    def init_data(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset.text, self.dataset.type, random_state=0,
            test_size=0.1)

        # Labelled list of data set
        x_train = d2vModel.label_text(x_train, 'Train')
        x_test = d2vModel.label_text(x_test, 'Test')
        all_data = x_train + x_test
        return x_train, x_test, y_train, y_test, all_data

    def init_test_data(self, sentence):
        x_test = d2vModel.label_text(sentence, 'Test')
        return x_test

    def run_classifier(self):
        # Prepare Data
        x_train, x_test, y_train, y_test, all_data = self.init_data()

        # Create d2v of dataset
        self.d2v.init_model(all_data)
        self.d2v.train_model()

        # Prepare Classifier & run the model
        self.classifier.init_model()
        self.classifier.train_model(self.d2v, x_train, y_train)
        self.classifier.test_model(self.d2v, x_test, y_test)
        return self.d2v, self.classifier

    def predict_via_classifier(self):
        self.read_file("newdataset.csv")
        x_new_str = self.dataset.text
        logging.info(' Desired outcome : 1 0 0 0 1 1 for x_new_str ')
        logging.info( x_new_str)
        x_new = self.init_test_data(x_new_str)
        self.classifier.predict(self.d2v, x_new)

def run(dataset_file):
    mc = MainClassifier()
    mc.read_file(dataset_file)
    mc.run_classifier()
    mc.predict_via_classifier()

if __name__ == "__main__":
    run("dataset.csv")

