#!/usr/bin/env python3

import io
import os
import pickle
import sys
import pathlib

import numpy as np
import sklearn.preprocessing
import pandas as pd
import tensorflow as tf
import transformers

import shared

synsemclass_classifier_nn = shared.load_module_from_file("synsemclass_classifier_nn", 
    str(pathlib.Path(__file__).parent.absolute()) + "/../SynSemClassML/synsemclass_classifier_nn.py")

class Predictor:
    def __init__(self, model_path):
        le, tokenizer, model = self.load_model(model_path)
        self.le = le
        self.tokenizer = tokenizer
        self.model = model

    def load_model(self, model_path):
        # Load model parameters
        with open("{}/args.pickle".format(model_path), "rb") as pickle_file:
            model_training_args = pickle.load(pickle_file)

        # Read target (synsemclass_id) strings to integers encoder/decoder.
        with open("{}/classes.pickle".format(model_path), "rb") as pickle_file:
            le = pickle.load(pickle_file)

        # Load the tokenizer
        print("Loading tokenizer {}".format(model_training_args.bert), file=sys.stderr, flush=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_training_args.bert)

        # Instantiate and compile the model
        model = synsemclass_classifier_nn.SynSemClassClassifierNN(multilabel=model_training_args.multilabel)
        model.compile(len(le.classes_), model_training_args, training_batches=0)
        model.load_checkpoint(model_path)

        return (le, tokenizer, model)

    def create_tf_dataset(self, sentences):
        BATCH_SIZE = 25

        # Create TF dataset as input to NN
        inputs = self.tokenizer(sentences)["input_ids"] # drop masks
        inputs = tf.ragged.constant(inputs)

        tf_dataset = tf.data.Dataset.from_tensor_slices((inputs))

        tf_dataset = tf_dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size=BATCH_SIZE))
        
        return tf_dataset

    def predict(self, sentences):
        THRESHOLD = None
        NBEST = 1
    
        # Predict classes on development data
        predicted_classes = self.model.predict(self.create_tf_dataset(sentences), threshold=THRESHOLD, nbest=NBEST)
        predicted_classes = self.le.inverse_transform(predicted_classes)

        return predicted_classes
    
    def predict_proba_raw(self, sentences):
        return self.model.predict_values(self.create_tf_dataset(sentences))

    def predict_proba(self, sentences, nbest):
        probs = self.predict_proba_raw(sentences)
        rtn = []
        for prob in probs:
            best = np.argsort(prob)[::-1][:nbest]
            classes = self.le.inverse_transform(best)
            rtn.append(list(zip(classes, prob[best])))
        return rtn
    
if __name__ == "__main__":
    def main():
        pred = Predictor(sys.argv[1])
        print("Loaded successfully")
        while True:
            inp = input()
            print(pred.predict_proba([inp], 5)[0])
            
    main()
