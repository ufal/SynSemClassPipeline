#!/usr/bin/env python3

import sys
import csv

def read_data(data_path):
    with open(data_path, "r") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        data = list(csv_reader)
        class_pos = header.index("synsemclass_id")
        sentece_pos = header.index("sentence")
        data = [(line[sentece_pos], line[class_pos]) for line in data]
        return data

def run_experiment(model, data_path):
    data = read_data(data_path)
    pred = model.predict([X for X, y in data])
    return 100 * sum(1 if y == p else 0 for (X, y), p in zip(data, pred)) / len(data), len(data)

class DummyModel:
    def predict(self, x):
        return ["vec00002"] * len(x)

def main(models_path, data_path):
    print("Loading ML libraries", file=sys.stderr)
    import synsemclass_prediction
    print("Successfully loaded ML libraries", file=sys.stderr)

    cde_model = synsemclass_prediction.Predictor(models_path + "/cde")
    all_deu, deu_size = run_experiment(cde_model, data_path + '/examples_test_deu.csv')
    print(f"cde on deu: {all_deu}")
    all_eng, eng_size = run_experiment(cde_model, data_path + '/examples_test_eng.csv')
    print(f"cde on eng: {all_eng}")
    all_ces, ces_size = run_experiment(cde_model, data_path + '/examples_test_ces.csv')
    print(f"cde on ces: {all_ces}")
    weight_average = (all_ces * ces_size + all_deu * deu_size + all_eng * eng_size) / (ces_size + deu_size + eng_size)
    del cde_model

    cd_model = synsemclass_prediction.Predictor(models_path + "/cd")
    cd_eng, _ = run_experiment(cd_model, data_path + '/examples_test_eng.csv')
    print(f"cd on eng: {cd_eng}")
    del cd_model

    ce_model = synsemclass_prediction.Predictor(models_path + "/ce")
    ce_deu, _ = run_experiment(ce_model, data_path + '/examples_test_deu.csv')
    print(f"ce on deu: {ce_deu}")
    del ce_model

    de_model = synsemclass_prediction.Predictor(models_path + "/de")
    de_ces, _ = run_experiment(de_model, data_path + '/examples_test_ces.csv')
    print(f"de on ces: {de_ces}")
    del de_model

    print( "Multilingual Multilabel Model Results (ces, deu, eng)")
    print( "=====================================================")
    print()
    print( "Test data accuracy")
    print( "------------------")
    print(f"Multilingual (ces, deu, eng) test data accuracy: {weight_average:.2f}")
    print(f"ces test data accuracy: {all_ces:.2f}")
    print(f"deu test data accuracy: {all_deu:.2f}")
    print(f"eng test data accuracy: {all_eng:.2f}")
    print()
    print( "Zero-shot experiments")
    print( "=====================")
    print()
    print( "Model trained on ces, deu; tested on eng:")
    print( "-----------------------------------------")
    print(f"all classes: {cd_eng:.2f}")
    print( "Model trained on ces, eng; tested on deu:")
    print( "-----------------------------------------")
    print(f"all classes: {ce_deu:.2f}")
    print( "Model trained on deu, eng; tested on ces:")
    print( "-----------------------------------------")
    print(f"all classes: {de_ces:.2f}")

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, help="Path to models.")
    parser.add_argument("--data", type=str, help="Path to data.")
    return parser.parse_args()

def _main(args):
    main(args.models, args.data)

if __name__ == "__main__":
    _main(get_args())