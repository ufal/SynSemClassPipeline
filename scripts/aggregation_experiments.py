#!/usr/bin/env python3

import argparse
import re

import lzma
import pickle

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-proposals-matrix", type=str, help="Path to the proposal file in matrix form.")
    parser.add_argument("--path-gold-corpus", type=str, help="Path to the corpus with gold annotations.")
    parser.add_argument("--path-gold-list", type=str, help="Path to the list of gold annotations.")
    parser.add_argument("--clean-data", type=str, help="Whether to remove any lemma not in gold.")
    parser.add_argument("--preset", type=str, help="Preset paths with experiments.")
    parser.add_argument("--class-list-path", type=str, help="Path to the list of classes.", required=True)
    return parser.parse_args()

def check_args(args): 
    if args.path_gold_corpus and args.path_gold_list:
        raise ValueError("Argument error: Gold path defined twice.")
    
    if args.path_proposals_matrix:
        raise ValueError("Argument error: Missing proposals.")
    
    if not (args.path_gold_corpus and args.path_gold_list):
        raise ValueError("Argument error: Missing gold data.")
    
    return True

def preset_name(preset):
    return {
        "ko-mini-raw": "Korean",
        "ko-mini-clean": "Korean",
        "ko-full": "Korean",
        "ces": "Czech",
        "deu": "German",
        "eng": "English",      
    }[preset]

def load_data(args):
    if args.preset:
        try:
            return {
                "ko-mini-raw":    lambda: (load_proposals("ssc_data/kor_mini_matrix.lzma.pickle"), load_gold_xml("data/corpus.xml"), False),
                "ko-mini-clean":  lambda: (load_proposals("ssc_data/kor_mini_matrix.lzma.pickle"), load_gold_xml("data/corpus.xml"), True),
                "ko-full":        lambda: (load_proposals("ssc_data/kor_full_matrix.lzma.pickle"), load_gold_xml("data/corpus.xml"), True),
                "ces":            lambda: (load_proposals("ssc_data/ces_matrix.lzma.pickle")     , load_gold_list("ssc_data/gold_ssc_ces.txt"), True),
                "deu":            lambda: (load_proposals("ssc_data/deu_matrix.lzma.pickle")     , load_gold_list("ssc_data/gold_ssc_deu.txt"), True),
                "eng":            lambda: (load_proposals("ssc_data/eng_matrix.lzma.pickle")     , load_gold_list("ssc_data/gold_ssc_eng.txt"), True),
            }[args.preset]()
        except KeyError:
            raise ValueError(f"Argument error: Unknown preset: {args.preset}")

    check_args(args)
    return load_proposals(args.path_proposals_matrix), load_gold(args), args.clean_data

def read_all(path):
    with open(path, mode="r", encoding="utf-8") as content_file:
        return content_file.read()

def load_gold(args):
    if args.path_gold_corpus:
        return load_gold_xml(args.path_gold_corpus)
    
    if args.path_gold_list:
        return load_gold_list(args.path_gold_list)
    
def load_gold_xml(gold_path):
    def process_verb(verb):
        lemma = re.findall("lemma=\"(.+?)\"", verb)[0]
        vec = re.findall("class=\"(.+?)\"", verb)[0]
        return (lemma, vec)

    gold = read_all(gold_path)
    gold = gold.replace("\n", "")
    verbs = re.findall(r"<verb class=\".+?\" lemma=\".+?\">", gold)
    return set(process_verb(verb) for verb in verbs)

def load_gold_list(gold_path):
    return set(tuple(l.split("\t")) for l in read_all(gold_path).splitlines())

def load_proposals(path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)
    
def load_classes(args):
    with open(args.class_list_path, "rb") as f:
        return pickle.load(f).classes_

def main(args):
    proposals, gold, clean = load_data(args)
    if clean:
        gold_lemmas = set(lemma for lemma, _ in gold)
        proposals = {lemma: predictions for lemma, predictions in proposals.items() if lemma in gold_lemmas}
    
    class_list = load_classes(args)
    
    # aggregator = Agregator(class_list, ag_mean, pre_agg_top_k=5)
    # results = aggregator.make(proposals)
    # points = annotate_gold(results, gold)
    
    skip_labels = []

    #draw_precision_recall_curve(points, len(gold), skip_labels, color="#5833FA")

    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    for k in [1, 3, 5, 7, None]:
        draw_precision_recall_curve(Agregator(class_list, ag_mean, pre_agg_top_k=k).make_points(proposals, gold), len(gold), label=f"K{k if k else 'inf'}")

    save_graph("Pre-aggregation top-K", args.preset, "pre_k")

    for k in [1, 3, 5, 7, None]:
        draw_precision_recall_curve(Agregator(class_list, ag_mean, post_agg_top_k=k).make_points(proposals, gold), len(gold), label=f"K{k if k else 'inf'}")

    save_graph("Post-aggregation top-K", args.preset, "post_k")

    for th in [.01, .03, .08, .15, .5, None]:
        draw_precision_recall_curve(Agregator(class_list, ag_mean, pre_agg_threshold=th).make_points(proposals, gold), len(gold), label=(f"{th:.2f}" if th else "None"))

    save_graph("Pre-aggregation threshold", args.preset, "thr")

    for fn, label in [(ag_mean, "Average"), (ag_sum, "Sum"), (ag_max, "Max"), (ag_prob, "Probability")]:
        draw_precision_recall_curve(Agregator(class_list, fn).make_points(proposals, gold), len(gold), label=label)

    save_graph("Function", args.preset, "fnc")

    # betas = frange(0.1, 3.1, 0.1)
    # draw_f_beta_curve(points, len(gold), betas, skip_labels, color="#E93279")
    # # plt.xlim((-0.05, 3.2))
    # # plt.ylim((1,102))
    
    # plt.xlabel("Beta in F_beta")
    # plt.ylabel("Optimal threshold")
    # plt.savefig("f_beta_thresholds.png")
    # plt.savefig("f_beta_thresholds.svg")
    # plt.cla()

def save_graph(title, preset, exp_name):
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.title(f"{preset_name(preset)}")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"images/exp/{preset}_{exp_name}.png")
    plt.savefig(f"images/exp/{preset}_{exp_name}.svg")
    plt.cla()

class Agregator:
    def __init__(self, class_list, agg_fn, pre_agg_top_k=None, post_agg_top_k=None, pre_agg_threshold=None):
        self.class_list = class_list
        self.agg_fn = agg_fn
        self.pre_agg_top_k = pre_agg_top_k
        self.post_agg_top_k = post_agg_top_k
        self.pre_agg_threshold = pre_agg_threshold

    def make_points(self, proposals, gold):
        return annotate_gold(self.make(proposals), gold)

    def make(self, proposals):
        results = []
        for lemma, preds in proposals.items():
            results.extend(self.make_single(lemma, preds))
        return results

    def make_single(self, lemma, preds):
        preds = np.ma.array(preds, mask=np.zeros_like(preds))
        self.apply_pre_agg_top_k(preds)
        self.apply_pre_agg_threshold(preds)
        results = self.agg_fn(preds)
        self.apply_post_agg_top_k(results)
        return [((lemma, self.class_list[i]), results[i]) for i in np.where(results.mask == False)[0]]

    def apply_pre_agg_top_k(self, preds):
        if self.pre_agg_top_k is None:
            return
        
        part = np.argpartition(preds, -self.pre_agg_top_k)[:, :-self.pre_agg_top_k]
        
        row_indices = np.arange(preds.shape[0])[:, None]
        preds.mask[row_indices, part] = True
    
    def apply_pre_agg_threshold(self, preds):
        if self.pre_agg_threshold is None:
            return

        preds.mask[preds < self.pre_agg_threshold] = True 

    def apply_post_agg_top_k(self, results):
        if self.post_agg_top_k is None:
            return
        
        part = np.ma.argsort(results, endwith=False)[:-self.post_agg_top_k]
        results.mask[part] = True

def ag_mean(preds):
    return preds.mean(axis=0)

def ag_prob(preds):
    return 1 - np.prod(1 - preds, axis=0)

def ag_sum(preds):
    return preds.sum(axis=0)

def ag_max(preds):
    return preds.max(axis=0)

def ag_score(preds):
    preds = np.ma.copy(preds)
    preds[preds >= .7] = 100
    preds[np.logical_and(preds < .7, preds >= .4)] = 10
    preds[preds < .4] = 1
    return preds.sum(axis=0)

@dataclass
class DataPoint:
    correct: bool
    score: float

def annotate_gold(results, gold):
    return [DataPoint(data in gold, score) for data, score in results]


def frange(start, stop, step):
    m = min(start, stop, step)
    return [m * r for r in range(int(start / m), int(stop / m), int(step / m))]

def draw_precision_recall_curve(data, total_good, annotate=False, skip_labels=[], **kwargs):
    thresholds, recalls, precisions = make_precision_recall_curve(data, total_good)
    thresholds = [0] + thresholds
    new_thresholds = [thresholds[0]]
    new_recalls = [recalls[0]]
    new_precisions = [precisions[0]]

    for i in range(1, len(recalls)):
        if new_recalls[-1] <= recalls[i]:
            new_thresholds.pop()
            new_recalls.pop()
            new_precisions.pop()

            new_thresholds.append(thresholds[i])
            new_recalls.append(recalls[i])
            new_precisions.append(precisions[i])

        if new_precisions[-1] < precisions[i]:
            new_thresholds.append(thresholds[i])
            new_recalls.append(recalls[i])
            new_precisions.append(precisions[i])

    plt.plot(new_recalls, new_precisions, "-", **kwargs)
    if annotate:
        points = list(zip(new_thresholds, new_recalls, new_precisions))
        for i in range(len(points)):
            # Annotate point only if the current point is above the line connecting its neighbors
            if i == 0 or i == len(points) - 1 or is_above_curve(points[i - 1][1:], points[i + 1][1:], points[i][1:], 0.01):
                th, r, p = points[i]
                if in_flist(th, skip_labels, 0.01):
                    continue

                plt.annotate(f"{th:.2f}", (r + 0.01, p + 0.01))

def in_flist(elem, lst, eps):
    return bool([1 for f in lst if abs(elem - f) < eps])

def is_above_curve(curve_1, curve_2, point, th):
    slope =  (curve_1[1] - curve_2[1]) / (curve_1[0] - curve_2[0])
    b = curve_1[1] - slope * curve_1[0]
    return point[1] > slope * point[0] + b + th

def make_precision_recall_curve(data, total_good):
    data.sort(key=lambda p: p.score)
    values = sorted(set(d.score for d in data))
    thresholds = [(a + b) / 2 for a, b in zip(values, values[1:])]
    
    good_predictions = sum(1 for p in data if p.correct)
    all_predictions = len(data)
    
    recalls = [good_predictions / total_good]
    precisions = [good_predictions / all_predictions]

    i = 0
    for threshold in thresholds:
        while data[i].score < threshold:
            good_predictions -= 1 if data[i].correct else 0
            all_predictions -= 1
            i += 1 
        recalls.append(good_predictions / total_good)
        precisions.append(good_predictions / all_predictions)

    return thresholds, recalls, precisions

def f_beta_curve(data, total_good, rng):
    thresholds, recalls, precisions = make_precision_recall_curve(data, total_good)
    thresholds = [0] + thresholds

    best_thresholds = []

    for beta in rng:
        best = -1
        best_thr = None

        for th, recall, prec in zip(thresholds, recalls, precisions):
            fb = f_beta(beta, prec, recall)
            if fb > best:
                best, best_thr = fb, th

        best_thresholds.append(best_thr)

    return best_thresholds

def f_beta(beta, prec, recall):
    return (1 + beta**2) * prec * recall / (beta ** 2 * prec + recall)

def draw_f_beta_curve(data, total_good, betas, skip_labels=[], **kwargs):
    thresholds = f_beta_curve(data, total_good, betas)
    plt.plot(betas, thresholds, **kwargs)
    prev_t = None
    for b, th in reversed(list(zip(betas, thresholds))[:-1]):
        if th != prev_t and not in_flist(th, skip_labels, 0.01):
            plt.annotate(f"{th:.2f}", (b + 0.05, th + 0.02))
            prev_t = th

if __name__ == "__main__":
    main(get_args())