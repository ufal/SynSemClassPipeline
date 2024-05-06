#!/usr/bin/env python3

from typing import List

import argparse
import re

from dataclasses import dataclass

import matplotlib.pyplot as plt

@dataclass
class Proposal:
    vec: str
    lemma: str
    score: float
    scores: List[float]
    correct: bool = None

def avg(scores):
    return sum(scores) / len(scores)

def prob(scores):
    prod = 1
    for s in scores:
        prod *= 1 - (s / 100)
    return 1 - prod 

def apply_score(proposals, scorer):
    for p in proposals:
        p.score = scorer(p.scores)

def main(args):
    proposals = find_proposals(args.path_proposals)
    gold = set(find_gold(args.path_minikorpus))
    for p in proposals:
        p.correct = (p.vec, p.lemma) in gold
    
    apply_score(proposals, avg)

    betas = frange(0.1, 3.1, 0.1)
    thresholds = f_beta_curve(proposals, len(gold), betas)
    plt.xlabel("Beta in F_beta")
    plt.ylabel("Optimal threshold")
    plt.xlim((-0.05, 3.2))
    plt.ylim((1,102))
    plt.plot(betas, thresholds, color="#E93279")
    prev_t = None
    annotated_thresholds = []
    for b, th in reversed(list(zip(betas, thresholds))[:-1]):
        if th != prev_t and th != 30.65:
            annotated_thresholds.append(th)
            plt.annotate(f"{th:.2f}", (b + 0.05, th + 2))
            prev_t = th
    plt.savefig("f_beta_thresholds.png")
    plt.savefig("f_beta_thresholds.svg")
    plt.cla()

    plt.xlim((0.185, 0.675))
    plt.ylim((0.175, 1.065))
    draw_precision_recall_curve(proposals, len(gold), annotated_thresholds)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("precision_recall.png")
    plt.savefig("precision_recall.svg")
    plt.cla()



def frange(start, stop, step):
    m = min(start, stop, step)
    return [m * r for r in range(int(start / m), int(stop / m), int(step / m))]

def draw_precision_recall_curve(data, total_good, thresholds_to_annotate, label=None):
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

    plt.plot(new_recalls, new_precisions, "^-", label=label, color="#5833FA")
    points = list(zip(new_thresholds, new_recalls, new_precisions))
    for i in range(len(points)):
        # Annotate point only if the current point is above the line connecting its neighbors
        if i == 0 or i == len(points) - 1 or is_above_curve(points[i - 1][1:], points[i + 1][1:], points[i][1:], 0.01):
            if points[i][0] in [30.65, 5.85]:
                continue

            th, r, p = points[i]
            plt.annotate(f"{th:.2f}", (r + 0.01, p + 0.01))

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

def find_proposals(path_proposals):
    proposals = read_all(path_proposals)
    proposals = proposals.replace("\n", "")

    classmembers = re.findall(r"<classmember .+?</classmember>", proposals)
    return [process_classmember(cm) for cm in classmembers]

def process_classmember(cm):
    vec = re.findall("vec\d\d\d\d\d", cm)[0]
    lemma = re.findall("lemma=\"(.+?)\"", cm)[0]
    scores = re.findall("score=\"(.+?)\"", cm)

    return Proposal(vec, lemma, float(scores[0]), list(map(float,scores[1:])))

def find_gold(gold_path):
    gold = read_all(gold_path)
    gold = gold.replace("\n", "")
    verbs = re.findall(r"<verb class=\".+?\" lemma=\".+?\">", gold)
    return set(verb_process(verb) for verb in verbs)

def verb_process(verb):
    vec = re.findall("class=\"(.+?)\"", verb)[0]
    lemma = re.findall("lemma=\"(.+?)\"", verb)[0]

    return (vec, lemma)

def read_all(path):
    with open(path, mode="r", encoding="utf-8") as content_file:
        return content_file.read()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-proposals", type=str, help="Path to the proposal file.", required=True)
    parser.add_argument("--path-minikorpus", type=str, help="Path to the minikorpus file for gold annotations.", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    main(get_args())