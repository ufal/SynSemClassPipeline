from collections import defaultdict
from main import Verb

from typing import List

def sorted_by_score(list):
    return sorted(list, key=lambda ex: ex.score, reverse=True)

class Predictions:
    MAX_CLASS = 1200

    def __init__(self, lang_name, corpref, aggregation, example_count, example_threshold):
        self.lang_name = lang_name
        self.corpref = corpref
        self.aggregation = aggregation
        self.example_count = example_count
        self.example_threshold = example_threshold
        self.members = defaultdict(lambda: defaultdict(list))

    def get_members(self, class_number):
        return self.members[class_number]
    
    def class_lemma(self, class_number):
        members = self.members[class_number]

        if not members:
            return ""
        
        return members[0].lemma

    
    @staticmethod
    def create(verbs: List[Verb], args):
        pred = Predictions(args.lang_name, args.corpref, args.aggregation, args.example_count, args.example_threshold)

        for verb in verbs:
            for clss, prob in verb.predictions:
                pred.members[get_class_number(clss)][verb.lemma].append(Example(verb, prob * 100))

        pred._normalize_members()

        return pred
    
    def _normalize_members(self):
        for clss in self.members:
            self.members[clss] = self._members_from_dict(self.members[clss], self.aggregation, self.example_threshold)

    @staticmethod
    def _members_from_dict(dct, aggregation, example_threshold):
        return sorted_by_score([Member(lemma, sorted_by_score(examples), aggregation, example_threshold) for lemma, examples in dct.items()])
    
    def filter_members(self, threshold):
        for clss in self.members:
            self.members[clss] = [member for member in self.members[clss] if member.score > threshold]


class Member:
    def __init__(self, lemma, examples, aggregation, threshold):
        self.lemma = lemma
        self.examples = examples
        self.aggregation = aggregation
        self.threshold = threshold
        self.compute_score()

    def compute_score(self):
        if self.aggregation == "avg":
            self.score = sum(ex.score for ex in self.examples if ex.score > self.threshold) / sum(1 for ex in self.examples if ex.score > self.threshold)
        elif self.aggregation == "max":
            self.score = max(ex.score for ex in self.examples if ex.score > self.threshold)
        else:
            raise ValueError(f"Invalid value for aggregation: {self.aggregation}")


class Example:
    def __init__(self, verb, score):
        self.verb = verb
        self.score = score
    
def get_class_number(clss):
    return int(clss[3:])
        
