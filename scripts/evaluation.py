#!/usr/bin/env python3

import xml.etree.ElementTree as ET

from typing import List
from dataclasses import dataclass
import random

class Verb:
    pass

class Sentence:
    pass

@dataclass
class Corpus:
    sentences: List[Sentence]

    def __init__(self, sentences):
        self.sentences = sentences
        self.count_on_verbs(lambda verb: verb, "total")

    def count_on_verbs(self, fnc, name):
        total = 0
        for sentence in self.sentences:
            sm = len([1 for verb in sentence.verbs if fnc(verb)])
            total += sm
            setattr(sentence, name, sm)
        setattr(self, name, total)

    def find_target(self):
        return [sentence.text for sentence in self.sentences]

    def find_source(self):
        return [sentence.source for sentence in self.sentences]

    def find_to_predict(self):
        marks =           [verb.mark           for sentence in self.sentences for verb in sentence.verbs]
        alignments =      [verb.alignment      for sentence in self.sentences for verb in sentence.verbs if verb.alignment]
        auto_alignments = [verb.auto_alignment for sentence in self.sentences for verb in sentence.verbs if verb.auto_alignment]
        return marks + alignments + auto_alignments

    def enrich_verbs(self, predictions):
        for sentence in self.sentences:
            for verb in sentence.verbs:
                verb.prediction = predictions[verb.mark]
                verb.alignment_prediction      = predictions[verb.alignment]      if verb.alignment      in predictions else None
                verb.auto_alignment_prediction = predictions[verb.auto_alignment] if verb.auto_alignment in predictions else None

                verb.prediction_correct                = verb.vec_class == verb.prediction
                verb.alignment_prediction_correct      = verb.vec_class == verb.alignment_prediction
                verb.auto_alignment_prediction_correct = verb.vec_class == verb.auto_alignment_prediction

    def evaluate_manual_alignment(self):
        self.count_on_verbs(lambda verb: verb.alignment,                "manually_aligned")
        self.count_on_verbs(lambda verb: verb.alignment_type == "verb", "manually_aligned_reasonable")

    def evaluate_auto_alignment(self, add_mark_index, source_conllu):
        for sentence, trans_conllu in zip(self.sentences, source_conllu):
            for verb in sentence.verbs:
                aligned = sorted([y for x, y in sentence.alignment if x == verb.index])
                verb.auto_alignment = add_mark_index(trans_conllu, aligned[0]) if aligned else None
        
        auto_alignment_correct = lambda verb: verb.auto_alignment == verb.alignment or (not verb.alignment and not verb.auto_alignment)
        auto_alignment_precision = lambda verb: verb.auto_alignment and verb.auto_alignment == verb.alignment

        self.count_on_verbs(lambda verb: auto_alignment_correct(verb),   "auto_alignment_correct")
        self.count_on_verbs(lambda verb: verb.auto_alignment,            "auto_aligned")
        self.count_on_verbs(lambda verb: auto_alignment_precision(verb), "auto_aligned_precision")

        self.count_on_verbs(lambda verb: auto_alignment_correct(verb)   and verb.alignment, "auto_alignment_correct_A")
        self.count_on_verbs(lambda verb: verb.auto_alignment            and verb.alignment, "auto_aligned_A")
        self.count_on_verbs(lambda verb: auto_alignment_precision(verb) and verb.alignment, "auto_aligned_precision_A")

        self.count_on_verbs(lambda verb: auto_alignment_correct(verb)   and verb.alignment_type == "verb", "auto_alignment_correct_AR")
        self.count_on_verbs(lambda verb: verb.auto_alignment            and verb.alignment_type == "verb", "auto_aligned_AR")
        self.count_on_verbs(lambda verb: auto_alignment_precision(verb) and verb.alignment_type == "verb", "auto_aligned_precision_AR")
                


    def evaluate_verbs(self, conllu_target, find_index_of_mark):
        for sentence, conllu in zip(self.sentences, conllu_target):
            sentence.verbs_found = 0
            for verb in sentence.verbs:
                verb.index = find_index_of_mark(verb.mark, conllu)
                if conllu.words[1 + verb.index].upostag == "VERB":
                    verb.found = True
                    sentence.verbs_found += 1
                else:
                    verb.found = False
            
            sentence.extra_verbs = len([1 for word in conllu.words[1:] if word.upostag == "VERB"]) - sentence.verbs_found

        self.verbs_found = sum(sentence.verbs_found for sentence in self.sentences)
        self.extra_verbs = sum(sentence.extra_verbs for sentence in self.sentences)


    def evaluate_predictions(self):
        self.count_on_verbs(lambda verb: verb.prediction_correct,                                                  "zeroshot_prediction_correct")
        self.count_on_verbs(lambda verb: verb.prediction_correct and verb.alignment,                               "zeroshot_prediction_correct_A")
        self.count_on_verbs(lambda verb: verb.prediction_correct and verb.alignment_type == "verb",                "zeroshot_prediction_correct_AR")
     
        self.count_on_verbs(lambda verb: verb.alignment_prediction_correct,                                        "alignment_prediction_correct")
        self.count_on_verbs(lambda verb: verb.alignment_prediction_correct and verb.alignment,                     "alignment_prediction_correct_A")
        self.count_on_verbs(lambda verb: verb.alignment_prediction_correct and verb.alignment_type == "verb",      "alignment_prediction_correct_AR")

        self.count_on_verbs(lambda verb: verb.auto_alignment_prediction_correct,                                   "auto_alignment_prediction_correct")
        self.count_on_verbs(lambda verb: verb.auto_alignment_prediction_correct and verb.alignment,                "auto_alignment_prediction_correct_A")
        self.count_on_verbs(lambda verb: verb.auto_alignment_prediction_correct and verb.alignment_type == "verb", "auto_alignment_prediction_correct_AR")

        self.count_on_verbs(lambda verb: verb.auto_alignment and verb.auto_alignment_prediction_correct,                                   "auto_alignment_prediction_precision")
        self.count_on_verbs(lambda verb: verb.auto_alignment and verb.auto_alignment_prediction_correct and verb.alignment,                "auto_alignment_prediction_precision_A")
        self.count_on_verbs(lambda verb: verb.auto_alignment and verb.auto_alignment_prediction_correct and verb.alignment_type == "verb", "auto_alignment_prediction_precision_AR")

        self.count_on_verbs(lambda verb: verb.found and verb.prediction_correct, "zeroshot_system_correct")
        self.count_on_verbs(lambda verb: verb.found and verb.auto_alignment_prediction_correct, "alignment_system_correct")
        self.count_on_verbs(lambda verb: verb.found and verb.auto_alignment, "found_and_aligned")


        recall_better_model = lambda sentence: (sentence.zeroshot_system_correct,  sentence.total)
        recall_worse_model  = lambda sentence: (sentence.alignment_system_correct, sentence.total)
        
        precision_better_model = lambda sentence: (sentence.zeroshot_system_correct,  sentence.verbs_found + sentence.extra_verbs)
        precision_worse_model  = lambda sentence: (sentence.alignment_system_correct, sentence.found_and_aligned + sentence.extra_verbs)

        def universal_metric(list):
            correct, total = zip(*list)
            return sum(correct) / sum(total)

        # probability one is better by chance
        self.p_zeroshot_better_recall    = self.permutation_test(universal_metric, recall_better_model, recall_worse_model)
        self.p_zeroshot_better_precision = self.permutation_test(universal_metric, precision_better_model, precision_worse_model)

    def permutation_test(self, metric, better_model, worse_model):
        NUMBER_OF_SAMPLES = 100000
        base = metric([better_model(sentence) for sentence in self.sentences])
        better = 0
        for sample_n in range(NUMBER_OF_SAMPLES):
            sample = [better_model(sentence) if random.randrange(0, 2) else worse_model(sentence) for sentence in self.sentences]
            if metric(sample) >= base:
                better += 1

        return better / NUMBER_OF_SAMPLES


def parse_corpus(path):
    tree = ET.parse(path)
    root = tree.getroot()
    return Corpus([parse_sentence(sentence) for sentence in root.iter("sentence")])

def parse_sentence(sentence_xml: ET.Element):
    sentence = Sentence()
    sentence.id = sentence_xml.attrib["id"]
    sentence.text = sentence_xml.find("text").text
    sentence.source = sentence_xml.find("source").text
    sentence.verbs = [parse_verb(verb) for verb in sentence_xml.iter("verb")]
    return sentence

def parse_verb(verb_xml):
    verb = Verb()
    verb.vec_class = verb_xml.attrib["class"]
    verb.lemma = verb_xml.attrib["lemma"]
    verb.mark = verb_xml.find("mark").text
    verb.alignment = verb_xml.find("alignment").text
    verb.alignment_type = verb_xml.find("alignment").attrib["to"]
    return verb
