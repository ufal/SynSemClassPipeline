#!/usr/bin/env python3

import argparse
import email.mime.multipart
import email.mime.nonmultipart
import json
from typing import List
import urllib.request
from collections import namedtuple
from dataclasses import dataclass
from collections import defaultdict

import lzma
import pickle

import sys
print("Loading ML libraries", file=sys.stderr)

import ufal.udpipe
from simalign import SentenceAligner

import tmx_parser
import synsemclass_prediction
import parallel_corpus_writer
import evaluation
import shared

print("Successfully loaded ML libraries", file=sys.stderr)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="ssc", type=str, help="Task to do, either corpus, ssc or evaluate.")

    parser.add_argument("--output", default="corpus.xml", type=str, help="Where to store the output file.")

    parser.add_argument("--udpipe-service", default="https://lindat.mff.cuni.cz/services/udpipe/api", type=str, help="UDPipe service URL.")
    parser.add_argument("--udpipe-model-target", type=str, help="UDPipe model for the target language.", required=True)
    parser.add_argument("--udpipe-model-source", default="en", type=str, help="UDPipe model for the source language.")
    parser.add_argument("--ssc-pred-model", type=str, help="Model for the SynSemClass prediction.", required=True)

    parser.add_argument("--verb-filter",  type=str, help="Path to a script to filter unwanted words.")
    parser.add_argument("--lemmatization", type=str, help="Path to a script to do extra lemmatization.")

    parser.add_argument("--target-corpus", type=str, help="Path to the corpus which is to be annotated.")
    parser.add_argument("--source-corpus", type=str, help="Path to translation of the target corpus.")

    parser.add_argument("--tmx-corpus", type=str, help="Path to the parallel corpus in TMX format.")

    parser.add_argument("--evaluation-corpus", type=str, help="Parallel corpus for evaluation.")

    parser.add_argument("--lang-name", type=str, help="Three letter language code that is inserted into the SynSemClass file.")
    parser.add_argument("--corpref", type=str, help="Name of the corpus to be inserted into the SynSemClass file.")

    parser.add_argument("--member-threshold", default=0., type=float, help="The threshold for members. Members with lower scores are not put into the SSC file.")

    parser.add_argument("--aggregation", default="avg", choices=["avg", "max"], help="The aggregation function to use.")
    parser.add_argument("--predict-max-k", default=9999, type=int, help="Only top K predictions for each example will be taken into account.")
    parser.add_argument("--example-threshold", default=0., type=float, help="Only predictions with higher probability will be taken into account.")

    parser.add_argument("--example-count", default=5, type=int, help="Number of examples to show for each member in the SSC file.")
    
    return parser.parse_args()

lemmatize = lambda word: word.lemma
verb_filter = lambda word: word.upostag == "VERB"

def load_scripts(args):
    if args.lemmatization:
        lemmatization_script = shared.load_module_from_file("lemmatization_script", args.lemmatization)
        global lemmatize
        lemmatize = lemmatization_script.lemmatize

    if args.verb_filter:
        verb_filter_script = shared.load_module_from_file("verb_filter_script", args.verb_filter)
        global verb_filter
        verb_filter = verb_filter_script.verb_filter

#The following method was entirely taken from the UDPipe Client by Milan Straka
def perform_request(server, method, params={}):
    if not params:
        request_headers, request_data = {}, None
    else:
        message = email.mime.multipart.MIMEMultipart("form-data")

        for name, value in params.items():
            payload = email.mime.nonmultipart.MIMENonMultipart("text", "plain")
            payload.add_header("Content-Disposition", "form-data; name=\"{}\"".format(name))
            payload.add_header("Content-Transfer-Encoding", "8bit")
            payload.set_payload(value, charset="utf-8")
            message.attach(payload)

        request_data = message.as_bytes().split(b"\n\n", maxsplit=1)[1]
        request_headers = {"Content-Type": message["Content-Type"]}

    with urllib.request.urlopen(urllib.request.Request(
        url="{}/{}".format(server, method), headers=request_headers, data=request_data
    )) as request:
        return json.loads(request.read())


def get_conllu(service_url, model, text):
    MAX_REQUEST_LINES = 500

    lines = text.splitlines()
    conllus = []

    print("Making CONLLU with UDPipe", file=sys.stderr)
    for i in range(0, len(lines), MAX_REQUEST_LINES):
        print(f"{i}/{len(lines)}", file=sys.stderr)
        cur_text = "\n".join(lines[i: i + MAX_REQUEST_LINES])
        conllus.append(get_conllu_single(service_url, model, cur_text))

    return "\n".join(conllus)



def get_conllu_single(service_url, model, text):
    data = {
        "output": "conllu",
        "tokenizer": "presegmented", 
        "tagger": None, # This actually means no options for the tagger, but use a tagger
        "model": model,
        "data": text,
    }

    response = perform_request(service_url, "process", data)
    if "result" in response:
        return response["result"]
    else:
        raise RuntimeError(f"{service_url} did not produce `result` in response")


def parse_conllu(conllu):
    conllu_reader = ufal.udpipe.InputFormat.newConlluInputFormat()
    conllu_reader.setText(conllu)

    sentences = []
    sentence = ufal.udpipe.Sentence()
    error = ufal.udpipe.ProcessingError()
    while conllu_reader.nextSentence(sentence, error):
        sentences.append(sentence)
        sentence = ufal.udpipe.Sentence()
    if error.occurred():
        raise RuntimeError(error.message)

    return sentences

MATCHING_METHOD=namedtuple("MatchingMethod", "long_name short_name")("itermax", "i")

def get_aligner():
    return SentenceAligner(model="xlmr", matching_methods=MATCHING_METHOD.short_name)

def print_alignment(tok_pres, tok_new, alignment):
    for (i, j) in alignment:
        print((tok_pres[i], tok_new[j]), end="")
    print()


def read_all(path):
    with open(path, mode="r", encoding="utf-8") as content_file:
        return content_file.read()

def read_corpus(args):
    pair_corp = args.target_corpus# and args.source_corpus #TODO: task-specific check
    if pair_corp and args.tmx_corpus:
        raise Exception("Only one type of corpus must be provided.")

    if pair_corp:
        target_corpus = read_all(args.target_corpus)
        source_corpus = read_all(args.source_corpus) if args.source_corpus else None
    elif args.tmx_corpus:
        pairs = tmx_parser.read_tmx_pairs(args.tmx_corpus)
        source_corpus = "\n".join(map(lambda x: x[0], pairs))
        target_corpus = "\n".join(map(lambda x: x[1], pairs))
    else:
        raise Exception("At least one type of corpus must be provided.")

    return target_corpus, source_corpus


def get_sentences_one(corpus, args):
    conllu = get_conllu(args.udpipe_service, args.udpipe_model_target, corpus)
    sentences = parse_conllu(conllu)
    return conllu, sentences

def get_sentences(target_corpus, source_corpus, args, print_conllu=False):
    conllu_target, sentences_target = get_sentences_one(target_corpus, args)
    conllu_source, sentences_source = get_sentences_one(source_corpus, args)

    if print_conllu:
        print("Printing CONLLU")
        print(conllu_source, "\n")
        print(conllu_target, "\n")

    return sentences_target, sentences_source

@dataclass
class Verb:
    mark: str = None
    mark_index: int = None
    verb_index: int = None
    lemma: str = None
    alignment: str = None
    predictions: List = None
    alignment_predictions: List = None

    def __init__(self, mark, mark_index, lemma, verb_index):
        self.mark = mark
        self.mark_index = mark_index
        self.lemma = lemma
        self.verb_index = verb_index

def mark_verbs(sentence):
    words = sentence.words[1:]  # 0 is root, skip
    verbs_at = [index for index, word in enumerate(words) if verb_filter(word)]
    return [  Verb(add_mark_index(sentence, mark_index),
                   mark_index,
                   lemmatize(words[mark_index]),
                   verb_index
                   )
            for verb_index, mark_index in enumerate(verbs_at)]

PREDICT_TAG = "^ "

def add_mark_index(sentence, mark_index):
    forms = []
    multiword_replace = {mwt.idFirst: mwt for mwt in sentence.multiwordTokens if mark_index + 1 not in range(mwt.idFirst + 1, mwt.idLast + 1)}
    sleep = 0
    
    for i, word in enumerate(sentence.words[1:]):
        if sleep:
            sleep -= 1
            continue
        
        if i == mark_index:
            forms.append(PREDICT_TAG)

        if i + 1 in multiword_replace:
            mwt = multiword_replace[i + 1]
            forms.append(mwt.form + mwt.getSpacesAfter())
            sleep = mwt.idLast - mwt.idFirst
        else:
            forms.append(word.form + word.getSpacesAfter())

    mark = "".join(forms).strip()
    return mark

def find_index_of_mark(mark, sentence):
    """
    The algorithm finds a closest place before the mark that it can placed
    so that the given tokenization is preserved. It is theoretically possible
    the mark is placed manually into a spot not recognized as token boundary by
    the tokenizer, thus the algorithm is very defensive.
    """
    
    premark = mark.split(PREDICT_TAG)[0]
    index = 0
    sentence_form = ""
    multiword_replace = {mwt.idFirst: mwt for mwt in sentence.multiwordTokens}
    multiword_on = True

    while True:
        if multiword_on and index + 1 in multiword_replace:
            mwt = multiword_replace[index + 1]
            new_sentence_form = sentence_form + mwt.form + mwt.getSpacesAfter()
            index_increment = mwt.idLast - mwt.idFirst + 1
        else:
            word = sentence.words[index + 1]
            new_sentence_form = sentence_form + word.form + word.getSpacesAfter()
            index_increment = 1

        if len(new_sentence_form) > len(premark):
            if not multiword_on:
                break

            # The mark might be manually placed inside a multiword token.
            # Turn off multiword search and try again.
            multiword_on = False
            continue

        sentence_form = new_sentence_form
        index += index_increment

    return index


def get_tokens(sentence):
    return [w.form for w in sentence.words[1:]]

class DeferredPrediction:
    def __init__(self, task):
        self.task = task

    @staticmethod
    def run(target, fields, fn):
        for field in fields:
            tasks = []
            storage = []
            for thing in target:
                if getattr(thing, field) and isinstance(getattr(thing, field), DeferredPrediction):
                    tasks.append(getattr(thing, field).task)
                    storage.append(thing)

            if tasks:
                results = fn(tasks)
                for thing, result in zip(storage, results):
                    setattr(thing, field, result)

def process_sentence_pair(target, source, aligner, predictor, pcw):
    verbs = mark_verbs(target)
    alignment = aligner.get_word_aligns(get_tokens(target), get_tokens(source))[MATCHING_METHOD.long_name]

    for verb in verbs:
        verb.predictions = DeferredPrediction(verb.mark)
        aligned = sorted([y for x, y in alignment if x == verb.mark_index])
        
        if not aligned:
            continue
        verb.alignment = add_mark_index(source, aligned[0])
        verb.alignment_predictions = DeferredPrediction(verb.alignment)

    DeferredPrediction.run(verbs, ["predictions", "alignment_predictions"], lambda tasks: predictor.predict_proba(tasks, 5))
    pcw.add(target.getText(), source.getText(), verbs)

def corpus(args):
    target_corpus, source_corpus = read_corpus(args)
    sentences_target, sentences_source = get_sentences(target_corpus, source_corpus, args, print_conllu=False)

    aligner = get_aligner()
    predictor = synsemclass_prediction.Predictor(args.ssc_pred_model)
    pcw = parallel_corpus_writer.ParallelCorpusWriter()
    for target, source in zip(sentences_target, sentences_source):
        process_sentence_pair(target, source, aligner, predictor, pcw)

    pcw.save(args.output)

def make_to_predict(args):
    target_corpus, _ = read_corpus(args)
    _, sentences = get_sentences_one(target_corpus, args)

    to_predict = []
    for i, sentence in enumerate(sentences):
        verbs = mark_verbs(sentence)
        for verb in verbs:
            verb.line = i
        to_predict.extend(verbs)

    for verb in to_predict:
        verb.predictions = DeferredPrediction(verb.mark)

    return to_predict

def ssc(args):
    import synsemclass_writer
    import predictions

    to_predict = make_to_predict(args)
    predictor = synsemclass_prediction.Predictor(args.ssc_pred_model)
    DeferredPrediction.run(to_predict, ["predictions"], lambda tasks: predictor.predict_proba(tasks, args.predict_max_k))

    preds = predictions.Predictions.create(to_predict, args)
    preds.filter_members(args.member_threshold)
    synsemclass_writer.write_synsemclass(args.output, preds)

def make_and_store_prediction_matrices(args):
    to_predict = make_to_predict(args)

    predictor = synsemclass_prediction.Predictor(args.ssc_pred_model)
    DeferredPrediction.run(to_predict, ["predictions"], lambda tasks: predictor.predict_proba_raw(tasks))

    lemmas = defaultdict(list)

    for verb in to_predict:
        lemmas[verb.lemma].append(verb.predictions)

    with lzma.open(f"ssc_data/{args.lang_name}_matrix.lzma.pickle", "wb") as f:
        pickle.dump(lemmas, f)


def evaluate(args):
    if not args.evaluation_corpus:
        print("Missing --evaluation-corpus", file=sys.stderr)
        exit(1)

    corpus = evaluation.parse_corpus(args.evaluation_corpus)
    target = "\n".join(corpus.find_target())
    source = "\n".join(corpus.find_source())


    sentences_target, sentences_source = get_sentences(target, source, args, print_conllu=False)
    corpus.evaluate_verbs(sentences_target, find_index_of_mark)


    corpus.evaluate_manual_alignment()
    
    aligner = get_aligner()
    for sentence, target_conllu, source_conllu in zip(corpus.sentences, sentences_target, sentences_source):
        alignment = aligner.get_word_aligns(get_tokens(target_conllu), get_tokens(source_conllu))[MATCHING_METHOD.long_name]
        sentence.alignment = alignment
    corpus.evaluate_auto_alignment(add_mark_index, sentences_source)

    to_predict = corpus.find_to_predict()
    predictor = synsemclass_prediction.Predictor(args.ssc_pred_model)
    predictions = predictor.predict(to_predict)
    corpus.enrich_verbs(dict(zip(to_predict, predictions)))
    corpus.evaluate_predictions()

    print(f"Verb recall (target):                 {corpus.verbs_found                          / corpus.total:.2f}")
    print(f"Verb precision (target):              {corpus.verbs_found                          / (corpus.verbs_found + corpus.extra_verbs):.2f}")
    print(f"Mean extra verbs (target):            {corpus.extra_verbs                          / len(corpus.sentences):.2f}")
    print(f"=====================================================================")
    print(f"No or any manual alignment")
    print(f"Percentage of data:                   {1.00:.2f}")
    print()
    print(f"Zero-shot        transfer accuracy:   {corpus.zeroshot_prediction_correct            / corpus.total:.2f}")
    print(f"Manual alignment transfer accuracy:   {corpus.alignment_prediction_correct           / corpus.total:.2f}")
    print(f"Auto-alignment   transfer accuracy:   {corpus.auto_alignment_prediction_correct      / corpus.total:.2f}")
    print(f"Auto-alignment  transfer precision:   {corpus.auto_alignment_prediction_precision    / corpus.auto_aligned:.2f}")
    print()  
    print(f"Automatic alignment accuracy:         {corpus.auto_alignment_correct                 / corpus.total:.2f}")
    print(f"Automatic alignment something:        {corpus.auto_aligned                           / corpus.total:.2f}")
    print(f"Automatic alignment precision:        {corpus.auto_aligned_precision                 / corpus.auto_aligned:.2f}")
    print(f"---------------------------------------------------------------------")
    print(f"Some manual alignment")
    print(f"Percentage of data:                   {corpus.manually_aligned                       / corpus.total:.2f}")
    print()
    print(f"Zero-shot        transfer accuracy:   {corpus.zeroshot_prediction_correct_A          / corpus.manually_aligned:.2f}")
    print(f"Manual alignment transfer accuracy:   {corpus.alignment_prediction_correct_A         / corpus.manually_aligned:.2f}")
    print(f"Auto-alignment   transfer accuracy:   {corpus.auto_alignment_prediction_correct_A    / corpus.manually_aligned:.2f}")
    print(f"Auto-alignment  transfer precision:   {corpus.auto_alignment_prediction_precision_A  / corpus.auto_aligned_A:.2f}")
    print()
    print(f"Automatic alignment accuracy:         {corpus.auto_alignment_correct_A               / corpus.manually_aligned:.2f}")
    print(f"Automatic alignment something:        {corpus.auto_aligned_A                         / corpus.manually_aligned:.2f}")
    print(f"Automatic alignment precision:        {corpus.auto_aligned_precision_A               / corpus.auto_aligned_A:.2f}")
    print(f"---------------------------------------------------------------------")
    print(f"Reasonable manual alignment")
    print(f"Percentage of data:                   {corpus.manually_aligned_reasonable            / corpus.total:.2f}")
    print()
    print(f"Zero-shot        transfer accuracy:   {corpus.zeroshot_prediction_correct_AR         / corpus.manually_aligned_reasonable:.2f}")
    print(f"Manual alignment transfer accuracy:   {corpus.alignment_prediction_correct_AR        / corpus.manually_aligned_reasonable:.2f}")
    print(f"Auto-alignment   transfer accuracy:   {corpus.auto_alignment_prediction_correct_AR   / corpus.manually_aligned_reasonable:.2f}")
    print(f"Auto-alignment  transfer precision:   {corpus.auto_alignment_prediction_precision_AR / corpus.auto_aligned_AR:.2f}")
    print()
    print(f"Automatic alignment accuracy:         {corpus.auto_alignment_correct_AR              / corpus.manually_aligned_reasonable:.2f}")
    print(f"Automatic alignment something:        {corpus.auto_aligned_AR                        / corpus.manually_aligned_reasonable:.2f}")
    print(f"Automatic alignment precision:        {corpus.auto_aligned_precision_AR              / corpus.auto_aligned_AR:.2f}")
    print(f"=====================================================================")  
    print(f"Whole system")  
    print(f"Zero-shot system recall:              {corpus.zeroshot_system_correct                / corpus.total:.2f}")
    print(f"Zero-shot system precision:           {corpus.zeroshot_system_correct                / (corpus.verbs_found + corpus.extra_verbs):.2f}")
    print()
    print(f"Automatic alignment system recall:    {corpus.alignment_system_correct               / corpus.total:.2f}")
    print(f"Automatic alignment system precision: {corpus.alignment_system_correct               / (corpus.found_and_aligned + corpus.extra_verbs):.2f}")
    print(f"=====================================================================")
    print(f"Probabilities zero-shot is better by chance")
    print(f"p-value recall:                       {corpus.p_zeroshot_better_recall:.5f}")
    print(f"p-value precision:                    {corpus.p_zeroshot_better_precision:.5f}")


def main(args):

    load_scripts(args)

    tasks = {
        "corpus": corpus,
        "ssc": ssc,
        "evaluate": evaluate,
        "pred-matrix": make_and_store_prediction_matrices
    }

    if args.task in tasks:
        tasks[args.task](args)
    else:
        print("Unknown task: " + args.task, file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    main(get_args())
