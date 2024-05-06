import csv
from collections import defaultdict
from collections import Counter

def make_into_gold_and_text(path):
    fields = ["lemma", "sentence", "synsemclass_id", "lang"]

    all_data = []

    with open(path, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        data = list(csv_reader)
        field_indices = {field: header.index(field) for field in fields}
        for line in data:
            dct = {field: line[field_indices[field]] for field in fields}
            all_data.append(dct)
    
    langs = sorted(set(d["lang"] for d in all_data))
    
    gold = defaultdict(set)
    sentences = defaultdict(set)

    for d in all_data:
        d["sentence"] = d["sentence"].replace("^ ", "")
        sentences[d["lang"]].add(d["sentence"])
        gold[d["lang"]].add(f"{d['lemma']}\t{d['synsemclass_id']}")


    for lang in langs:
        with open(f"ssc_data/text_ssc_{lang}.txt", "w", encoding="utf-8") as f:
            for k in sorted(sentences[lang]):
                f.write(k)
                f.write("\n")

        with open(f"ssc_data/gold_ssc_{lang}.txt", "w", encoding="utf-8") as f:
            for k in sorted(gold[lang]):
                f.write(k)
                f.write("\n")

def main():
    path = r"/home/strakova/synsemclass4.0/data/synsemclass4.0_80-10-10_active/examples.csv"
    make_into_gold_and_text(path)

if __name__ == "__main__":
    main()