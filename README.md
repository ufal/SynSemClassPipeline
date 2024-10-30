# Toolchain for pre-annotation of a new language in a semantic ontology

## Nástroj pro předanotaci nového jazyka v sémantické ontologii

### Petr Kašpárek

---

This toolchain is part of the SynSemClass project, whose goal is to "create specifications and definitions of a hierarchical event-type ontology, populated with words denoting events or states." See the [official website](https://ufal.mff.cuni.cz/synsemclass) for a more detailed overview, scientific publications and data.

The ontology is currently populated by English, Czech and German. This toolchain uses existing tools to provide annotations suggestions for new languages.

Two main approaches are supported:
- zero shot transfer -- the annotation is done directly on the target langauge corpus
- cross-lingual annotation projection -- the annotation is done on a parallel corpus and the annotation is projected using word alignment

# Existing tools used

- [UDPipe](https://ufal.mff.cuni.cz/udpipe) (Straka)
- [SymAlign](https://github.com/cisnlp/simalign) (Jalili Sabet et al.)
- [Custom tool ML for SynSemClass](https://github.com/ufal/SynSemClassML) (Straková et al.)

# Installation

1. Clone the repository

```sh
git clone https://github.com/ufal/SynSemClassPipeline
cd ./SynSemClassPipeline
```

2. Initialize the `SynSemClassML` submodule

```sh
git submodule init
git submodule update
```

3. Create a Python virtual environment and install the requirements

Linux:
```sh
python3 -m venv venv
venv/bin/pip3 install -r requirements.txt
```

Windows:
```sh
python -m venv venv
venv/Scripts/pip3.exe install -r requirements.txt
```

4. In order to run anything, you need to have a SynSemClass classification model. You can either train your own model, as described in the [SynSemClassML repository](https://github.com/ufal/SynSemClassML) and the relevant [publication](https://aclanthology.org/2023.law-1.9/), or contact either me or the [SynSemClassML lead author](https://github.com/strakova) directly to get an already trained model.

### Test command
To try if everything works, run this command from the root of the directory with the virtual environment active. Don't forget to fill in the path to the model.
```sh
scripts/main.py --ssc-pred-model=FILL_IN --task=evaluate --evaluation-corpus=data/corpus.xml --udpipe-model-target=ko --udpipe-model-source=en --lemmatization=scripts/ko_filter_lemma.py --verb-filter=scripts/ko_filter_lemma.py
```

# Usage

The pipeline is run through the `main.py` script (located in the scripts folder). The pipeline uses corpora in one or two languages. One language, which we call target, is the one that is yet to be added to the ontology and which we want to annotate. The second one we call source and is the language used to provide possibly better suggestions for annotations of the target language. The source language is usually English, but not necessarily.

The `main.py` script accepts the following arguments: (See `.vscode/launch.json` for examples.)
- task: possible values
    - corpus: Generates a corpus with suggested SSC annotations.
    - ssc: Generates an SSC language-specific file with suggested annotations. (Not yet implemented.)
    - evaluate: Evaluates the quality of the pipeline suggestions using given gold data.
    - pred-matrix: Creates and saves a compact representation of all the predictions for use in experiments.

The following arguments are used by all tasks:
- udpipe-service: URL to the UDPipe service (default: https://lindat.mff.cuni.cz/services/udpipe/api)
- udpipe-model-target (required): UDPipe model for the target language, a two letter language code can be used to select a model automatically.
- udpipe-model-source: UDPipe model for the source language.
- ssc-pred-model (required): path to the model used for ssc predictions
- lemmatization: path to the script to perform lemmatization. The script should contain a function called `lemmatize` that takes a word from the `ufal.udpipe` package and returns the lemma. If not provided, the lemma returned by UDPipe 2 is used.
- verb-filter: path to the script to perform verb filtering. The script should contain a function called `verb_filter` that takes a word from the `ufal.udpipe` package and True iff the word should be considered for the pipeline. If not provided, all words marked as `VERB` by UDPipe 2 will be used.

An example lemmatization and verb filtering script can be found in [`scripts/ko_filter_lemma.py`](scripts/ko_filter_lemma.py).

The following arguments are used for the corpus and ssc tasks:
- output: path to where the output corpus/ssc file will be produced.

Either of these should be provided:
- target-corpus: Plain text corpus of the target language. (Sentence per line)
- source-corpus: Corpus parallel to the target corpus in the source language.

Or:
- tmx-corpus: A parallel corpus in the tmx format.

The following arguments are used for the ssc task:
- lang-name: Three letter language code that is inserted into the SynSemClass file.
- corpref: A short name of the corpus to be inserted into the SynSemClass file.
- member-threshold: The threshold for members. Members with lower scores will not be put into the SynSemClass file.
- aggregation: The aggregation function to use.
- predict-max-k: Only top K predictions for each example will be taken into account.
- example-threshold: Only predictions with higher probability will be taken into account.
- example-count: Number of examples to show for each member in the SynSemClass file.

The following argument is used for the evaluation task:
- evaluation-corpus: Golden data used as evaluation in the format as produced by the corpus task (see below for description).

## Parallel corpus format
The pipeline uses a custom format for parallel corpora with SSC annotations. See below for an example.

The top tag is `sentences`. It is filled with `sentence` tags. The `id` of a sentence denotes its order in the original corpus. The `text` tag contains the text of the sentence in the target language, the `source` tag in the source language. They are followed by a `verbs` tag, which contains instances of verbs in the target language. Each verb contains an SSC `class`, which is not generated by the pipeline and is yet to be added by the annotator, and a `lemma`. Inside each `verb` tag a `mark` tag has the sentence in the target language marked with the hat (`^`) symbol. The generated corpus then contains a `predictions` tags with several `pred` tags for possible SSC classes. Each prediction has a `prob` attribute for the probability of this class, which is in percent, i.e. the max value is 100. The class is in the text of the tag. The `predictions` tag is followed by an `alignment` tag. If no alignment was made, the tag is empty, otherwise it contains the verb marked in the source sentence. In the generated corpus, if an alignment was made, an `alignment_predictions` tag follows with the same format as the `predictions` tag.

This is how the corpus would look after it is produced by the pipeline. For simplicity, the target language is English, but surrounded by two asterisks. For demonstration, aligned on the second verb failed.

```xml
<sentences>
  <sentence id="0">
    <text>**I sleep and eat.**</text>
    <source>I sleep and eat.</source>
    <verbs>
      <verb class="" lemma="sleep">
        <mark>**I ^ sleep and eat.**</mark>
        <predictions>
          <pred prob="58.6">vec00735</pred>
          <pred prob="2.6">vec00440</pred>
          <pred prob="2.2">vec00921</pred>
          <pred prob="1.8">vec01118</pred>
          <pred prob="1.7">vec00556</pred>
        </predictions>
        <alignment>I ^ sleep and eat.</alignment>
        <alignment_predictions>
          <pred prob="82.5">vec00077</pred>
          <pred prob="7.4">vec00270</pred>
          <pred prob="3.8">vec00092</pred>
          <pred prob="1.7">vec00337</pred>
          <pred prob="1.6">vec00810</pred>
        </alignment_predictions>
      </verb>
      <verb class="" lemma="eat">
        <mark>**I sleep and ^ eat.**</mark>
        <predictions>
          <pred prob="46.9">vec00077</pred>
          <pred prob="13.0">vec00092</pred>
          <pred prob="4.4">vec00270</pred>
          <pred prob="1.3">vec00337</pred>
          <pred prob="1.2">vec00120</pred>
        </predictions>
        <alignment></alignment>
      </verb>
    </verbs>
  </sentence>
  <sentence id=”1”>
    <!--another sentence would be here-->
  </sentence>
  <!--more sentences would follow-->
</sentences>
```

This is how the same corpus would look like after annotation.

```xml
<sentences>
  <sentence id="0">
    <text>**I sleep and eat.**</text>
    <source>I sleep and eat.</source>
    <verbs>
      <verb class="vec99999" lemma="sleep">
        <mark>**I ^ sleep and eat.**</mark>
        <alignment>I ^ sleep and eat.</alignment>
      </verb>
      <verb class="vec12345" lemma="eat">
        <mark>**I sleep and ^ eat.**</mark>
        <alignment>I sleep and ^ eat.</alignment>
      </verb>
    </verbs>
  </sentence>
  <sentence id=”1”>
    <!--another sentence would be here-->
  </sentence>
  <!--more sentences would follow-->
</sentences>
```
