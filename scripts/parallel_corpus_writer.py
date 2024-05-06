#!/usr/bin/env python3

import xml.etree.cElementTree as ET

class ParallelCorpusWriter:
    def __init__(self):
        self.root = ET.Element("sentences")
        self.cur_sentence_index = 0

    def add(self, target_text, source_text, verbs):
        sentence = ET.SubElement(self.root, "sentence", id=str(self.cur_sentence_index))
        self.cur_sentence_index += 1
        
        ET.SubElement(sentence, "text").text = target_text
        ET.SubElement(sentence, "source").text = source_text

        verbs_elem = ET.SubElement(sentence, "verbs")
        for verb in verbs:
            v_elem = ET.SubElement(verbs_elem, "verb")
            v_elem.set("class", "")
            v_elem.set("lemma", verb.lemma)
            ET.SubElement(v_elem, "mark").text = verb.mark
            predictions_elem = ET.SubElement(v_elem, "predictions")
            self.write_predictions(predictions_elem, verb.predictions)
            ET.SubElement(v_elem, "alignment").text = verb.alignment if verb.alignment else ""
            if verb.alignment:
                alignment_predictions_elem = ET.SubElement(v_elem, "alignment_predictions")
                self.write_predictions(alignment_predictions_elem, verb.alignment_predictions)
        
    def write_predictions(self, head, predictions):
      for prediction in predictions:
        prob_str = f"{100*prediction[1]:0.1f}"
        ET.SubElement(head, "pred", prob=prob_str).text = prediction[0]

    def save(self, path):
        tree = ET.ElementTree(self.root)
        ET.indent(tree)
        tree.write(path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)
