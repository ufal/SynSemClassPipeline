import xml.etree.cElementTree as ET


def write_synsemclass(file_path, predictions):
    root = ET.Element(f"SynSemClass_{predictions.lang_name.upper()}")
    write_header(root, predictions)
    write_body(root, predictions)

    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(file_path, encoding="utf-8")


def write_header(root: ET.Element, predictions):
    header = ET.SubElement(root, "header")
    ET.SubElement(header, "edition").text = "1"
    ET.SubElement(header, "version").text = "1.0"
    ET.SubElement(header, "description").text = f"SynSemClass Lexicon file for {predictions.lang_name} classmembers"

    list_of_users = ET.SubElement(header, "list_of_users")
    ET.SubElement(list_of_users, "user", {"id": "AA", "can_modify": "YES", "name": "Annotator"})
    ET.SubElement(list_of_users, "user", {"id": "SYS", "can_modify": "YES", "name": "Admin"})
    ET.SubElement(list_of_users, "user", {"id": "SL", "can_modify": "YES", "name": "SynSemClass LEXICON"})

    reflexicons = ET.SubElement(header, "reflexicons")

    synsemclass_lexicon = ET.SubElement(reflexicons, "lexicon", {"id": "synsemclass", "name": "synsemclass"})
    ET.SubElement(synsemclass_lexicon, "lexref")
    ET.SubElement(synsemclass_lexicon, "lexbrowsing")
    ET.SubElement(synsemclass_lexicon, "lexsearching")
    argumentsused = ET.SubElement(synsemclass_lexicon, "argumentsused")
    for arg in "012345mM":
        argdesc = ET.SubElement(argumentsused, "argumentsused", {"id": f"vecargSA{arg}"})
        ET.SubElement(argdesc, "comesfrom", {"lexicon": "synsemclass"})
        ET.SubElement(argdesc, "label").text = "SynSemClassArg" + arg
        ET.SubElement(argdesc, "shortlabel").text = "SA" + arg

    ET.SubElement(reflexicons, "lexicon", {"id": predictions.corpref, "name": predictions.corpref}).text = "   "


def write_body(root: ET.Element, predictions):
    body = ET.SubElement(root, "body")

    for number in range(1, predictions.MAX_CLASS + 1):
        write_class(body, number, predictions)


def get_class_number(number):
    return "{:0>5d}".format(number)

def make_class_name(number):
    return "vec" + get_class_number(number)

def make_member_name(class_number, lang_name, member_number):
    return f"vec{get_class_number(class_number)}-{lang_name}-cm{get_class_number(member_number)}"

def write_class(body, class_number, predictions):
    class_name = make_class_name(class_number)
    veclass = ET.SubElement(body, "veclass", {"id": class_name, "lemma": predictions.class_lemma(class_number)})
    ET.SubElement(veclass, "class_definition")
    classmembers = ET.SubElement(veclass, "classmembers")
    
    for i, member in enumerate(predictions.get_members(class_number)):
        member_name = make_member_name(class_number, predictions.lang_name, i + 1)
        classmember = ET.SubElement(classmembers, "classmember",
                                    {"id": member_name,
                                     "idref": f"SynSemClass-ID-{member_name}",
                                     "lang": predictions.lang_name,
                                     "status": "not_touched",
                                     "lexidref": "synsemclass",
                                     "lemma": member.lemma,
                                     "score": f"{member.score:.1f}"})
        ET.SubElement(classmember, "maparg")
        ET.SubElement(classmember, "restrict")
        ET.SubElement(classmember, "cmnote")
        ET.SubElement(classmember, "extlex")
        examples = ET.SubElement(classmember, "examples")

        for ex in member.examples[:predictions.example_count]:
            ET.SubElement(examples, "example", {"corpref": predictions.corpref,
                                                "frpair": member.lemma,
                                                "nodeid": f"{predictions.corpref}-{ex.verb.line}-v{ex.verb.verb_index}",
                                                "score": f"{ex.score:.1f}"})

def test():
    # Kept here as an informal definition of the interface
    # Remove when outdated or deemed unnecessary

    print("running test synsemclass print")

    class Pred:
        MAX_CLASS = 5

        def __init__(self):
            self.lang_name = "etg"
            self.corpref = "korp"
            self.members = {
                2: [Member("맞다", [Example(1, 55.5)])],
                3: [Member("하다", [Example(16, 98.9), Example(17, 71.4)]), Member("가다", [Example(77, 12.5)])]
            }
        
        def get_members(self, class_number):
            if class_number not in self.members:
                return []
            return self.members[class_number]
        
        def class_lemma(self, class_number):
            if class_number not in self.members:
                return ""
            
            members = self.members[class_number]

            if not members:
                return ""
            
            return members[0].lemma


    class Member:
        def __init__(self, lemma, examples):
            self.lemma = lemma
            self.examples = examples
            self.score = max(ex.score for ex in self.examples)

    class Example:
        def __init__(self, nodeid, score):
            self.nodeid = nodeid
            self.score = score

    write_synsemclass("test", Pred())
        
if __name__ == "__main__":
    #test()
    pass