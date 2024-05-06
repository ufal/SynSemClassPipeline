import xml.etree.ElementTree as ET

def read_tmx_pairs(path, count=-1):
    pairs = []
    with open(path, mode="r", encoding="utf-8") as file:
        counter = 0
        while counter < count or count == -1:
            counter += 1
            next_tu = read_next_tu_block(file)
            if not next_tu:
                break
            pairs.append(next_tu)
    return pairs


class TMXParseException(Exception):
    pass

def read_next_tu_block(file):
    text = read_until(file, "<tu")
    if not text or not read_until(file, "</tu>", text):
        return None
    xml = "".join(text)

    root = ET.fromstring(xml)
    if len(root) != 2:
        raise TMXParseException("Invalid length\n" + xml)
    return (get_text_from_tuv(root[0], xml), get_text_from_tuv(root[1], xml))
    

def get_text_from_tuv(tuv, xml):
    if tuv.tag != "tuv":
        raise TMXParseException("Expected tuv but got\n" + xml)
    seg = tuv.findall("seg")
    if len(seg) != 1:
        raise TMXParseException("Expected exactly one seg\n" + xml)
    return seg[0].text
    

def read_until(file, stop_string, read_list=None):
    buffer = [None] * len(stop_string)
    while buffer != list(stop_string):
        buffer.pop(0)
        next_char = file.read(1)
        if next_char:
            buffer.append(next_char)
            if read_list:
                read_list.append(next_char)
        else:
            return None

    return buffer
