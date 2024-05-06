
def morphemes_from_udpipe(udpipe_word):
    return udpipe_word.lemma.split("+")

def list_equals_and_present(list, index, compare):
    return len(list) > index and list[index] == compare

def lemmatize(udpipe_word):
    morphemes = morphemes_from_udpipe(udpipe_word)

    # Normalize passive to active
    if list_equals_and_present(morphemes, 1, "하") or \
       list_equals_and_present(morphemes, 1, "되") or \
       list_equals_and_present(morphemes, 1, "시키"):
        return morphemes[0] + "하다"

    return morphemes[0] + "다"

def verb_filter(udpipe_word):
    if udpipe_word.upostag != "VERB":
        return False
    
    morphemes = morphemes_from_udpipe(udpipe_word)

    # 이 - to be
    # -스럽다 word should have been marked as ADJ by UDPipe, but there are examples marked as verbs in the KAIST corpus for some reason
    if "이" in morphemes or \
       "스럽" in morphemes:
       return False
    
    

    return True