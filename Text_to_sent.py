from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def Text_into_sent(text):
    text=str(text)
    for c in ['...', '......']:
        text = text.replace(c, '. ')
    sentences = sent_tokenize(text)
    
    for s in sentences:
        if not any(c.isalpha() for c in s):
            sentences.remove(s)
    return sentences

def Num_of_sent(text):
    text=str(text)
    for c in ['...', '......']:
        text = text.replace(c, '. ')
    sentences = sent_tokenize(text)
    
    for s in sentences:
        if not any(c.isalpha() for c in s):
            sentences.remove(s)
    return len(sentences)