import nltk
from nltk.corpus import stopwords
from nltk import ngrams, Tree
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import re

"""functions to extract meaningfuly phrases"""
"""
Pattern 1: <JJ|JJR|JJS>+<NN|NNS>               exp: good sleep, strong taste
Pattern 2: <JJ|JJR|JJS>+<NN|NNS>+<NN|NNS>      exp: long sleep hours
Pattern 3: <NN|NNS>+<NN|NNS>                   exp: sleep time
Pattern 4: NN/NNS+RB/RBR/RBP+JJ                exp: taste very strong (was or is removed because they are stop words)
Pattern 5: <RB|RBR|RBS>+<JJ>+<NN|NNS>          exp: highly recommended product
Pattern 6: <RB|RBR|RBS>+<VB|VBD|VBG|VBN|VBP|VBZ>  exp: quickly wake up
Pattern 7: <VB|VBD|VBG|VBN|VBP|VBZ>+<RB|RBR|RBS>  exp:sleep well
Pattern 8: <RB|RBR|RBS>+<VB|VBD|VBG|VBN|VBP|VBZ>+<RB|RBR|RBS>  exp:not sleep well
Pattern 9: <VB|VBD|VBG|VBN|VBP|VBZ>+<NN|NNS>      exp: have dreams
"""



def Chunk_to_term(chunked, label):
    terms=list()
    for node in chunked:
        if isinstance(node, Tree):               
            if node.label() == label:
                term=' '.join(node[n][0] for n in range(len(node)))
                terms.append(term)
    return (terms)

def Phrases(text):
    stop = set(stopwords.words('english'))-set(['not'])
    terms=list()
    words = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(words)
    tagged=[tup for tup in tagged if tup[0] not in stop]
    try:
        chunk1=r"""Chunk: {<JJ|JJR|JJS>+<NN|NNS>}"""
        chunk2=r"""Chunk: {<JJ|JJR|JJS>+<NN|NNS>+<NN|NNS>}"""
        chunk3=r"""Chunk: {<NN|NNS>+<NN|NNS>}"""
        chunk4=r"""Chunk: {<NN|NNS>+<RB|RBR|RBS>+<JJ|JJR|JJS>}"""
        chunk5=r"""Chunk: {<RB|RBR|RBS>+<JJ>+<NN|NNS>}"""
        chunk6=r"""Chunk: {<RB|RBR|RBS>+<VB|VBD|VBG|VBN|VBP|VBZ>}"""
        chunk7=r"""Chunk: {<VB|VBD|VBG|VBN|VBP|VBZ>+<RB|RBR|RBS>}"""
        chunk8=r"""Chunk: {<RB|RBR|RBS>+<VB|VBD|VBG|VBN|VBP|VBZ>+<RB|RBR|RBS>}"""
        chunk9=r"""Chunk: {<VB|VBD|VBG|VBN|VBP|VBZ>+<NN|NNS>}"""
        chunks=[chunk1,chunk2,chunk3,chunk4,chunk5, chunk6, chunk7,chunk8]
        for chunk in chunks:
            chunkParser = nltk.RegexpParser(chunk)
            chunked = chunkParser.parse(tagged)
            terms.extend(Chunk_to_term(chunked, 'Chunk'))
        return (terms)
    except Exception as e:
        print(str(e))  

def Phrase_set(terms):
    return (terms[[sum([s in t for t in terms])==1 for s in terms]])

"""function to preprocess the text data"""
def Preprocessor(x):
    stop = set(stopwords.words('english'))-set(['not'])
    """list of words with meaning of "not" and the form they are to be tranformed into"""
    transform={'dont':'do not',
     'don\'t':'do not',
     'donot':'do not',
     'cannot':'can not',
     'can\'t':'can not',
     'cant':'can not',
     'didnt':'did not',
     'didn\'t':'did not',
     'isnt':'is not',
     'isn\'t':'is not',
     'arent':'are not',
     'aren\'t':'are not',
     'arenot':'are not',
     'wasn\'t':'was not',
     'wasnt':'was not',    
      'wasnot':'was not',
     'weren\'t':'were not',
     'werent':'were not',
     'wont':'will not',
     'won\'t':'will not',
     'couldnt':'could not',
     'couldn\'t':'could not',
     'shouldnt':'should not',
     'shouldn\'t':'should not',
     'wouldn\'t':'would not',
     'wouldnt': 'would not',
     'hasnt':'has not',
     'hasn\'t':'has not',
     'havent':'have not',
     'haven\'t':'have not',
     'hadnt':'had not',
     'hadn\'t':'had not',
     'gonna':'going to',
     'wanna':'want to'
     }
    
    x=str(x)
    """only select letters and transform all letters to be lowercase"""
    x = re.sub('[^A-Za-z,:?!\']', ' ', x.lower()) 
    
    """seperate not from words so that it can be kept for later analysis"""
    x = ' '.join([transform.get(i, i) for i in x.split()])
    
    """spelling correction"""
    x = ' '.join([re.compile(r"(.)\1{2,}").sub(r"\1", w) for w in x.split()])
    
    """remove punctuations"""
    x = re.sub('[^A-Za-z]', ' ', x)  
    
    """remove stopwords from the sentence"""
    x = ' '.join([w for w in x.split() if w not in set(stop)])
    
    """lemmatization"""
    lem=WordNetLemmatizer()
    x = ' '.join([lem.lemmatize(w) for w in x.split()])
    
    return (x)

"""function to preprocess the text data"""
def Extract_terms(x):
    stop = set(stopwords.words('english'))-set(['not'])
    """list of words with meaning of "not" and the form they are to be tranformed into"""
    transform={'dont':'do not',
     'don\'t':'do not',
     'donot':'do not',
     'cannot':'can not',
     'can\'t':'can not',
     'cant':'can not',
     'didnt':'did not',
     'didn\'t':'did not',
     'isnt':'is not',
     'isn\'t':'is not',
     'arent':'are not',
     'aren\'t':'are not',
     'arenot':'are not',
     'wasn\'t':'was not',
     'wasnt':'was not',    
      'wasnot':'was not',
     'weren\'t':'were not',
     'werent':'were not',
     'wont':'will not',
     'won\'t':'will not',
     'couldnt':'could not',
     'couldn\'t':'could not',
     'shouldnt':'should not',
     'shouldn\'t':'should not',
     'wouldn\'t':'would not',
     'wouldnt': 'would not',
     'hasnt':'has not',
     'hasn\'t':'has not',
     'havent':'have not',
     'haven\'t':'have not',
     'hadnt':'had not',
     'hadn\'t':'had not',
     'gonna':'going to',
     'wanna':'want to'
     }
    
    x=str(x)
    """seperate not from words so that it can be kept for later analysis"""
    x = ' '.join([transform.get(i, i) for i in x.split()])
    
    """spelling correction"""
    x = ' '.join([re.compile(r"(.)\1{2,}").sub(r"\1\1", w) for w in x.split()])
  
    """generate chunked terms without stop words"""
    terms=Phrases(x)
    terms_set=Phrase_set(np.array(terms))
    terms_set=[re.sub('[^A-Za-z]', ' ', s.lower()) for s in terms_set]

    lem=WordNetLemmatizer()
    terms_set=[' '.join([lem.lemmatize(w) for w in t.split(' ')]) for t in terms_set]
    
    return (terms_set)

def Preprocessor_with_punc(x):  
    """list of words with meaning of "not" and the form they are to be tranformed into"""
    transform={'dont':'do not',
     'don\'t':'do not',
     'donot':'do not',
     'cannot':'can not',
     'can\'t':'can not',
     'cant':'can not',
     'didnt':'did not',
     'didn\'t':'did not',
     'isnt':'is not',
     'isn\'t':'is not',
     'arent':'are not',
     'aren\'t':'are not',
     'arenot':'are not',
     'wasn\'t':'was not',
     'wasnt':'was not',    
     'wasnot':'was not',
     'weren\'t':'were not',
     'werent':'were not',
     'wont':'will not',
     'won\'t':'will not',
     'couldnt':'could not',
     'couldn\'t':'could not',
     'shouldnt':'should not',
     'shouldn\'t':'should not',
     'wouldn\'t':'would not',
     'wouldnt': 'would not',
     'hasnt':'has not',
     'hasn\'t':'has not',
     'havent':'have not',
     'haven\'t':'have not',
     'hadnt':'had not',
     'hadn\'t':'had not',
     'gonna':'going to',
     'wanna':'want to',
     'it\'s':'it is',
     'he\'s':'he is',
     'she\'s':'she is',
     }
    
    x=str(x)
    
    """only select letters and transform all letters to be lowercase"""
    x = re.sub('[^A-Za-z,.:?!\']', ' ', x.lower()) 
    
    """seperate not from words so that it can be kept for later analysis"""
    x = ' '.join([transform.get(i, i) for i in x.split()])
    
    """remove stopwords from the sentence"""
    x = ' '.join([w for w in x.split() if w not in set(stop)])

    """spelling correction"""
    x = ' '.join([re.compile(r"(.)\1{2,}").sub(r"\1\1", w) for w in x.split()])    
      
    
    
    """add spaces after punctuation to ensure appropriate work tokenization"""
    x=re.sub(r'(?<=[.,?!:])(?=[^\s])', r' ', x)

    """lemmatization"""
    lem=WordNetLemmatizer()
    x = ' '.join([lem.lemmatize(w) for w in x.split()])
    
    return (x)
    
