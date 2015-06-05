import nltk.data
from nltk.tokenize import word_tokenize
from sbx_utils import load_tagger

SENTENCE_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
TAGGER = load_tagger()

def parse_sents(annotated_text):
    res = []
    result = annotated_text.splitlines()
    for s in result:
        sents = SENTENCE_DETECTOR.tokenize(s)
        res.extend(sents)

    return res

def tokenize(sentence):
    return word_tokenize(sentence)

def tag(tokens):
    return TAGGER.tag(tokens)

def tokenize_and_tag(annotated_text):
    '''
    Split in sents, tokenize and tag the given text
    :param annotated_text:
    :return:
    '''
    return [tag(tokenize(sent)) for sent in parse_sents(annotated_text)]