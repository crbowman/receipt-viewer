from ner_tokenize import *
import sys
from nltk import word_tokenize

def write_default_ner(tagged_sents, writer):
    write_product_ner(tagged_sents, writer, default=True)

def write_product_ner(tagged_sents, writer, default = False):
    for tagged_sent in tagged_sents:
        for index, tuple in enumerate(tagged_sent):
            if default:
                lbl = ' O'
            elif index == 0:
                lbl = ' B-PROD'
            else:
                lbl = ' I-PROD'
            writer.write(tuple[0] + ' ' + tuple[1] + lbl)
            writer.write('\n')
        writer.write('\n')

        
def ner_product(product_name):
    write_product_ner(tokenize_and_tag(product_name), sys.stdout)


def auto_tag_receipt(tagged_sents, writer, products):
    annotated = []
    written = False
    for sent in tagged_sents:
        found_product = None
        for prod_index, prod in enumerate(products):
            tokens = word_tokenize(prod)
            sent_tokens = [x[0] for x in sent]
            index = get_index(sent_tokens, tokens)
            if index is not None:
                annotated.append(product_annotate(sent, index, tokens))
                found_product = index
                break
        if found_product is not None:
            # products.pop(prod_index)
            pass
        else:
            annotated.append(default_annotate(sent))
    for sent in annotated:
        for word, t, cls in sent:
            line = "{} {} {}\n".format(word, t, cls)
            if len(line.strip()) > 0:
                written = True
            writer.write(line)
        writer.write("\n")
    return written


def get_index(sent, prod_tokens):
    offset = 0
    index = None
    while len(sent[offset:]) > len(prod_tokens):
        first_tok = prod_tokens[0]
        if first_tok in sent[offset:] and sent.index(first_tok, offset) + len(prod_tokens) <= len(sent):
            first_index = sent.index(first_tok, offset)
            sub_sent = sent[first_index + offset: first_index + offset + len(prod_tokens)]
            if sub_sent == prod_tokens:
                index = first_index
                break
        else:
            break
        offset = first_index + 1
    return index


def product_annotate(sent, index, product_tokens):
    before = default_annotate(sent[:index])
    after = default_annotate(sent[index + len(product_tokens):])
    inside = []
    for i, _ in enumerate(product_tokens):
        word, t = sent[index + i]
        if i == 0:
            inside.append((word, t, "B-PROD"))
        else:
            inside.append((word, t, "I-PROD"))
    return before + inside + after

def default_annotate(sent):
    response = []
    for word, t in sent:
        response.append((word, t, "O"))
    return response

if __name__ == '__main__':
    ner_product(' '.join(sys.argv[1:]))