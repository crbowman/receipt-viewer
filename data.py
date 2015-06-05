import pickle
from itertools import dropwhile, takewhile
import json
from nltk import word_tokenize, pos_tag
from numpy import random

SHOULD_ADD = [True, False]
PROD_TAGS = ['B-PROD', 'I-PROD']

class ReceiptDecorator():
    """ Given an annotated file with B/I/O tagging, and a collection of products ,
     we add the right number of product lines to th annotated file so that
     the final percentage of the B/I tags inside it is approximately 50%"""
    def __init__(self, reader, product_reader):
        """

        :param reader: AnnotatedReader object
        :param product_reader: a object of type ProductFileReader that has
        the products already read
        :return:
        """
        self.reader = reader
        self.product_reader = product_reader
        if len(self.reader.data) == 0:
            raise Exception('AnnotatedReader has no data !')
        first_sent = None
        for idx, sent in enumerate(self.reader.data):
            for tok in sent:
                if tok['class'] in PROD_TAGS:
                    first_sent = sent
                    self.insert_index = idx
                    break
        self.has_product = first_sent is not None
        if first_sent is not None:
            self.product_before = list(takewhile(lambda w: w['class'] == 'O', first_sent))
            self.products_after = list(dropwhile(lambda w: w['class'] != 'O', first_sent[len(self.product_before):]))
            self.pad_length = len(self.product_before + self.products_after)
            self._get_stats()
        else:
            print('File {} has no products ... skipping'.format(self.reader.file_name))

    def _get_stats(self):
        o_count = 0.
        bi_count = 0.
        for sent in self.reader.data:
            for tagged in sent:
                if tagged['class'] == 'O':
                    o_count += 1.
                elif tagged['class'] in PROD_TAGS:
                    bi_count += 1.
                else:
                    raise Exception('Tag not recognized : ' + tagged['class'])
        self.o_count = o_count
        self.bi_count = bi_count

    def enrich(self):
        while self.has_product and self.bi_count/self.o_count <= 1.:
            added_o = self.pad_length
            product = self.product_reader.get_product_line(min_len=added_o)
            added_bi = len(product)
            new_ratio = (self.bi_count + added_bi) / (self.o_count + added_o)
            if new_ratio >= 1.:
                # this is the final iteration... do we add the product or not?
                # we break either way but we decide if we want to add the product or not
                should_add = random.choice(SHOULD_ADD)
                if should_add:
                    self.add_product(product)
                    self.bi_count += added_bi
                    self.o_count += added_o
                break
            else:
                # add the product as the new ratio will not be larger than 1
                self.add_product(product)
                self.bi_count += added_bi
                self.o_count += added_o

    def add_product(self, product):
        before = [(x['word'], x['class']) for x in self.product_before]
        after = [(x['word'], x['class']) for x in self.products_after]
        product_line = before + product + after
        pos_tagged = [y[1] for y in pos_tag([x[0] for x in product_line])]
        tagged_and_classed = []
        for i in range(len(pos_tagged)):
            tagged_and_classed.append({
                'class': product_line[i][1],
                'word': product_line[i][0],
                'tag': pos_tagged[i]
            })

        self.reader.data.insert(self.insert_index, tagged_and_classed)


class ProductFileReader():
    def __init__(self, product_file_path):
        with open(product_file_path) as input_file:
            product_lines = input_file.readlines()
        self.products = [word_tokenize(p) for p in product_lines]

    def get_product_line(self, min_len=0):
        products = [p for p in self.products if len(p) > min_len]
        if len(products) == 0:
            raise Exception('No product found with {} tokens'.format(min_len + 1))
        product = random.choice(products) if len(products) > 1 else products[0]
        product = [(w, 'B-PROD') if index == 0 else (w, 'I-PROD') for index, w in enumerate(product)]
        return product


class AnnotatedReader():
    def __init__(self, file_name):
        self.file_name = file_name
        self._read()

    def _read(self):
        with open(self.file_name) as input_file:
            data = input_file.read()
            # Take out the first part, the product names
        split = data.split('\n\n\n')
        if len(split) > 1:
            prods, sents = split[0], split[1]
        else:
            prods, sents = '', split[0]
        if len(sents) == 0 or len(sents.split('\n\n')) <= 1:
            prods, sents = sents, prods
        self.products = prods.split('\n')
        sents = sents.split('\n\n')
        sents = [x.split('\n') for x in sents if len(x.split('\n'))>0]
        sents = [list(filter(lambda x: len(x) > 0, g)) for g in sents]
        self.data = []
        for s in sents:
            sent = []
            for group in s:
                split = group.split(' ')
                sent.append({'word': split[0], 'tag': split[1], 'class': split[2]})
            self.data.append(sent)

    def get_data(self):
        return self.data

    def get_products(self):
        return self.products

    def data_as_tuple(self):
        sents = []
        for s in self.data:
            pairs =[]
            for group in s:
                pairs.append((group['word'], group['tag']))
            sents.append(pairs)
        return sents

    def __repr__(self):
        return repr(self.data)

    def as_file(self, file_path):
        with open(file_path, 'w') as out_file:
            for sent in self.data:
                lines = ['{} {} {}'.format(w['word'], w['tag'], w['class']) for w in sent]
                s = '\n'.join(lines)
                out_file.write(s + '\n\n')

    def output_readable(self, file_path):
        try:
            with open(file_path) as backup:
                backup_data = backup.read()
        except FileNotFoundError:
            backup_data = None
        try:
            with open(file_path, 'w') as out_file:
                for s in self.data:
                    formatted = ('{} ' * len(s)) + '\n'
                    line = formatted.format(*[x['word'] for x in s])
                    out_file.write(line)
        except Exception:
            if backup_data is not None:
                with open(file_path, 'w') as backup:
                    backup.write(backup_data)


class TrainingDataLoader:
    @staticmethod
    def from_json(path):
        """
        Loads data from JSON format
        :param path: The path of the JSON file
        :return: A tuple of training_data and target vector
        """

        json_data = json.loads(open(path, 'rb').read().decode(errors='ignore'))
        training_data = [ParsedMessage(m[0]) for m in json_data]
        target = [x[1] for x in json_data]
        return training_data, target

    @staticmethod
    def from_pickle(path):
        """
        Loads pickled training and target
        :param path: The path of the pickle file
        :return: A tuple of training_data and target vector
        """

        all_data = pickle.load(open(path, 'rb'))
        training_data = [x[0] for x in all_data]
        target = [x[1] for x in all_data]
        return training_data, target


class Content(object):
    """
    Contains the text parsed from an email message
    text_list : A list of texts that had the content 'text/plain' inside the initial message
    html_list : A list of texts that had the content 'text/html' inside the initial message
    """
    def __init__(self, text_list, html_list):
        self.text_list = text_list
        self.html_list = html_list

    def __eq__(self, other):
        return type(self) == type(other) and self.text == other.text and self.html == other.html


class ParsedMessage(object):
    """
    Contains data parsed from a email message
    From : the sender address
    To : the receiver address
    Subject : The email subject
    Source : The source file it was parsed from (this can be empty)
    has_attachment : True if the email has a pdf attachment , false otherwise
    Content : A Content type object
    """
    def __init__(self, message):
        self.Source = message.get('Source-File', '')
        self.To = message.get('To', '')
        self.From = message.get('From', '')
        self.Subject = message.get('Subject', '')
        self.Content = self._get_content(message)
        self.has_attachment = message.get('Attachment', False)

    @staticmethod
    def _get_content(message):
        html = []
        text = []
        for part in message['Content']:
            if part['type'] == 'text/html':
                html.append(part['content'])
            elif part['type'] == 'text/plain':
                text.append(part['content'])
        return Content(text, html)

    def __eq__(self, other):
        return type(self) == type(other) and self.Content == other.Content and self.Source == other.Source \
            and self.To == other.To and self.From == other.From and self.Subject == other.Subject