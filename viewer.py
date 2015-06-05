import os
import re
from json import dumps
from ner_product import auto_tag_receipt	
from sbx_utils import files_in
from data import AnnotatedReader												
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)


@app.route('/annotate', methods=['POST'])
def annotate():
    json = request.get_json(force=True)
    print(json)
    file_name = json.get('filename')
    folder = json.get('folder')
    products = json.get('product_names')
    if not file_name or not products or len(products) == 0:
        return dumps({'BAD REQUEST': 'filename or product names missing'}), 401
    reader = AnnotatedReader(folder + "/" + file_name)
    data = reader.data_as_tuple()
    if len(data) == 0:
        return dumps({'INTERNAL ERROR': 'AnnotatedReader data list is empty'}), 500
    with open(folder + '/' + file_name) as old_data_file:
        old_data = old_data_file.read()
    with open(folder + "/" + file_name, 'w') as out_file:
        try:
            written = auto_tag_receipt(data, out_file, products)
        except Exception as e:
            out_file.write(old_data)
            return dumps({'INTERNAL ERROR': traceback.format_exc()}), 500
        if not written:
            out_file.write(old_data)
            return dumps({'INTERNAL ERROR': traceback.format_exc()}), 500
    return dumps({'status': 'OK'}), 200

@app.route('/home', methods=['GET'])
def extract():
    try:
        return load_template(), 200
    except Exception as e:
        print(e)
        return dumps({'error':'Unexpected error occured'})

@app.route('/pdf')
def get_pdf():
    file_name = request.args.get('name', '')
    folder_name = request.args.get('folder', '')
    if file_name.endswith('.txt'):
        file_name = file_name.split('.')[0] + ".eml.pdf"
    return send_from_directory(folder_name, file_name)

@app.route('/file', methods=['GET'])
def get_file():
    file_name = request.args.get('name', '')
    folder_name = request.args.get('folder', '')
    lst = []
    try:
        annotated_text = load_file(folder_name, file_name)
        lines = annotated_text.splitlines()
        for line in lines:
            tuple = line.split(' ')
            if len(tuple) == 3:
                lst.append({'value': tuple[0], 'pos': tuple[1], 'iob': tuple[2]})
        return dumps(lst), 200
    except Exception as e:
        print(e)
        return dumps({'error':'Unexpected error occured'})

@app.route('/files', methods=['GET'])
def get_files():
    folder = request.args.get('folder', '')
    try:
        return dumps(files_in(folder)), 200
    except Exception as e:
        print(e)
        return dumps({'error':'Unexpected error occured'})

def load_template():
    with open('templates/product.html') as template:
        return template.read()

def load_file(folder, file_name):
    with open(folder + '/' + file_name) as f:
        return f.read()

if __name__ == '__main__':
	app.run()