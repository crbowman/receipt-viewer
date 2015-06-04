import os
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/annotate')
def annotate():
    return render_template('product.html')

@app.route('/'):
def home():
	return render_template('home.html', my_string='home template', my_list['receipt 1', 'receipt 2', 'receipt 3'])