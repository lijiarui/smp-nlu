"""RESTful"""

import os
import json
import pickle
import simplejson
from flask import Flask, jsonify
from flask_cors import CORS
from nlu.log import LOG

def load_models(model_dir='./tmp/nlu_model'):
    config_path = os.path.join(model_dir, 'config.json')

    if not os.path.exists(config_path):
        LOG.error('config_path not exsits "{}"'.format(config_path))
        exit(1)
    pipeline_config = json.load(open(config_path))
    models = []
    for model_name in pipeline_config:
        model = pickle.load(open(os.path.join(model_dir, '{}.pkl'.format(model_name)), 'rb'))
        models.append((model_name, model))
    return models

APP = Flask('NLU RESTful', static_folder='nlu/www')
CORS(APP)
MODELS = load_models()

@APP.route('/')
def web_root():
    return APP.send_static_file('index.html')

@APP.route('/parse/<sentence>')
def web_parse(sentence=None):
    if sentence is None:
        return jsonify(success=False, message='sentence is None')

    nlu_obj = {
        'intents': [None],
        'domains': [None],
        'domains_pos': [],
        'intents_pos': [],
        'slots': [],
        'text': sentence,
        'tokens': list(sentence),
    }

    for _, model in MODELS:
        nlu_obj = model.pipeline(nlu_obj)
    
    # return jsonify(
    #     success=True,
    #     result=nlu_obj
    # )

    return APP.response_class(
        response=simplejson.dumps({
            'success': True,
            'result': nlu_obj
        }, indent=4, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )