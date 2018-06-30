"""RESTful"""

import os
import json
import time
import pickle
import simplejson
from flask import Flask, jsonify
from flask_cors import CORS
from nlu.log import LOG

def load_models(model_dir='./tmp/nlu_model'):
    """加载模型"""
    config_path = os.path.join(model_dir, 'config.json')

    if not os.path.exists(config_path):
        LOG.error('config_path not exsits "%s"', config_path)
        exit(1)
    pipeline_config = json.load(open(config_path))
    models = []
    for model_name in pipeline_config:
        model = pickle.load(
            open(os.path.join(
                model_dir,
                '{}.pkl'.format(model_name)), 'rb'))
        models.append((model_name, model))
    return models

APP = Flask('NLU RESTful', static_folder='nlu/www')
CORS(APP)
MODELS = load_models()

@APP.route('/')
def web_root():
    """根目录，测试用"""
    return APP.send_static_file('index.html')

@APP.route('/parse/<sentence>')
def web_parse(sentence=None):
    """提供NLU服务"""
    if sentence is None:
        return jsonify(success=False, message='sentence is None')

    nlu_obj = {
        'intent': None,
        'domain': None,
        'slots': [],
        'text': sentence,
        'tokens': list(sentence),
    }

    start_time = time.time()

    LOG.debug('start %s models', len(MODELS))
    for model_name, model in MODELS:
        LOG.debug('through %s model %s', model_name, time.time() - start_time)
        if model.domain_implement:
            LOG.debug('through %s model predict_domain %s', model_name, time.time() - start_time)
            nlu_obj = model.predict_domain(nlu_obj)
        if model.intent_implement:
            LOG.debug('through %s model predict_intent %s', model_name, time.time() - start_time)
            nlu_obj = model.predict_intent(nlu_obj)
        if model.slot_implement:
            LOG.debug('through %s model predict_slot %s', model_name, time.time() - start_time)
            nlu_obj = model.predict_slot(nlu_obj)

    # print(nlu_obj)
    LOG.debug('return %s', time.time() - start_time)
    return APP.response_class(
        response=simplejson.dumps({
            'success': True,
            'result': nlu_obj
        }, indent=4, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )
