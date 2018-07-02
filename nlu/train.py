"""训练NLU"""

import os
import json
import pickle
from nlu.engine.regex_engine import RegexEngine
from nlu.engine.ml_intent_classifier import MLIntentClassifier
from nlu.engine.dl_intent_classifier import DLIntentClassifier
from nlu.engine.neural_intent_classifier_slot_filler import NeuralIntentClassifierSlotFiller
from nlu.engine.crf_slot_filler import CRFSlotFiller
from nlu.engine.neural_slot_filler import NeuralSlotFiller
from nlu.utils.data_iob import data_to_iob
from nlu.utils.data_loader import load_nlu_data
from nlu.log import LOG

def pipline():
    """训练pipline"""

    nlu_data = './nlu_data'
    model_dir = './tmp/nlu_model'

    pipline_config = [
        # 'regex_engine',
        # 'ml_intent_classifier',
        # 'crf_slot_filler',
        'dl_intent_classifier',
        # 'neural_slot_filler',
        # 'neural_intent_classifier_slot_filler',
    ]

    build_model(nlu_data, model_dir, pipline_config)


def build_model(nlu_data, model_dir, pipline_config):
    """构建模型"""
    models = []

    LOG.info('start build')

    intents, entities = load_nlu_data(nlu_data)
    iob = [None, None, None]
    def _get_iob(iob):
        """load iob only once"""
        if iob[0] is None:
            LOG.info('build IOB data')
            (sentence_result,
             slot_result,
             domain_result) = data_to_iob(intents, entities)
            iob = sentence_result, slot_result, domain_result
        return iob

    for item in pipline_config:
        LOG.info('train "%s"', item)

        if item == 'regex_engine':
            reng = RegexEngine(intents, entities)
            models.append(('regex_engine', reng))

        elif item == 'ml_intent_classifier':
            ml_intent = MLIntentClassifier()
            iob = _get_iob(iob)
            sentence_result, _, domain_result = iob
            ml_intent.fit(sentence_result, domain_result)
            models.append(('ml_intent_classifier', ml_intent))

        elif item == 'dl_intent_classifier':
            dl_intent = DLIntentClassifier()
            iob = _get_iob(iob)
            sentence_result, _, domain_result = iob
            dl_intent.fit(sentence_result, domain_result)
            models.append(('dl_intent_classifier', dl_intent))

        elif item == 'crf_slot_filler':
            crf_slot = CRFSlotFiller()
            iob = _get_iob(iob)
            sentence_result, slot_result, _ = iob
            crf_slot.fit(sentence_result, slot_result)
            models.append(('crf_slot_filler', crf_slot))

        elif item == 'neural_slot_filler':
            crf_slot = NeuralSlotFiller()
            iob = _get_iob(iob)
            sentence_result, slot_result, _ = iob
            crf_slot.fit(sentence_result, slot_result)
            models.append(('neural_slot_filler', crf_slot))

        elif item == 'neural_intent_classifier_slot_filler':
            nicsf = NeuralIntentClassifierSlotFiller()
            iob = _get_iob(iob)
            sentence_result, slot_result, domain_result = iob
            y_data = list(zip(slot_result, domain_result))
            nicsf.fit(sentence_result, y_data)
            models.append(('neural_intent_classifier_slot_filler', nicsf))

        else:
            LOG.error('invalid engine "%s"', item)
            raise Exception('invalid engine "%s"' % item)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as fp:
        json.dump(pipline_config, fp, indent=4, ensure_ascii=False)

    for model_name, model in models:
        model_path = os.path.join(model_dir, '{}.pkl'.format(model_name))
        with open(model_path, 'wb') as fp:
            pickle.dump(model, fp)

    LOG.info('train and saved')

if __name__ == '__main__':
    pipline()
