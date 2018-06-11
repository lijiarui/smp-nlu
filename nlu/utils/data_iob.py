"""

NLU_LOG_LEVEL=debug python3 -m nlu.utils.data_iob
"""

import re
import numpy as np
from joblib import Parallel, delayed
from nlu.utils.data_utils import DOMAIN_SPLITOR
from nlu.utils.data_utils import SLOT_SPLITOR
from nlu.utils.data_utils import SPLITOR
from nlu.log import LOG
from nlu.utils.chinese_aug import chineseAug

def get_index_entities_data(entities):
    ret = {}
    for x in entities:
        data = []
        for item in x['data']:
            if isinstance(item, str):
                data.append(item)
            elif isinstance(item, dict):
                k = list(item.keys())[0]
                data += item[k]
        ret[x['entity']] = data
    return ret


def fill_iob(slot_name, slot_value):
    b_tag = 'B_{}'.format(slot_name)
    i_tag = 'I_{}'.format(slot_name)
    return [b_tag] + (
        [i_tag] * (len(slot_value) - 1)
    )

def convert_item(intent, item, index_entities_data):

    domain_ret = DOMAIN_SPLITOR.split(item)
    if len(domain_ret) == 1:
        domains = ['None' + SPLITOR + intent]
        domain_sentences = [domain_ret[0]]
    else:
        domain_ret = domain_ret[1:]
        domains = [
            domain_ret[x] if SPLITOR in domain_ret[x] else domain_ret[x] + SPLITOR + intent
            for x in range(0, len(domain_ret), 2)
        ]
        domain_sentences = [domain_ret[x] for x in range(1, len(domain_ret), 2)]

    sentence_result = []
    slot_result = []
    domain_result = []

    for domain, sentence in zip(domains, domain_sentences):

        one_sentence_result = []
        one_slot_result = []
        one_domain_result = []

        domain_intent_name = domain if domain is not None else ''
        if SPLITOR not in domain_intent_name:
            # domain only, concat intent
            domain_intent_name = domain_intent_name + SPLITOR + intent

        num = 1
        slots = SLOT_SPLITOR.findall(sentence)
        for slot_name, _ in slots:
            n = len(index_entities_data[slot_name])
            n = min(n, 5)
            num *= n
        
        for _ in range(num):

            placeholder = []
            placeholder_values = []
            def _replace(pattern):
                slot_name = pattern.group(1)
                choice_data = np.random.choice(index_entities_data[slot_name])

                placeholder.append(slot_name)
                placeholder_values.append(choice_data)
                temp = '{slot_example}'
                return temp.format(
                    slot_example='_____{}_____'.format(len(placeholder))
                )
                
            p = SLOT_SPLITOR.sub(_replace, sentence)
            p = chineseAug(p)
            # print(p)
            
            pices = re.split(r'_____\d+_____', p)
            assert len(pices) == len(placeholder) + 1, 'pices {} != placeholder - 1 {}'.format(
                len(pices), len(placeholder) - 1
            )
            
            sentence_part = []
            slot_part = []
            for i in range(len(pices)):
                sentence_part += list(pices[i])
                slot_part += ['O'] * len(pices[i])
                if i < len(placeholder):
                    p = placeholder[i]
                    pv = placeholder_values[i]
                    sentence_part += list(pv)
                    slot_part += fill_iob(p, pv)

            domain_part = ['B_{}'.format(domain_intent_name)] + [
                'I_{}'.format(domain_intent_name)
            ] * (len(sentence_part) - 1)

            one_sentence_result.append(sentence_part)
            one_slot_result.append(slot_part)
            one_domain_result.append(domain_part)
        
        sentence_result.append(one_sentence_result)
        slot_result.append(one_slot_result)
        domain_result.append(one_domain_result)

    ret_sentence_result = [concat_list(x) for x in zip(*sentence_result)]
    ret_slot_result = [concat_list(x) for x in zip(*slot_result)]
    ret_domain_result = [concat_list(x) for x in zip(*domain_result)]

    return ret_sentence_result, ret_slot_result, ret_domain_result

def concat_list(x):
    """[[a], [b], [c]] => [a, b, c]
    """
    r = []
    for xx in x:
        r += xx
    return r


def data_to_iob(intents, entities):
    """把数据转换为IOB格式
    Inside-outside-beginning"""

    np.random.seed(0)

    index_entities_data = get_index_entities_data(entities)

    sentence_result, slot_result, domain_result = [], [], []

    quest = []
    for intent in intents:
        for item in intent['data']:
            quest.append((intent['intent'], item, index_entities_data))
    
    LOG.debug('parallel job {}'.format(len(quest)))
    ret = Parallel(n_jobs=8, verbose=6)(
        delayed(convert_item)(p1, p2, p3)
        for p1, p2, p3 in quest
    )

    LOG.debug('parallel job done')

    for r1, r2, r3 in ret:
        sentence_result += r1
        slot_result += r2
        domain_result += r3
    
    LOG.debug('return IOB data')
    
    return sentence_result, slot_result, domain_result


def unit_test():
    """unit test"""
    from nlu.utils.data_loader import load_nlu_data

    intents, entities = load_nlu_data('nlu_data')

    intents = [x for x in intents if x['intent'] == 'lottery_inform']
    # print(intents)

    sentence_result, slot_result, domain_result = data_to_iob(intents, entities)
    for s, ss, d in zip(sentence_result, slot_result, domain_result):
        print(s)
        print(ss)
        print(d)
        print('---' * 10)


if __name__ == '__main__':
    unit_test()