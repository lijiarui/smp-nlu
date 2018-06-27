"""

NLU_LOG_LEVEL=debug python3 -m nlu.utils.data_iob
"""

import re
import numpy as np
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from nlu.log import LOG
from nlu.utils.data_utils import SPLITOR
from nlu.utils.chinese_aug import chinese_aug

def get_index_entities_data(entities):
    """转换实体列表到索引"""
    ret = {}
    for x in entities:
        data = []
        for item in x['data']:
            if isinstance(item, str):
                data.append(item)
            elif isinstance(item, list):
                for iitem in item:
                    if isinstance(iitem, str):
                        data.append(iitem)
        if x['entity'] not in ret:
            ret[x['entity']] = []
        ret[x['entity']] += data
    for k, v in ret.items():
        ret[k] = shuffle(v)
    return ret


def fill_iob(slot_name, slot_value):
    """填充IOB格式
    call: fill_iob('date', '2018')
    result: ['B_date', 'I_date', 'I_date', 'I_date', 'I_date']
    Args:
        slot_name: 槽值名称
        slot_value: 槽值结果
    """
    b_tag = 'B_{}'.format(slot_name)
    i_tag = 'I_{}'.format(slot_name)
    return [b_tag] + (
        [i_tag] * (len(slot_value) - 1)
    )

def convert_item(intent, index_entities_data, slot_count):
    """转换一条"""

    slot_name_index = {}
    def _choice(slot_name):
        if slot_name not in slot_name_index:
            slot_name_index[slot_name] = 0
        if slot_name_index[slot_name] >= len(index_entities_data[slot_name]):
            slot_name_index[slot_name] = 0
        value = index_entities_data[
            slot_name
        ][slot_name_index[slot_name]]
        slot_name_index[slot_name] += 1
        return value

    sentence_results, slot_results, domain_results = [], [], []

    loop = [10]
    for item in intent['data']:
        if 'name' in item:
            slot_name = item['name']
            assert slot_name in index_entities_data
            loop.append(
                int(min(
                    500,
                    len(index_entities_data[slot_name])
                    )) # / slot_count[slot_name])
            )
    
    loop = max(loop)
    llen = len(re.findall(r'\|', ' '.join([x['text'] for x in intent['data']])))
    if llen > 0:
        loop *= llen

    for _ in range(loop):

        sentence_result = []
        slot_result = []

        for i, item in enumerate(intent['data']):
            is_tail = (i == len(intent['data']) - 1)
            if 'name' in item:
                slot_name = item['name']
                slot_value = _choice(slot_name)

                sentence_result += list(slot_value)
                slot_result += fill_iob(slot_name, slot_value)
                if is_tail:
                    tail = chinese_aug('', en_tail=True)
                    if tail:
                        sentence_result += list(tail)
                        slot_result += ['O'] * len(tail)
            else:
                text = item['text']
                # 转换 [[要|想要]] => 随机一个
                text = re.sub(
                    r'\[\[([^\]]+)\]\]',
                    lambda x: np.random.choice(x.group(1).split('|')),
                    text)
                # 文字尾部强化
                if is_tail:
                    text = chinese_aug(text, en_tail=True)
                else:
                    text = chinese_aug(text, en_tail=False)
                sentence_result += list(text)
                slot_result += ['O'] * len(text)

        domain_result = str(intent['domain']) + SPLITOR + str(intent['intent'])

        # if len(intent['data']) == 2 and 'B_date' in slot_result:
        #     print(sentence_result)
        #     print(slot_result)

        sentence_results.append(sentence_result)
        slot_results.append(slot_result)
        domain_results.append(domain_result)

    return sentence_results, slot_results, domain_results


def data_to_iob(intents, entities):
    """把数据转换为IOB格式
    Inside-outside-beginning"""

    np.random.seed(0)

    index_entities_data = get_index_entities_data(entities)
    keys = sorted([(k, len(v)) for k, v in index_entities_data.items()], key=lambda x: x[1])
    for k, v in keys:
        LOG.debug('kv %s %s', k, v)

    slot_count = {}
    for intent in intents:
        for item in intent['data']:
            if 'name' in item:
                slot_name = item['name']
                if slot_name not in slot_count:
                    slot_count[slot_name] = 0
                slot_count[slot_name] += 1

    sentence_result, slot_result, domain_result = [], [], []

    LOG.debug('parallel job %s', len(intents))
    ret = Parallel(n_jobs=8, verbose=6)(
        delayed(convert_item)(intent, index_entities_data, slot_count)
        for intent in intents)

    LOG.debug('parallel job done')

    for r1, r2, r3 in ret:
        sentence_result += r1
        slot_result += r2
        domain_result += r3

    with open('/tmp/nlu_iob.txt', 'w') as fp:
        for a, b, c in zip(sentence_result, slot_result, domain_result):
            fp.write('\t'.join(a) + '\n')
            fp.write('\t'.join(b) + '\n')
            fp.write(c + '\n')
            fp.write('\n')
    
    LOG.debug('return IOB data')
    return sentence_result, slot_result, domain_result


def unit_test():
    """unit test"""
    from nlu.utils.data_loader import load_nlu_data

    intents, entities = load_nlu_data('nlu_data')

    sentence_result, slot_result, domain_result = data_to_iob(intents, entities)
    for s, ss, d in zip(sentence_result, slot_result, domain_result):
        print(s)
        print(ss)
        print(d)
        print('---' * 10)


if __name__ == '__main__':
    unit_test()
