"""
正则表达式的NLU解析模型，会同时返回domain, intent, slots

NLU_LOG_LEVEL=debug python3 -m nlu.engine.regex_engine

"""

import re
from sklearn.utils import shuffle
from nlu.log import LOG
from nlu.engine.engine_core import EngineCore

LIMIT = 1000

class RegexItem(object):
    """根据某条intent数据和实体列表，构建一个用于判断intent、domain、slot的对象"""

    slot_name_splitor = '___'

    def __init__(self, intent, index_entities_regex):
        """初始化"""

        assert isinstance(intent['intent'], str), '错误的意图'
        assert intent['intent'].strip(), '意图不能为空'
        self.intent = intent['intent'].strip()
        self.domain = None \
            if not isinstance(intent['domain'], str) or \
                len(intent['domain'].strip()) <= 0 \
            else intent['domain'].strip()
        self.data = intent['data']

        slot_index = {}
        def _replace(part):
            """转换部分句子结构，如果这个部分是实体，就返回正则表达式，如果是普通文本，就返回文本"""
            if 'name' in part:
                slot_name = part['name']
                if slot_name not in slot_index:
                    slot_index[slot_name] = 0
                slot_index[slot_name] += 1
                temp = '(?P<{slot_name}{splitor}{index}>{slot_regex})'
                if slot_name in index_entities_regex:
                    return temp.format(
                        slot_name=slot_name,
                        splitor=self.slot_name_splitor,
                        index=slot_index[slot_name],
                        slot_regex=index_entities_regex[slot_name])
                # else:
                return temp.format(
                    slot_name=slot_name,
                    slot_regex=clean_re(part['text']))
            return clean_re(part['text'])

        self.patten = re.compile(
            '^' + \
            ''.join([_replace(x) for x in self.data]) + \
            '$')
        LOG.debug('pattens: %s', self.patten)

    def predict(self, text):
        """预测一条文本，给出domain，intent，slots"""
        indexgroup = {v: k for k, v in self.patten.groupindex.items()}
        sentence_match = self.patten.match(text)
        if sentence_match is None:
            return None
        ret = {
            'slots': [],
            'intent': self.intent,
            'domain': self.domain
        }
        for i, g in enumerate(sentence_match.groups()):
            if g is None:
                continue
            slot_name = indexgroup[i + 1]
            slot_value = sentence_match.group(i + 1)
            reg = sentence_match.regs[i + 1]

            ret['slots'].append({
                'slot_value': slot_value,
                'slot_name': slot_name.split(self.slot_name_splitor)[0],
                'pos': reg})
        return ret

def clean_re(x):
    """去掉特殊字符，下面的地址中包含所有特殊字符
    Here’s a complete list of the metacharacters;
    their meanings will be discussed in the rest of this HOWTO.
    https://docs.python.org/3/howto/regex.html
    """
    l = ['.', '^', '$', '*', '+', '?', '{' '}', '[' ']', '\\', '|', '(', ')']
    for ll in l:
        x = x.replace(ll, '\\' + ll)
    return x

class RegexEngine(EngineCore):
    """包含多个RegexItem
    注意文本都会变小写"""

    items = []

    def __init__(self, intents, entities):
        """初始化Regex识别引擎"""
        super(RegexEngine, self).__init__(
            domain_implement=True,
            intent_implement=True,
            slot_implement=True)

        index_entities_regex = self.get_index_entities_regex(entities)

        print('intents', len(intents))
        ret = []
        for intent in intents:
            print('intent', intent)
            r = RegexItem(intent, index_entities_regex)
            ret.append(r)

        self.items = ret

    def predict_domain(self, nlu_obj):
        """预测领域"""
        text = nlu_obj['text']
        ret = self.predict(text)
        if ret is not None:
            nlu_obj['intent'] = ret['intent']
            nlu_obj['domain'] = ret['domain']
            nlu_obj['slots'] = ret['slots']
            nlu_obj['regex_engine'] = ret
        nlu_obj['regex_engine'] = None
        return nlu_obj

    def predict_intent(self, nlu_obj):
        """预测意图"""
        return self.predict_domain(nlu_obj)

    def predict_slot(self, nlu_obj):
        """预测实体"""
        return self.predict_domain(nlu_obj)

    def predict(self, text):
        """预测一个文本"""
        text = text.lower()
        for item in self.items:
            p = item.predict(text)
            if p is not None:
                return p
        return None

    def get_index_entities_regex(self, entities):
        """将实体列表转换为正则表达式
        """
        ret = {}
        for x in entities:
            assert 'entity' in x and isinstance(x['entity'], str), \
                '实体必须有entity属性且为字符串类型'
            data = []
            for item in x['data']:
                if isinstance(item, str):
                    data.append(item)
                elif isinstance(item, list):
                    for iitem in item:
                        if isinstance(iitem, str):
                            data.append(iitem)
            if len(data) > LIMIT:
                data = shuffle(data, random_state=0)
                data = data[:LIMIT]
            data = [
                clean_re(x)
                for x in data
            ]
            r = '(?:' + '|'.join(data) + ')'
            if 'regex' in x:
                r += '|(?:' + x['regex'] + ')'
            regex = '\\s*(?:' + r + ')\\s*'
            ret[x['entity']] = regex
            LOG.debug('entity: %s regex: %s', x['entity'], regex)
        return ret


def unit_test():
    """unit test"""

    from nlu.utils.data_loader import load_nlu_data

    intents, entities = load_nlu_data('nlu_data')

    reng = RegexEngine(intents, entities)

    for intent in intents:
        text = ''.join(x['text'] for x in intent['data'])
        print(text)
        print(reng.predict(text))
        print('---' * 10)

if __name__ == '__main__':
    unit_test()
