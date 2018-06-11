"""
正则表达式的NLU解析模型，会同时返回domain, intent, slots

NLU_LOG_LEVEL=debug python3 -m nlu.engine.regex_engine

"""

import re
from nlu.utils.data_utils import DOMAIN_SPLITOR
from nlu.utils.data_utils import SLOT_SPLITOR
from nlu.utils.data_utils import SPLITOR

class RegexItem(object):
    """根据某条intent数据和实体列表，构建一个用于判断intent、domain、slot的对象"""

    def __init__(self, intent, item, index_entities_regex):
        """初始化"""

        item = item.lower()

        self.intent = intent
        self.item = item
        domain_ret = DOMAIN_SPLITOR.split(item)
        if len(domain_ret) == 1:
            self.domains = ['None' + SPLITOR + intent]
            self.domain_sentences = [domain_ret[0]]
        else:
            domain_ret = domain_ret[1:]
            self.domains = [
                domain_ret[x] if SPLITOR in domain_ret[x] else domain_ret[x] + SPLITOR + intent
                for x in range(0, len(domain_ret), 2)
            ]
            self.domain_sentences = [domain_ret[x] for x in range(1, len(domain_ret), 2)]
        
        pattens = []
        domain_pattens = []
        for domain, sentence in zip(self.domains, self.domain_sentences):
            domain_intent_name = domain if domain is not None else ''
            if SPLITOR not in domain_intent_name:
                # domain only, concat intent
                domain_intent_name = domain_intent_name + SPLITOR + intent
            count = {}
            def _make_replace(with_name):
                def _replace(pattern):
                    slot_name = pattern.group(1)

                    if with_name:
                        temp = '(?P<{slot_name}>{slot_regex})'
                        if slot_name not in count:
                            count[slot_name] = 0
                        else:
                            count[slot_name] += 1
                        slot_name_intead = slot_name + SPLITOR + str(count[slot_name])
                        return temp.format(
                            slot_name=slot_name_intead,
                            slot_regex=index_entities_regex[slot_name]
                        )
                    else:
                        temp = '(?:{slot_regex})'
                        return temp.format(
                            slot_regex=index_entities_regex[slot_name]
                        )
                return _replace
            p = SLOT_SPLITOR.sub(_make_replace(True), sentence)
            pattens.append(re.compile('^' + p + '$'))
            p = SLOT_SPLITOR.sub(_make_replace(False), sentence)
            domain_pattens.append('(?P<{domain}>{p})'.format(domain=domain, p=p))
        self.pattens = pattens
        self.domain_patten = re.compile('^' + '\\s*'.join(domain_pattens) + '$')
    
    
    def predict(self, text):
        """预测一条文本，给出domain，intent，slots"""
        group_match = self.domain_patten.match(text)

        if group_match is None:
            return None
        print(self.domain_patten)
        ret = {
            'domains': [x.split(SPLITOR)[0] for x in self.domains],
            'intents': [x.split(SPLITOR)[1] for x in self.domains],
            'domains_pos': [],
            'intents_pos': [],
            'slots': []
        }
        groups = group_match.groups()
        groups = [x for x in groups if x is not None]
        base = 0
        for domain, group, patten in zip(self.domains, groups, self.pattens):
            ret['domains_pos'].append((
                base, base + len(group)
            ))
            ret['intents_pos'].append((
                base, base + len(group)
            ))
            indexgroup = {v: k for k, v in patten.groupindex.items()}
            sentence_match = patten.match(group)
            for i, g in enumerate(sentence_match.groups()):
                if g is None:
                    continue
                slot_name = indexgroup[i + 1].split(SPLITOR)[0]
                slot_value = sentence_match.group(i + 1)
                reg = sentence_match.regs[i + 1]

                ret['slots'].append({
                    'slot_value': slot_value,
                    'slot_name': slot_name,
                    'pos': reg,
                    'domain': domain.split(SPLITOR)[0],
                    'intent': domain.split(SPLITOR)[1],
                })
                
            base += len(group)
        return ret


class RegexEngine(object):
    """包含多个RegexItem
    注意文本都会变小写"""
    
    items = []

    def __init__(self, intents, entities):

        index_entities_regex = get_index_entities_regex(entities)

        ret = []
        for intent in intents:
            for item in intent['data']:
                r = RegexItem(intent['intent'], item, index_entities_regex)
                ret.append(r)
        
        self.items = ret
    
    def pipeline(self, nlu_obj):
        text = nlu_obj['text']
        ret = self.predict(text)
        if ret is not None:
            nlu_obj['intents'] = ret['intents']
            nlu_obj['domains'] = ret['domains']
            nlu_obj['slots'] = ret['slots']
            nlu_obj['regex_engine'] = ret
        return nlu_obj
    
    def predict(self, text):
        text = text.lower()
        for item in self.items:
            p = item.predict(text)
            if p is not None:
                return p


def get_index_entities_regex(entities):
    ret = {}
    for x in entities:
        data = []
        for item in x['data']:
            if isinstance(item, str):
                data.append(item)
            elif isinstance(item, dict):
                k = list(item.keys())[0]
                data += item[k]
        # print('data', data)
        r = '(?:' + '|'.join(data) + ')'
        if 'regex' in x:
            r += '|(?:' + x['regex'] + ')'
        ret[x['entity']] = '\\s*(?:' + r + ')\\s*'
    return ret


def unit_test():
    """unit test"""

    from nlu.utils.data_loader import load_nlu_data

    intents, entities = load_nlu_data('nlu_data')

    intents = [x for x in intents if x['intent'] == 'lottery_inform']
    # print(intents)

    reng = RegexEngine(intents, entities)

    for intent in intents:
        for item in intent['data']:
            text = DOMAIN_SPLITOR.sub('', SLOT_SPLITOR.sub('\\2', item))
            print(text)
            print(reng.predict(text))
            print('---' * 10)

if __name__ == '__main__':
    unit_test()
