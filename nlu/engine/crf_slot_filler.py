"""
使用sklearn-crfsuite进行NER识别

NLU_LOG_LEVEL=debug python3 -m nlu.engine.crf_slot_filler

"""

import os
import re
import scipy
import numpy as np
from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from nlu.log import LOG
from nlu.engine.engine_core import EngineCore

def single_sentence_to_features(sentence):
    sentence = [x.lower() for x in sentence]
    features = []
    for i in range(len(sentence)):
        token = sentence[i]
        token_prev_3 = '_' if i - 3 < 0 else sentence[i - 3]
        token_prev_2 = '_' if i - 2 < 0 else sentence[i - 2]
        token_prev = '_' if i == 0 else sentence[i - 1]
        token_next = '_' if i == (len(sentence) - 1) else sentence[i + 1]
        token_next_2 = '_' if (i + 2) >= len(sentence) else sentence[i + 2]
        token_next_3 = '_' if (i + 3) >= len(sentence) else sentence[i + 3]
        feature = {
            'bias': 1.0,
            'BOS': True if i == 0 else False,
            'EOS': True if i == (len(sentence) - 1) else False,
            'token-3': token_prev_3,
            'token-2': token_prev_2,
            'token-1': token_prev,
            'token': token,
            'token+1': token_next,
            'token+2': token_next_2,
            'token+3': token_next_3,
            'token-3:token-2': token_prev_3 + token_prev_2,
            'token-2:token-1': token_prev_2 + token_prev,
            'token-1:token': token_prev + token,
            'token:token+1': token + token_next,
            'token+1:token+2': token_next + token_next_2,
            'token+2:token+3': token_next_2 + token_next_3,
            'isEnglishChar': True if re.match(r'[a-zA-Z]', token) else False,
            'isNumberChar': True if re.match(r'[0-9]', token) else False,
            'isChineseNumberChar': True if re.match(r'[一二三四五六七八九十零俩仨]', token) else False,
            'isChineseChar': True if re.match(r'[\u4e00-\u9ffff]', token) else False,
            'isChinesePunctuationChar': True if re.match(r'[，。？！：；《》]', token) else False,
            'isChinesePunctuationChar-1': True if re.match(r'[，。？！：；《》]', token_prev) else False,
            'isChinesePunctuationChar+1': True if re.match(r'[，。？！：；《》]', token_next) else False,
        }
        features.append(feature)
    return features


def sentences_to_features(sentence_result):
    x_train = []
    for s in sentence_result:
        x_train.append(single_sentence_to_features(s))
    return x_train

def get_slots(sentence, slot):
    current = None
    current_str = []
    ret = []
    for s, ss in zip(sentence, slot):
        if ss != 'O':
            ss = ss[2:]
            if current is None:
                current = ss
                current_str = [s]
            else:
                if current == ss:
                    current_str.append(s)
                else:
                    ret.append((current, ''.join(current_str)))
                    current = ss
                    current_str = []
        else:

            # 应对 B1 O B1 的情况，B1和B1很可能是连续的，而O是空格
            if (s == ' ' or s == '　'):
                continue

            if current is not None:
                ret.append((current, ''.join(current_str)))
                current = None
                current_str = []

    if current is not None:
        ret.append((current, ''.join(current_str)))
        
    ret_dict = {}
    for s, v in ret:
        if s not in ret_dict:
            ret_dict[s] = []
        ret_dict[s].append(v)
    return ret_dict


def get_slots_detail(sentence, slot):
    """
    example:
    sentence == ['买', '2', '手']
    slot == ['O', 'B_number', 'O']
    """
    current = None
    current_str = []
    ret = []
    for i, (s, ss) in enumerate(zip(sentence, slot)):
        if ss != 'O':
            ss = ss[2:]
            if current is None:
                current = ss
                current_str = [s]
            else:
                if current == ss:
                    current_str.append(s)
                else:
                    ret.append((current, ''.join(current_str), i - len(current_str), i))
                    current = ss
                    current_str = [s]
        else:
            if current is not None:
                ret.append((current, ''.join(current_str), i - len(current_str), i))
                current = None
                current_str = []

    if current is not None:
        ret.append((current, ''.join(current_str), i - len(current_str), i))
        
    ret_list = []
    for s, v, start, end in ret:
        ret_list.append({
            'slot_name': s,
            'slot_value': v,
            'pos': (start, end)
        })
    return ret_list


def get_exact_right(slot_true, slot_pred):
    import json
    for s, v in slot_true.items():
        if s not in slot_pred:
            return False
        v = json.dumps(v)
        vp = json.dumps(slot_pred[s])
        if v != vp:
            return False
    return True

class CRFSlotFiller(EngineCore):
    """注意文本都会变小写"""

    def __init__(self):
        """初始化"""
        super(CRFSlotFiller, self).__init__(
            domain_implement=False,
            intent_implement=False,
            slot_implement=True)
        self.crf = None
    
    def fit(self,
            sentence_result, slot_result,
            max_iterations=100, c1=0.17, c2=0.01):
        """fit model"""

        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations

        LOG.debug('fit CRFSlotFiller')

        x_train = sentences_to_features(sentence_result)
        y_train = slot_result
        labels = set()
        for x in slot_result:
            labels.update(x)
        labels = sorted(list(labels))
        labels.remove('O')
        LOG.debug('labels: {}'.format(', '.join(labels)))
        self.labels = labels

        try:
            LOG.debug('CRFSlotFiller try write tmp train data')
            with open('/tmp/crf_slot_filler.tmp', 'w') as fp:
                for x, y in zip(sentence_result, slot_result):
                    line = []
                    for i in range(len(x)):
                        line.append('{}\t{}'.format(x[i], y[i]))
                    fp.write('\n'.join(line) + '\n\n')
            LOG.debug('CRFSlotFiller try write tmp train data done')
        except:
            pass

        LOG.debug('x_train %d, y_train %d' % (len(x_train), len(y_train)))

        if os.environ.get('CRF') == 'search':
            crf = CRF(
                algorithm='lbfgs',
                max_iterations=50,
                all_possible_transitions=True
            )
            params_space = {
                'c1': scipy.stats.expon(scale=0.5),
                'c2': scipy.stats.expon(scale=0.05),
            }
            f1_score = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)
            rs = RandomizedSearchCV(
                crf,
                params_space,
                cv=3,
                verbose=1,
                n_jobs=2,
                n_iter=8*8,
                scoring=f1_score
            )
            rs.fit(x_train, y_train)
            LOG.debug('best params: {}'.format(rs.best_params_))
            LOG.debug('best cv score: {}'.format(rs.best_score_))
            self.crf = rs.best_estimator_
        else:

            crf = CRF(
                algorithm='lbfgs',
                c1=c1,
                c2=c2,
                max_iterations=max_iterations,
                all_possible_transitions=True
            )
            crf.fit(x_train, y_train)

            self.crf = crf

    def predict_slot(self, nlu_obj):
        tokens = nlu_obj['tokens']
        tokens = [x.lower() for x in tokens]
        ret = self.predict([tokens])
        LOG.debug('crf_slot_filler raw %s', ret)
        crf_ret = get_slots_detail(nlu_obj['tokens'], ret[0])
        nlu_obj['crf_slot_filler'] = crf_ret
        if len(nlu_obj['slots']) <= 0:
            nlu_obj['slots'] = crf_ret
        return nlu_obj
    
    def predict(self, sentence_result):
        assert self.crf is not None, 'model not fitted'

        x_test = sentences_to_features(sentence_result)
        y_pred = self.crf.predict(x_test)
        return y_pred

    def eval(self, sentence_result, slot_result):

        y_pred = self.predict(sentence_result)
        y_test = slot_result
        return {
            'precision': metrics.flat_precision_score(y_test, y_pred, average='weighted'),
            'recall': metrics.flat_recall_score(y_test, y_pred, average='weighted'),
            'f1': metrics.flat_f1_score(y_test, y_pred, average='weighted'),
            'accuracy': metrics.flat_accuracy_score(y_test, y_pred),
        }
    
    @staticmethod
    def cv_eval(sentence_result, slot_result, cv=5, max_iterations=100, c1=0.17, c2=0.01):
        """用cv验证模型"""

        x_train = sentences_to_features(sentence_result)
        y_train = slot_result
        f1_score = make_scorer(metrics.flat_f1_score, average='weighted')

        crf = CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )

        cv_result = cross_validate(
            crf, x_train, y_train,
            scoring=f1_score, cv=cv, verbose=10
        )

        for k, v in cv_result.items():
            print(k)
            print(np.mean(v))
            print(v)

    
    def exact_eval(self, sentence_result, slot_result):

        y_pred = self.predict(sentence_result)
        y_test = slot_result
        acc = 0
        bad = []
        for sent, real, pred in zip(sentence_result, y_test, y_pred):
            real_slot = get_slots(sent, real)
            pred_slot =  get_slots(sent, pred)
            a = get_exact_right(real_slot, pred_slot)
            acc += a
            if not a:
                bad.append((sent, real, pred, real_slot, pred_slot))
        acc /= len(sentence_result)
        return acc, bad
    

def unit_test():
    """unit test"""

    from nlu.utils.data_loader import load_nlu_data
    from nlu.utils.data_iob import data_to_iob
    intents, entities = load_nlu_data('nlu_data')
    # intents = [x for x in intents if x['intent'] == 'lottery_inform']
    # print(intents)
    sentence_result, slot_result, _ = data_to_iob(intents, entities)

    # print(max([len(x) for x in sentence_result]))

    CRFSlotFiller.cv_eval(sentence_result, slot_result, cv=5)
    exit(0)

    eng = CRFSlotFiller()
    eng.fit(sentence_result, slot_result)

    LOG.debug('crf fitted')

    metrics = eng.eval(sentence_result, slot_result)
    for k, v in metrics.items():
        print(k, v)
    
    acc, bad = eng.exact_eval(sentence_result, slot_result)
    print('exact acc', acc)
    print('bad count', len(bad))

    # print(eng.predict([list('我要买第18138期')]))

if __name__ == '__main__':
    unit_test()