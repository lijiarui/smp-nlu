"""
Neural Slot Filler
"""

import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_validate
from .ner import NER
from .data_utils import batch_flow_bucket
from .fake_data import generate
from .word_sequence import WordSequence
from nlu.engine.engine_core import EngineCore
from nlu.utils.data_utils import SPLITOR


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

            # 应对 B1 O B1 的情况，B1和B1很可能是连续的，而O是空格
            if (s == ' ' or s == '　'):
                continue

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


class NeuralIntentClassifierSlotFiller(EngineCore):
    """ use seq2seq """

    x_ws = None
    y_ws = None
    model_bytes = None
    tmp_dir = '/tmp'
    tmp_model_name = 'tmp_neural_slot_filler.ckpt'
    model = None
    model_bytes = None
    model_params = {}

    def __init__(self, n_epoch=10, batch_size=64, learning_rate=0.001,
                 hidden_units=32, embedding_size=32, max_decode_step=100,
                 bidirectional=True, cell_type='lstm', depth=2,
                 use_residual=False, use_dropout=True, dropout=0.4,
                 output_project_active='tanh', crf_loss=True,
                 use_gpu=False):
        super(NeuralIntentClassifierSlotFiller, self).__init__(
            domain_implement=True,
            intent_implement=True,
            slot_implement=True)

        self.model_params = {
            'n_epoch': n_epoch,
            'max_decode_step': max_decode_step,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'bidirectional': bidirectional,
            'cell_type': cell_type,
            'depth': depth,
            'use_residual': use_residual,
            'use_dropout': use_dropout,
            'dropout': dropout,
            'output_project_active': output_project_active,
            'hidden_units': hidden_units,
            'embedding_size': embedding_size,
            'crf_loss': crf_loss,
            'use_gpu': use_gpu,
        }
    
    def fit(self, sentence_result, y_data):
        """fit model"""
        use_gpu = self.model_params['use_gpu']
        batch_size = self.model_params['batch_size']
        n_epoch = self.model_params['n_epoch']
        # max_decode_step = self.model_params['max_decode_step']

        slot_result, domain_result = list(zip(*y_data))

        x_train = sentence_result
        y_train = slot_result

        y_target = [
            x
            for x in domain_result
        ]

        self.labels = labels = sorted(list(set(y_target)), key=lambda x: (len(x), x))
        self.index_label = index_label = {}
        self.label_index = label_index = {}
        for index, label in enumerate(labels):
            index_label[index] = label
            label_index[label] = index

        n_target = len(labels)
        self.model_params['n_target'] = n_target

        y_target = np.array([label_index[y] for y in y_target])

        # max_decode_step = max([len(x) for x in y_train])
        # self.model_params['max_decode_step'] = max_decode_step

        x_ws, y_ws = WordSequence(), WordSequence()
        x_ws.fit(x_train)
        y_ws.fit(y_train)

        self.x_ws = x_ws
        self.y_ws = y_ws

        steps = int(len(x_train) / batch_size) + 1

        self.config = config = tf.ConfigProto(
            device_count={
                'CPU': 0 if use_gpu else 1,
                'GPU': 1 if use_gpu else 0,
            },
            allow_soft_placement=True,
            log_device_placement=False
        )

        # tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = sess = tf.Session(config=config)

            self.model = model = NER(
                n_target=n_target,
                input_vocab_size=len(x_ws),
                target_vocab_size=len(y_ws),
                max_decode_step=self.model_params['max_decode_step'],
                batch_size=self.model_params['batch_size'],
                learning_rate=self.model_params['learning_rate'],
                bidirectional=self.model_params['bidirectional'],
                cell_type=self.model_params['cell_type'],
                depth=self.model_params['depth'],
                use_residual=self.model_params['use_residual'],
                use_dropout=self.model_params['use_dropout'],
                dropout=self.model_params['dropout'],
                output_project_active=self.model_params['output_project_active'],
                hidden_units=self.model_params['hidden_units'],
                embedding_size=self.model_params['embedding_size'],
                parallel_iterations=64,
                crf_loss=self.model_params['crf_loss']
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(1, n_epoch + 1):
                costs = []
                flow = batch_flow_bucket(
                    [x_train, y_train, y_target], [x_ws, y_ws, None], batch_size
                )
                bar = tqdm(range(steps),
                        desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl, yt, _ = next(flow)
                    cost = model.train(sess, x, xl, y, yl, yt)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f}'.format(
                        epoch,
                        np.mean(costs)
                    ))

            model.save(sess, os.path.join(
                self.tmp_dir,
                self.tmp_model_name
            ))

        model_files = [x for x in os.listdir(self.tmp_dir) if self.tmp_model_name in x]
        self.model_bytes = {}
        for model_file_name in model_files:
            path = os.path.join(self.tmp_dir, model_file_name)
            with open(path, 'rb') as fp:
                self.model_bytes[model_file_name] = fp.read()
                os.remove(path)
        
        # self.restore_model()
    
    def restore_model(self):
        if self.model_bytes is not None:
            for model_file_name in self.model_bytes:
                with open(os.path.join(self.tmp_dir, model_file_name), 'wb') as fp:
                    fp.write(self.model_bytes[model_file_name])
            
            # tf.reset_default_graph()
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = sess = tf.Session(config=self.config)
                self.model = model = NER(
                    n_target=self.model_params['n_target'],
                    input_vocab_size=len(self.x_ws),
                    target_vocab_size=len(self.y_ws),
                    max_decode_step=self.model_params['max_decode_step'],
                    batch_size=self.model_params['batch_size'],
                    learning_rate=self.model_params['learning_rate'],
                    bidirectional=self.model_params['bidirectional'],
                    cell_type=self.model_params['cell_type'],
                    depth=self.model_params['depth'],
                    use_residual=self.model_params['use_residual'],
                    use_dropout=self.model_params['use_dropout'],
                    dropout=self.model_params['dropout'],
                    output_project_active=self.model_params['output_project_active'],
                    hidden_units=self.model_params['hidden_units'],
                    embedding_size=self.model_params['embedding_size'],
                    crf_loss=self.model_params['crf_loss'],
                    parallel_iterations=64
                )
                init = tf.global_variables_initializer()
                sess.run(init)
                model.load(sess, os.path.join(self.tmp_dir, self.tmp_model_name))

                for model_file_name in self.model_bytes:
                    os.remove(os.path.join(self.tmp_dir, model_file_name))
    
    def __getstate__(self):
        return {
            'x_ws': self.x_ws,
            'y_ws': self.y_ws,
            'model_bytes': self.model_bytes,
            'model_params': self.model_params,
            'config': self.config,
            'index_label': self.index_label,
            'label_index': self.label_index,
            'domain_implement': self.domain_implement,
            'intent_implement': self.intent_implement,
            'slot_implement': self.slot_implement,
        }

    def __setstate__(self, state):
        self.domain_implement = state['domain_implement']
        self.intent_implement = state['intent_implement']
        self.slot_implement = state['slot_implement']
        
        self.x_ws = state['x_ws']
        self.y_ws = state['y_ws']
        self.model_bytes = state['model_bytes']
        self.model_params = state['model_params']
        self.config = state['config']
        self.index_label = state['index_label']
        self.label_index = state['label_index']

        # print('self.model_params', state['model_params'])
        # exit(1)
        self.restore_model()
    
    def get_params(self, deep=True):
        return self.model_params
    
    def set_params(self, n_epoch=10, batch_size=64, learning_rate=0.001,
                   hidden_units=64, embedding_size=64,
                   bidirectional=True, cell_type='lstm', depth=1,
                   use_residual=False, use_dropout=True, dropout=0.2,
                   output_project_active='tanh', crf_loss=True,
                   use_gpu=False):
        self.model_params = {
            # 'max_decode_step': max_decode_step,
            'n_epoch': n_epoch,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'bidirectional': bidirectional,
            'cell_type': cell_type,
            'depth': depth,
            'use_residual': use_residual,
            'use_dropout': use_dropout,
            'dropout': dropout,
            'output_project_active': output_project_active,
            'hidden_units': hidden_units,
            'embedding_size': embedding_size,
            'crf_loss': crf_loss,
        }


    def predict(self, sentence_result, progress=False):
        ret = []
        reti = []

        batch_size = self.model_params['batch_size']
        bar = range(0, len(sentence_result), batch_size)
        if progress:
            bar = tqdm(bar)
        for i in bar:
            batch = sentence_result[i : i + batch_size]
            x_test = [self.x_ws.transform(x) for x in batch]

            while len(x_test) < batch_size:
                x_test += x_test
            if len(x_test) > batch_size:
                x_test = x_test[:batch_size]

            x_test_len = [len(x) for x in batch]

            while len(x_test_len) < batch_size:
                x_test_len += x_test_len
            if len(x_test_len) > batch_size:
                x_test_len = x_test_len[:len(x_test)]

            x_test, x_test_len = np.array(x_test), np.array(x_test_len)

            # print('x_test.shape', x_test.shape, 'x_test_len.shape', x_test_len.shape, x_test_len)

            with self.graph.as_default():
                # try:
                r, i = self.model.predict(
                    self.sess,
                    x_test,
                    x_test_len
                )
                # except:
                #     print('x_test', x_test, 'x_test_len', x_test_len)
                #     r = []
                #     for x in batch:
                #         r.append(['O'] * len(x))
            ret += [
                self.y_ws.inverse_transform(y)[:len(x)]
                for x, y in zip(batch, r)
            ]
            reti += [
                self.index_label[x] for x in i
            ]

        return np.array(ret[:len(sentence_result)]), np.array(reti[:len(sentence_result)])

    def predict_domain(self, nlu_obj):
        tokens = nlu_obj['tokens']
        tokens = [x.lower() for x in tokens]
        slots, intent = self.predict([tokens])
        crf_ret = get_slots_detail(tokens, slots[0])
        nicsf_ret = {
            'domain': intent[0].split(SPLITOR)[0],
            'intent': intent[0].split(SPLITOR)[1],
            'slots': crf_ret
        }
        nlu_obj['neural_intent_classifier_slot_filler'] = nicsf_ret
        if len(nlu_obj['slots']) <= 0:
            nlu_obj['slots'] = crf_ret
        return nlu_obj
    
    def predict_intent(self, nlu_obj):
        return self.predict_domain(nlu_obj)
    
    def predict_slot(self, nlu_obj):
        return self.predict_domain(nlu_obj)

    def eval(self, sentence_result, y_data, progress=False):

        slot_result, domain_result = list(zip(*y_data))

        y_pred, y_pred_target = self.predict(sentence_result, progress=progress)
        y_test = slot_result
        y_target = np.array([
            [x]
            for x in domain_result
        ])
        y_pred_target = np.array([
            [x]
            for x in y_pred_target
        ])

        # print(y_target.shape)
        # print(y_pred_target.shape)
        
        return OrderedDict((
            ('accuracy', metrics.flat_accuracy_score(y_test, y_pred)),
            ('precision', metrics.flat_precision_score(y_test, y_pred, average='weighted')),
            ('recall', metrics.flat_recall_score(y_test, y_pred, average='weighted')),
            ('f1', metrics.flat_f1_score(y_test, y_pred, average='weighted')),
            ('softmax_accuracy', metrics.flat_accuracy_score(y_target, y_pred_target)),
            ('softmax_precision', metrics.flat_precision_score(y_target, y_pred_target, average='weighted')),
            ('softmax_recall', metrics.flat_recall_score(y_target, y_pred_target, average='weighted')),
            ('softmax_f1', metrics.flat_f1_score(y_target, y_pred_target, average='weighted')),
        ))
    
    @staticmethod
    def cv_eval(sentence_result, y_data, cv=3):
        """用cv验证模型"""

        np.random.seed(0)
        
        x_train = sentence_result
        y_train = y_data

        slot_result, domain_result = list(zip(*y_data))
        y_target = [
            x
            for x in domain_result
        ]
        labels = sorted(list(set(y_target)), key=lambda x: (len(x), x))
        index_label = {}
        label_index = {}
        for index, label in enumerate(labels):
            index_label[index] = label
            label_index[label] = index

        print('=' * 30)
        print('Intent Classification Score')
        print('=' * 30)

        def score_func_intent(y_true, y_pred):
            _, y_true_b = list(zip(*y_true))
            _, y_pred_b = y_pred
            y_true_b = [x for x in y_true_b]
            y_true_b = [label_index[x] for x in y_true_b]
            y_pred_b = [label_index[x] for x in y_pred_b]
                
            return f1_score(y_true_b, y_pred_b, average='weighted') 

        crf = NeuralIntentClassifierSlotFiller()

        cv_result = cross_validate(
            crf, x_train, y_train,
            scoring=make_scorer(score_func_intent), cv=cv, verbose=1
        )

        for k, v in cv_result.items():
            print(k)
            print(np.mean(v))
            print(v)

        print('=' * 30)
        print('Slot Filler Score')
        print('=' * 30)

        def score_func_slot(y_true, y_pred):
            y_true_a, _ = list(zip(*y_true))
            y_pred_a, _ = y_pred
            return metrics.flat_f1_score(y_true_a, y_pred_a, average='weighted')

        crf = NeuralIntentClassifierSlotFiller()

        cv_result = cross_validate(
            crf, x_train, y_train,
            scoring=make_scorer(score_func_slot), cv=cv, verbose=1
        )

        for k, v in cv_result.items():
            print(k)
            print(np.mean(v))
            print(v)
        

def unit_test():
    import pickle

    print('get data')
    x_data, y_data, _, _ = generate()
    print(len(x_data), len(x_data[0]), x_data[0])
    print(len(y_data), len(y_data[0]), y_data[0])

    print('train model')
    m = NeuralSlotFiller()
    m.fit(x_data, y_data)

    print('dump model')
    with open('/tmp/tmp.file', 'wb') as fp:
        pickle.dump(m, fp)
    
    print('load model')
    with open('/tmp/tmp.file', 'rb') as fp:
        m_load = pickle.load(fp)

    print(m_load.predict(x_data[:3]))

if __name__ == '__main__':
    unit_test()