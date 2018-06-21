"""classify intent by keras(tensorflow)"""

import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nlu.engine.engine_core import EngineCore
from nlu.utils.data_utils import SPLITOR

class KerasIntentClassifier(EngineCore):
    """keras model wrapper"""

    model = None
    ohe = None
    tokenizer = None
    unk = '<unk>'

    def __init__(self,
                 max_len=100,
                 max_features=10000,
                 embedding_size=64,
                 optimizer='adam',
                 epochs=10):
        super(KerasIntentClassifier, self).__init__(
            domain_implement=True,
            intent_implement=True,
            slot_implement=False)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()

        self.model_params = {
            'max_len': max_len,
            'max_features': max_features,
            'embedding_size': embedding_size,
            'optimizer': optimizer,
            'epochs': epochs,
        }

    def build_model(self):
        """构建模型"""

        n_target_domain = self.model_params['n_target_domain']
        n_target_intent = self.model_params['n_target_intent']
        max_len = self.model_params['max_len']
        max_features = self.model_params['max_features']
        embedding_size = self.model_params['embedding_size']
        optimizer = self.model_params['optimizer']

        input_layer = keras.layers.Input(shape=(max_len,))
        model = input_layer
        model = keras.layers.Embedding(
            max_features,
            embedding_size
        )(model)
        model = keras.layers.Dropout(0.25)(model)
        model = keras.layers.LSTM(32, recurrent_dropout=0.05)(model)

        domain_model = model
        domain_model = keras.layers.Dense(32)(domain_model)
        domain_model = keras.layers.Dropout(0.25)(domain_model)
        domain_model = keras.layers.Dense(n_target_domain)(domain_model)
        domain_model = keras.layers.Activation(
            'softmax', name='do')(domain_model)

        intent_model = model
        intent_model = keras.layers.Dense(32)(intent_model)
        intent_model = keras.layers.Dropout(0.25)(intent_model)
        intent_model = keras.layers.Dense(n_target_intent)(intent_model)
        intent_model = keras.layers.Activation(
            'softmax', name='io')(intent_model)

        model = keras.models.Model(
            inputs=[input_layer],
            outputs=[domain_model, intent_model]
        )

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.model = model

    def fit(self, sentence_result, domain_result):
        """拟合模型"""

        x_text = sentence_result
        y_target_domain = [
            x.split(SPLITOR)[0]
            for x in domain_result
        ]
        y_target_intent = [
            x.split(SPLITOR)[1]
            for x in domain_result
        ]

        self.labels_domain = labels_domain = \
            sorted(list(set(y_target_domain)), key=lambda x: (len(x), x))
        self.index_label_domain = index_label_domain = {}
        self.label_index_domain = label_index_domain = {}
        for index, label in enumerate(labels_domain):
            index_label_domain[index] = label
            label_index_domain[label] = index

        self.labels_intent = labels_intent = \
            sorted(list(set(y_target_intent)), key=lambda x: (len(x), x))
        self.index_label_intent = index_label_intent = {}
        self.label_index_intent = label_index_intent = {}
        for index, label in enumerate(labels_intent):
            index_label_intent[index] = label
            label_index_intent[label] = index

        n_target_domain = len(labels_domain)
        n_target_intent = len(labels_intent)
        self.model_params['n_target_domain'] = n_target_domain
        self.model_params['n_target_intent'] = n_target_intent
        self.model_params['index_label_domain'] = index_label_domain
        self.model_params['label_index_domain'] = label_index_domain
        self.model_params['index_label_intent'] = index_label_intent
        self.model_params['label_index_intent'] = index_label_intent

        self.tokenizer = tokenizer = keras.preprocessing.text.Tokenizer(
            self.model_params['max_features'],
            oov_token=self.unk
        )
        tokenizer.fit_on_texts([
            ' '.join(list(x))
            for x in x_text
        ])
        x_seq = tokenizer.texts_to_sequences([
            ' '.join(list(x))
            for x in x_text
        ])
        x_train = keras.preprocessing.sequence.pad_sequences(
            x_seq,
            maxlen=self.model_params['max_len']
        )

        self.ohe_domain = ohe_domain = OneHotEncoder(n_target_domain)
        y_target_domain = [label_index_domain[y] for y in y_target_domain]
        ohe_domain.fit(np.array(y_target_domain).reshape(-1, 1))
        y_train_domain = ohe_domain.transform(
            np.array(y_target_domain).reshape(-1, 1)
        )

        self.ohe_intent = ohe_intent = OneHotEncoder(n_target_intent)
        y_target_intent = [label_index_intent[y] for y in y_target_intent]
        ohe_intent.fit(np.array(y_target_intent).reshape(-1, 1))
        y_train_intent = ohe_intent.transform(
            np.array(y_target_intent).reshape(-1, 1)
        )

        cw_domain = class_weight.compute_class_weight('balanced', np.unique(y_target_domain), y_target_domain)
        cw_intent = class_weight.compute_class_weight('balanced', np.unique(y_target_intent), y_target_intent)

        with self.graph.as_default():
            keras.backend.set_session(self.sess)
            self.build_model()
            self.model.fit(
                x_train, [y_train_domain, y_train_intent],
                epochs=self.model_params['epochs'],
                class_weight={
                    'do': cw_domain,
                    'io': cw_intent
                }
            )

    def predict(self, sentence_result):
        """预测结果"""
        with self.graph.as_default():
            keras.backend.set_session(self.sess)
            x_text = sentence_result
            x_seq = self.tokenizer.texts_to_sequences([
                ' '.join(list(x))
                for x in x_text
            ])
            x_test = keras.preprocessing.sequence.pad_sequences(
                x_seq,
                maxlen=self.model_params['max_len']
            )
            pred = self.model.predict(x_test)
            pred_domain = pred[0]
            pred_intent = pred[1]

            ret_domain = pred_domain.argmax(1)
            ret_intent = pred_intent.argmax(1)
            prob_domain = np.max(pred_domain, axis=1)
            prob_intent = np.max(pred_intent, axis=1)
            ret_domain = [self.model_params['index_label_domain'][x]
                          for x in ret_domain]
            ret_intent = [self.model_params['index_label_intent'][x]
                          for x in ret_intent]
            return ret_domain, prob_domain, ret_intent, prob_intent

    def predict_domain(self, nlu_obj):
        tokens = nlu_obj['tokens']
        tokens = [x.lower() for x in tokens]
        ret_domain, prob_domain, ret_intent, prob_intent = self.predict([tokens])
        dl_ret = {
            'domain': ret_domain[0],
            'intent': ret_intent[0],
            'domain_prob': float(prob_domain[0]),
            'intent_prob': float(prob_intent[0]),
        }
        nlu_obj['dl_intent_classifier'] = dl_ret
        if nlu_obj['domain'] is None:
            nlu_obj['domain'] = dl_ret['domain']
        if nlu_obj['intent'] is None:
            nlu_obj['intent'] = dl_ret['intent']
        return nlu_obj

    def predict_intent(self, nlu_obj):
        return self.predict_domain(nlu_obj)

    def __getstate__(self):
        assert self.model is not None, 'model not fitted'
        tmp_path = '/tmp/keras_intent_classifier.tmp'
        self.model.save_weights(tmp_path)
        with open(tmp_path, 'rb') as fp:
            model_data = fp.read()
        os.remove(tmp_path)
        return {
            'model_data': model_data,
            'model_params': self.model_params,
            'domain_implement': self.domain_implement,
            'intent_implement': self.intent_implement,
            'slot_implement': self.slot_implement,
            'tokenizer': self.tokenizer,
        }

    def __setstate__(self, state):
        self.domain_implement = state['domain_implement']
        self.intent_implement = state['intent_implement']
        self.slot_implement = state['slot_implement']
        self.tokenizer = state['tokenizer']

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            keras.backend.set_session(self.sess)
            tmp_path = '/tmp/keras_intent_classifier.tmp'
            with open(tmp_path, 'wb') as fp:
                fp.write(state['model_data'])
            self.model_params = state['model_params']
            self.build_model()
            self.model.load_weights(tmp_path)
            os.remove(tmp_path)

    def eval(self, sentence_result, domain_result):
        """测试模型"""
        assert self.tokenizer is not None, 'tokenizer not fitted'
        assert self.model is not None, 'model not fitted'

        # x_test = self.vectorizer.transform([''.join(x) for x in sentence_result])
        y_test_domain = [x.split(SPLITOR)[0] for x in domain_result]
        y_test_intent = [x.split(SPLITOR)[1] for x in domain_result]
        y_pred_domain, _, y_pred_intent, _ = self.predict(sentence_result)
        metrics = {}

        print('-' * 10 + ' domain')

        t = {
            'accuracy_domain': accuracy_score(
                y_test_domain, y_pred_domain),
            'precision_domain': precision_score(
                y_test_domain, y_pred_domain, average='weighted'),
            'recall_domain': recall_score(
                y_test_domain, y_pred_domain, average='weighted'),
            'f1_domain': f1_score(
                y_test_domain, y_pred_domain, average='weighted'),
        }
        for k, v in t.items():
            metrics[k] = v

        bad = []
        for sent, r, p, in zip(sentence_result, y_test_domain, y_pred_domain):
            if r != p:
                bad.append((
                    ''.join(sent),
                    r,
                    p
                ))

        metrics['bad_domain'] = bad

        print('-' * 10 + ' intent')

        t = {
            'accuracy_intent': accuracy_score(
                y_test_intent, y_pred_intent),
            'precision_intent': precision_score(
                y_test_intent, y_pred_intent, average='weighted'),
            'recall_intent': recall_score(
                y_test_intent, y_pred_intent, average='weighted'),
            'f1_intent': f1_score(
                y_test_intent, y_pred_intent, average='weighted'),
        }
        for k, v in t.items():
            metrics[k] = v

        bad = []
        for sent, r, p, in zip(sentence_result, y_test_intent, y_pred_intent):
            if r != p:
                bad.append((
                    ''.join(sent),
                    r,
                    p
                ))

        metrics['bad_intent'] = bad

        return metrics
