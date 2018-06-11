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
                 epochs=5):
        super(KerasIntentClassifier, self).__init__(
            domain_implement=True,
            intent_implement=True,
            slot_implement=False)
        
        self.graph = tf.Graph()

        self.model_params = {
            'max_len': max_len,
            'max_features': max_features,
            'embedding_size': embedding_size,
            'optimizer': optimizer,
            'epochs': epochs,
        }
    
    def build_model(self):
        n_target = self.model_params['n_target']
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
        model = keras.layers.LSTM(8, recurrent_dropout=0.05)(model)
        model = keras.layers.Dense(32)(model)
        model = keras.layers.Dropout(0.25)(model)
        model = keras.layers.Dense(n_target)(model)
        model = keras.layers.Activation('softmax')(model)
        model = keras.models.Model(
            inputs=[input_layer],
            outputs=[model]
        )

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.model = model

    def fit(self, sentence_result, domain_result):

        x_text = sentence_result
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
        self.model_params['index_label'] = index_label
        self.model_params['label_index'] = label_index

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

        self.ohe = ohe = OneHotEncoder(n_target)
        y_target = [label_index[y] for y in y_target]
        ohe.fit(np.array(y_target).reshape(-1, 1))
        y_train = ohe.transform(
            np.array(y_target).reshape(-1, 1)
        )

        cw = class_weight.compute_class_weight('balanced', np.unique(y_target), y_target)

        self.build_model()
        self.model.fit(
            x_train, y_train,
            epochs=self.model_params['epochs'],
            class_weight=cw
        )

    
    def predict(self, sentence_result):
        with self.graph.as_default():
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
            ret = pred.argmax(1)
            prob = np.max(pred, axis=1)
            ret = [self.model_params['index_label'][x] for x in ret]
            return ret, prob

    def predict_domain(self, nlu_obj):
        tokens = nlu_obj['tokens']
        tokens = [x.lower() for x in tokens]
        ret, prob = self.predict([tokens])
        dl_ret = {
            'domain': ret[0].split(SPLITOR)[0],
            'intent': ret[0].split(SPLITOR)[1],
            'prob': float(prob[0]),
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

            tmp_path = '/tmp/keras_intent_classifier.tmp'
            with open(tmp_path, 'wb') as fp:
                fp.write(state['model_data'])
            self.model_params = state['model_params']
            self.build_model()
            self.model.load_weights(tmp_path)
            os.remove(tmp_path)
    
    def eval(self, sentence_result, domain_result):
        assert self.tokenizer is not None, 'tokenizer not fitted'
        assert self.model is not None, 'model not fitted'

        # x_test = self.vectorizer.transform([''.join(x) for x in sentence_result])
        y_test = [x for x in domain_result]
        y_pred, _ = self.predict(sentence_result)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }

        bad = []
        for sent, r, p, in zip(sentence_result, y_test, y_pred):
            if r != p:
                bad.append((
                    ''.join(sent),
                    r,
                    p
                ))
        
        metrics['bad'] = bad

        return metrics