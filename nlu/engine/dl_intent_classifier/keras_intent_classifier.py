"""classify intent by keras(tensorflow)"""

from tensorflow import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KerasIntentClassifier(object):
    """keras model wrapper"""

    ohe = None
    tokenizer = None
    unk = '<unk>'
    
    def __init__(self,
                 max_len=100,
                 max_features=10000,
                 embedding_size=64,
                 optimizer='adam',
                 epochs=5):
        self.model_params = {
            'max_len': max_len,
            'max_features': max_features,
            'embedding_size': embedding_size,
            'optimizer': optimizer,
            'epochs': epochs,
        }
    
    def build_model(self, n_target):
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
            x[0][2:]
            for x in domain_result
        ]

        self.labels = labels = sorted(list(set(y_target)), key=lambda x: (len(x), x))
        self.index_label = index_label = {}
        self.label_index = label_index = {}
        for index, label in enumerate(labels):
            index_label[index] = label
            label_index[label] = index

        n_target = len(labels)

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

        self.build_model(n_target)
        self.model.fit(
            x_train, y_train,
            epochs=self.model_params['epochs'],
            class_weight=cw
        )

    
    def predict(self, sentence_result):
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
        ret = [self.index_label[x] for x in ret]
        return ret, prob
    
    
    def eval(self, sentence_result, domain_result):
        assert self.tokenizer is not None, 'tokenizer not fitted'
        assert self.model is not None, 'model not fitted'

        # x_test = self.vectorizer.transform([''.join(x) for x in sentence_result])
        y_test = [x[0][2:] for x in domain_result]
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