"""
使用sklearn中的机器学习算法进行意图识别

@TODO 现在只实现了一句话只存在一个意图、领域的情况，如果一句话存在多个领域和意图，在后续实现

NLU_LOG_LEVEL=debug python3 -m nlu.engine.ml_intent_classifier

"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from nlu.log import LOG
from nlu.utils.data_utils import SPLITOR
from nlu.engine.engine_core import EngineCore


class MLIntentClassifier(EngineCore):
    """注意文本都会变小写"""

    def __init__(self):
        """初始化"""
        super(MLIntentClassifier, self).__init__(
            domain_implement=True,
            intent_implement=True,
            slot_implement=False)
        self.model = None
        self.vectorizer = None
    
    def build_vectorizer(self, feature):
        if feature == 'tfidf':
            v = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_features=20000)
            self.vectorizer = v
        elif feature == 'hashing':
            v = HashingVectorizer(analyzer='char', ngram_range=(1, 2), n_features=500)
            self.vectorizer = v

        if self.vectorizer is None:
            raise Exception('Unknown feature "{}"'.format(feature))

    def build_model(self, sentence_result, domain_result, feature, algorithm):

        self.build_vectorizer(feature)

        x_text = [
            ''.join(x).lower() for x in sentence_result
        ]

        y_class = [
            x
            for x in domain_result
        ]

        try:
            with open('/tmp/ml_intent_classifier.tmp', 'w') as fp:
                for x, y in zip(x_text, y_class):
                    fp.write('{}\t{}\n'.format(x, y))
        except:
            pass
        
        x_train = self.vectorizer.fit_transform(x_text)

        class_index = {}
        index_class = {}
        for i, c in enumerate(sorted(list(set(y_class)))):
            class_index[c] = i
            index_class[i] = c
        self.class_index = class_index
        self.index_class = index_class

        y_train = [self.class_index[x] for x in domain_result]

        model = None
        if algorithm == 'RandomForest':
            model = RandomForestClassifier(random_state=0, class_weight='balanced', n_jobs=-1)
        elif algorithm == 'SVC':
            model = SVC(random_state=0, probability=True, class_weight='balanced')
        elif algorithm == 'LinearSVC':
            model = LinearSVC(random_state=0, class_weight='balanced')
        else:
            raise Exception('Unknown algorithm "{}"'.format(algorithm))
        
        return model, x_train, y_train


    def fit(self,
            sentence_result, domain_result,
            feature='tfidf',
            algorithm='RandomForest'):
        """fit model"""

        LOG.debug('fit MLIntentClassifier')
        model, x_train, y_train = self.build_model(sentence_result, domain_result, feature, algorithm)
        self.model = model
        self.model.fit(x_train, y_train)

    def predict_domain(self, nlu_obj):
        tokens = nlu_obj['tokens']
        tokens = [x.lower() for x in tokens]
        ret = self.predict([tokens])
        ml_ret = {
            'domain': ret[0][0].split(SPLITOR)[0],
            'intent': ret[0][0].split(SPLITOR)[1],
            'prob': ret[1][0],
        }
        nlu_obj['ml_intent_classifier'] = ml_ret
        if nlu_obj['domain'] is None:
            nlu_obj['domain'] = ml_ret['domain']
        if nlu_obj['intent'] is None:
            nlu_obj['intent'] = ml_ret['intent']
        return nlu_obj
    
    def predict_intent(self, nlu_obj):
        return self.predict_domain(nlu_obj)

    def predict(self, sentence_result):
        assert self.vectorizer is not None, 'vectorizer not fitted'
        assert self.model is not None, 'model not fitted'

        x_test = self.vectorizer.transform([''.join(x).lower() for x in sentence_result])
        y_pred = self.model.predict(x_test)
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(x_test).max(1)
        else:
            y_prob = [[-1]] * len(y_pred)
        
        return [self.index_class[x] for x in y_pred], [x for x in y_prob]
    
    def eval(self, sentence_result, domain_result):
        assert self.vectorizer is not None, 'vectorizer not fitted'
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
    
    @staticmethod
    def cv_eval(sentence_result, domain_result, cv=5,
                feature='tfidf',
                algorithm='RandomForest'):
        np.random.seed(0)
        ml = MLIntentClassifier()
        model, x_train, y_train = ml.build_model(sentence_result, domain_result, feature, algorithm)
        score = make_scorer(f1_score, average='weighted')
        cv_result = cross_validate(
            model, x_train, y_train,
            scoring=score, cv=cv, verbose=1
        )

        for k, v in cv_result.items():
            print(k)
            print(np.mean(v))
            print(v)


def unit_test():
    """unit test"""

    from nlu.utils.data_loader import load_nlu_data
    from nlu.utils.data_iob import data_to_iob
    intents, entities = load_nlu_data('nlu_data')
    # intents = [x for x in intents if x['intent'] == 'lottery_inform']
    # print(intents)
    sentence_result, _, domain_result = data_to_iob(intents, entities)

    # print(max([len(x) for x in sentence_result]))

    feature = 'tfidf'
    algorithm = 'LinearSVC'

    for feature in ('hashing', 'tfidf'):
        for algorithm in ('RandomForest', 'LinearSVC'):

            print('feature', feature, 'algorithm', algorithm)

            MLIntentClassifier.cv_eval(
                sentence_result, domain_result, cv=5,
                feature=feature, algorithm=algorithm
            )
    exit(0)

    eng = MLIntentClassifier()
    eng.fit(sentence_result, domain_result, feature=feature, algorithm=algorithm)

    LOG.debug('ml fitted')

    metrics = eng.eval(sentence_result, domain_result)

    print('metrics')
    for k, v in metrics.items():
        if k != 'bad':
            print(k, v)
    
    print('bad count', len(metrics['bad']))
    for b in metrics['bad']:
        # sentence, real, pred
        print('{}\t{}\t{}'.format(b[0], b[1], b[2]))

if __name__ == '__main__':
    unit_test()