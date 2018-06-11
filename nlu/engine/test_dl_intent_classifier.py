
from nlu.log import LOG
from nlu.engine.dl_intent_classifier import DLIntentClassifier

def unit_test():
    """unit test"""

    from nlu.utils.data_loader import load_nlu_data
    from nlu.utils.data_iob import data_to_iob
    intents, entities = load_nlu_data('nlu_data')
    # intents = [x for x in intents if x['intent'] == 'lottery_inform']
    # print(intents)
    sentence_result, _, domain_result = data_to_iob(intents, entities)

    # print(max([len(x) for x in sentence_result]))

    # feature = 'tfidf'
    # algorithm = 'LinearSVC'

    # for feature in ('hashing', 'tfidf'):
    #     for algorithm in ('RandomForest', 'LinearSVC'):

    #         print('feature', feature, 'algorithm', algorithm)

    #         MLIntentClassifier.cv_eval(
    #             sentence_result, domain_result, cv=5,
    #             feature=feature, algorithm=algorithm
    #         )
    # exit(0)

    eng = DLIntentClassifier()
    eng.fit(sentence_result, domain_result)

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