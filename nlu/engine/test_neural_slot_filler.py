""" for unit test
"""

from nlu.log import LOG
from nlu.engine.neural_slot_filler import NeuralSlotFiller

def unit_test():
    """unit test"""

    from nlu.utils.data_loader import load_nlu_data
    from nlu.utils.data_iob import data_to_iob
    intents, entities = load_nlu_data('nlu_data')
    # intents = [x for x in intents if x['intent'] == 'lottery_inform']
    # print(intents)
    sentence_result, slot_result, _ = data_to_iob(intents, entities)

    # print(max([len(x) for x in sentence_result]))

    NeuralSlotFiller.cv_eval(sentence_result, slot_result, cv=5)
    exit(0)

    eng = NeuralSlotFiller()
    eng.fit(sentence_result, slot_result)

    LOG.debug('crf fitted')

    metrics = eng.eval(sentence_result, slot_result, progress=True)
    for k, v in metrics.items():
        print(k, v)
    
    # acc, bad = eng.exact_eval(sentence_result, slot_result)
    # print('exact acc', acc)
    # print('bad count', len(bad))

    print(eng.predict([list('我要买第18138期')]))

if __name__ == '__main__':
    unit_test()