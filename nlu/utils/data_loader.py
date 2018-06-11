"""
NLU_LOG_LEVEL=debug python3 -m nlu.utils.data_loader
"""


import os
import re
import yaml
from nlu.log import LOG

def load_nlu_data(data_dir):
    """读取NLU数据目录的信息
    目录中应该有intents与entities子目录，分别保存意图和实体信息，为yaml格式
    """
    assert os.path.exists(data_dir), '数据目录不存在'
    entities_dir = os.path.join(data_dir, 'entities')
    intents_dir = os.path.join(data_dir, 'intents')
    assert os.path.exists(entities_dir), '实体目录不存在'
    assert os.path.exists(intents_dir), '意图目录不存在'

    LOG.debug('开始读取entities')

    entities = []
    
    for dirname, _, filenames in os.walk(entities_dir):
        filenames = [x for x in filenames if x.endswith('.yml')]
        for filename in filenames:
            path = os.path.join(dirname, filename)
            obj = yaml.load(open(path))
            assert 'entity' in obj, 'entity的yaml文件必须包括entity键值 {}'.format(path)
            assert isinstance(obj['entity'], str), 'entity的entity值必须是字符串 {}'.format(path)
            assert 'data' in obj, 'entity的yaml文件必须包括data键值 {}'.format(path)
            assert isinstance(obj['data'], list), 'entity的data值必须是列表 {}'.format(path)
            assert len(obj['data']) > 0, 'entity的data列表元素应大于0 {}'.format(path)
            entities.append(obj)

    LOG.debug('开始读取intents')

    intents = []
    
    for dirname, _, filenames in os.walk(intents_dir):
        filenames = [x for x in filenames if x.endswith('.yml')]
        for filename in filenames:
            path = os.path.join(dirname, filename)
            obj = yaml.load(open(path))
            assert 'intent' in obj, 'intent的yaml文件必须包括intent键值 {}'.format(path)
            assert isinstance(obj['intent'], str), 'intent的intent值必须是字符串 {}'.format(path)
            assert 'data' in obj, 'intent的yaml文件必须包括data键值 {}'.format(path)
            assert isinstance(obj['data'], list), 'intent的data值必须是列表 {}'.format(path)
            assert len(obj['data']) > 0, 'intent的data列表元素应大于0 {}'.format(path)
            intents.append(obj)
    
    return intents, entities
    


def unit_test():
    """unit test"""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, '..', '..', 'nlu_data')
    load_nlu_data(data_dir)


if __name__ == '__main__':
    unit_test()

