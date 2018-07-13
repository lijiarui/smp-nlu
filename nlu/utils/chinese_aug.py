"""Chinese Augmentation"""

import re
import numpy as np

def space_re(_):
    return np.random.choice([
        ' ',
        ',',
        '，',
        '；',
        '',
        '  ', # 2个空格
        '　', # 中文空格
    ])

def de_re(_):
    if np.random.random() < 0.1:
        return '地'
    return '的'

def tail(sentence):
    if np.random.random() < 0.5:
        s = np.random.choice([
            '。', '？', '！', '…', '的', '吧', '的吧', '呢', '呢？', '的呢', '啦', '啊', '阿', '了'
        ])
        if not sentence.endswith(s):
            sentence += s
    return sentence

def chinese_aug(sentence, en_tail=False):
    """对中文进行数据增强"""
    # sentence = re.sub(r'\s', space_re, sentence)
    sentence = re.sub(r'的', de_re, sentence)
    if en_tail:
        sentence = tail(sentence)
    return sentence
