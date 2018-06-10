from typing import List, Union
import os
import re
from my_lib import script_dir
from my_lib import load


STOP_WORDS = set(load(script_dir() / 'stop_words.json'))

def tokenize_en(text: str) -> List[str]:
    '''split *text* into English words and remove stop words
    >>> tokenize_en('Hello! World! I am a *scientist* in 2017.')
    ['hello', 'world', 'scientist']
    '''
    return list(filter(lambda x: x, map(_normalize_word, re.split(r'[^a-zA-Z0-9_-]', text))))


def _normalize_word(word: str) -> Union[str, None]:
    word = word.lower().strip('_-')

    if word == '':
        return None

    if re.match(r'^-?[0-9]+$', word):
        return None

    if re.match(r'^[_-]+$', word):
        return None


    if word in STOP_WORDS:
        return None

    return word
