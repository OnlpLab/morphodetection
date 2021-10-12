from hyper_params import *
import numpy as np
import time
import unicodedata
from gensim.models import Word2Vec
from copy import deepcopy

if Language == 'ru':
    alphabet = set('ёйцукенгшщзхъфываролджэячсмитьбюп')
    vowels = set('аэыоуяеиёю')
    def remove_accents(input_str):
        return input_str
else:
    alphabet = 'abcdefghijklmnopqrstuvwxyzıß'
    if Language=='de':
        alphabet = set(alphabet+alphabet.upper())
    else:
        alphabet = set(alphabet)
    vowels = 'aeiouyı'
    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

print('loading word vectors...')
start = time.time()

if UseValidatedWords:
    cond = lambda x: x == x.lower()
    if Language=='de':
        cond = lambda x: x[1:] == x[1:].lower()
    with open(ValidatedWordsPath, encoding='utf8') as f:
        validated_words = f.readlines()
        validated_words = {x.strip() for x in validated_words if cond(x)}
        validated_words = {x for x in validated_words if all([c in alphabet for c in remove_accents(x)])
                                                        and any([c in vowels for c in remove_accents(x)])}

word_vectors = {}
if PreTrainedVecsPath.endswith(('.txt', '.vec')):
    with open(PreTrainedVecsPath, encoding='utf8', errors='ignore') as f:
        header = f.readline()
        vocab_size, vector_size = (int(x) for x in header.split())  # throws away invalid file format
        for line in f:
            parts = line.rstrip().split()
            word, vec = parts[0], np.asarray([float(x) for x in parts[1:]])
            if UseValidatedWords:
                if word not in validated_words:
                    continue
            if any([c not in alphabet for c in remove_accents(word)]):
                continue
            vec /= np.linalg.norm(vec)
            word_vectors[word] = vec
elif PreTrainedVecsPath.endswith('.bin'):
    wv = Word2Vec.load(PreTrainedVecsPath).wv
    for word in wv.vocab:
        if UseValidatedWords:
            if word not in validated_words:
                continue
        if any([c not in alphabet for c in remove_accents(word)]):
            continue
        vec = deepcopy(wv[word])
        vec /= np.linalg.norm(vec)
        word_vectors[word] = vec
else:
    raise NotImplementedError

print(len(word_vectors))
print('took {:.0f} s'.format(time.time()-start))
