import numpy as np
from random import seed
from hyper_params import Paradigm, Language, FixLemmas, MaxSupervision, OptimalEdits
from contextlib import suppress
from Levenshtein import opcodes
from typing import Dict, Set, Tuple
seed(15703)

def unfold_dict_values(a_dict: Dict[str, Set[str]]) -> Set[Tuple[str, str]]:
    new = set()
    for k, vs in a_dict.items():
        for v in vs:
            new.add((k, v))
    return new

def query_dict_by_set(a_dict, a_set):
    key = set(a_dict.keys()) & a_set
    key = list(key)[0]
    return a_dict[key]

def normalize(vector):
    if len(vector.shape) == 1:
        if np.linalg.norm(vector) == 0:
            print('what?!')
            return np.zeros_like(vector)
        return vector / np.linalg.norm(vector)
    elif len(vector.shape) == 2:
        return vector / np.linalg.norm(vector, axis=1).reshape(-1, 1)
    else:
        raise NotImplementedError

def cosine_dist(v1, v2):
    return 1. - np.nan_to_num(np.dot(normalize(v1), normalize(v2).T))

def special_deduction(vec1, vec2):
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        return None
    return vec1-vec2

def fast_unique_chars(x, y):
    res_x, res_y = '',''
    for a,b,c,d,e in opcodes(x,y):
        if a is 'equal':
            continue
        elif a is 'insert':
            res_y += y[d:e]
        elif a is 'delete':
            res_x += x[b:c]
        else: # a is 'replace'
            res_x += x[b:c]
            res_y += y[d:e]
    return res_x, res_y

def optimal_unique_chars(x: str, y: str):
    '''
    implementation of http://web.karabuk.edu.tr/hakankutucu/CME222/MIT[1].Press.Introduction.to.Algorithms.2nd.Edition.eBook-TLFeBOOK.pdf
    by Cormen et al. section 15.4
    :param x:
    :param y:
    :return: 0 - up; 1 - left; 2 - diagonal
    '''
    m = len(x)
    n = len(y)
    B = np.zeros((m, n), int)
    C = np.zeros((m+1, n+1))

    for i in range(m):
        ci = i+1
        for j in range(n):
            cj = j+1
            if x[i]==y[j]:
                B[i,j] = 2
                C[ci,cj] = C[ci-1,cj-1] + 1
            elif C[ci-1, cj] <= C[ci, cj-1]:
                B[i,j] = 1
                C[ci,cj] = C[ci, cj-1]
            else:
                C[ci,cj] = C[ci-1, cj]

    common_idxs_in_x, common_idxs_in_y = set_output_LCS(B, m-1, n-1, [set(), set()])
    unique_to_x = ''.join([x[i] for i in range(len(x)) if i not in common_idxs_in_x])
    unique_to_y = ''.join([y[i] for i in range(len(y)) if i not in common_idxs_in_y])

    return unique_to_x, unique_to_y, len(common_idxs_in_x)

if OptimalEdits:
    unique_chars = optimal_unique_chars
else:
    def unique_chars(x,y):
        res1, res2 = fast_unique_chars(x, y)
        return res1, res2, 0

def set_output_LCS(B, i, j, currents):
    if i==0 and j==0:
        if B[i,j] == 2:
            currents[0].add(i)
            currents[1].add(j)
        return currents
    if B[i,j]==2:
        currents[0].add(i)
        currents[1].add(j)
        if i==0 or j==0:
            return currents
        return set_output_LCS(B, i-1, j-1, currents=currents)
    elif B[i,j]==0:
        if i == 0:
            i = 1
            j -= 1
        return set_output_LCS(B, i-1, j, currents=currents)
    # else, B[i,j]==1
    if j == 0:
        j = 1
        i -= 1
    return set_output_LCS(B, i, j-1, currents=currents)

def choose_covered_lemmas(words_from_embeddings, unimorph_path, num_supervision=MaxSupervision, write_path=None):
    if num_supervision == 5 and FixLemmas:
        if Language == 'en' and Paradigm == 'V':
            return ['know', 'use', 'look', 'keep', 'take']
        if Language == 'ru' and Paradigm == 'V':
            return ['сказать', 'образоваться', 'делать', 'идти', 'смотреть']
        if Language == 'lv' and Paradigm == 'V':
            return ['nesaprast', 'darīt', 'atrast', 'atbildēt', 'braukt']
        if Language == 'fr' and Paradigm == 'V':
            return ['agir', 'dire', 'choisir', 'tirer', 'réussir']
        if Language == 'fi' and Paradigm == 'V':
            return ['käyttää', 'sanoa', 'saada', 'jatkaa', 'muistaa']
        if Language == 'es' and Paradigm == 'V':
            return ['ayudar', 'comer', 'contar', 'creer', 'decir']
        if Language == 'tr' and Paradigm == 'V':
            return ['istemek', 'bilmek', 'okumak', 'yazmak', 'görmek']
        if Language == 'hu' and Paradigm == 'V':
            return ['fogad', 'találkozik', 'csatlakozik', 'reagál', 'megtesz']
    else:
        words = set(words_from_embeddings)

        lemma_counts_total = {}
        lemma_counts_vec = {}
        with open(unimorph_path, encoding='utf-8') as f:
            for line in f:
                elements = line.split()
                if not elements:
                    continue
                if not elements[-1].startswith(Paradigm):
                    continue
                if len(elements)!=3:
                    continue
                if Language == 'ru' and elements[-1].startswith('V.PTCP'):
                    continue
                if Language == 'es' and Paradigm == 'V':
                    if elements[0].endswith('se'):
                        continue
                    # excluding most of the verb-se reflexive forms (that have the same lemma in unimorph for some reason)
                    if elements[1].endswith(('rse', 'dose', 'monos', 'nse', 'ate', 'ete', 'aos', 'eos', 'íos')):
                        continue
                lemma = elements[0]
                # word = max(elements[1:-1], key=len)
                word = elements[1]

                if lemma not in lemma_counts_total:
                    lemma_counts_total[lemma] = 0
                    lemma_counts_vec[lemma] = 0
                lemma_counts_total[lemma] += 1
                if word in words and len(elements[1:-1]) == 1:
                    lemma_counts_vec[lemma] += 1

        lemma_percentage = {}
        for lemma in lemma_counts_total:
            lemma_percentage[lemma] = lemma_counts_vec[lemma]/lemma_counts_total[lemma]

        lemmas_list = sorted(lemma_counts_vec.items(), reverse=True, key=lambda x: x[1])

        lemmas_list = [x[0] for x in lemmas_list]
        if Language=='en':
            lemmas_list = ['know','use','look','keep','take','say','go','get','make','think','see','come','want','find','give','tell','work','call','try','ask','need','feel','become','leave','put','mean','let','begin','seem','help','talk','turn','start','show','hear','play','run','move','like','live','believe','hold','bring','happen','write','provide','sit','stand','lose','pay','meet','include','continue','set','learn','change','lead','understand','watch','follow','stop','create','speak','read','allow','add','grow','open','walk','win','offer','remember','love','consider','appear','buy','wait','serve','die','send','expect','build','stay','fall','cut','reach','kill','remain','suggest','raise','pass','sell','require','report','decide','pull','request','close']\
                           + lemmas_list
        with suppress(ValueError):
            if Language=='fr':
                lemmas_list.remove('avoir')
                lemmas_list.remove('avoyr')
                lemmas_list.remove('estre')
            elif Language=='es':
                lemmas_list.remove('ir')
                lemmas_list.remove('ser')
                lemmas_list.remove('estar')
            elif Language == 'fi':
                lemmas_list.remove('olla')
                lemmas_list.remove('ei')
            elif Language == 'ru':
                lemmas_list.remove('быть')
            elif Language == 'mt':
                lemmas_list.remove('jisem')
            elif Language == 'lv':
                lemmas_list.remove('būt')
                lemmas_list.remove('nebūt')
            elif Language == 'he':
                lemmas_list.remove('היה')
                lemmas_list.remove('יכול')

        if Language == 'en' and num_supervision == 5:
            return ['get', 'make', 'know', 'see', 'use']
        return lemmas_list[:num_supervision]
