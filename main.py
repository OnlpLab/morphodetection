# import sys
# import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
# sys.path.append(path)

from hyper_params import *
from supervision import supervision_provider
from utils import *
import tagging, data_generator
from evaluator import evaluation_class
from typing import Dict, List
import torch
from random import sample
from words_pool import word_vectors
from write_dataset import main as write_dataset
from datetime import datetime

def rearrange_embeddings(word_vectors: Dict[str, np.ndarray], idx2word_path='') -> (np.ndarray, Dict[str, int]):

    embeddings_list = [np.zeros_like(next(iter(word_vectors.values())))]

    word2idx = {PlaceHolderToken: 0}
    if idx2word_path:
        with open(idx2word_path, 'rb') as f:
            idx2word = pickle.load(f)['idx2word']

        if idx2word[0] != PlaceHolderToken:
            idx2word = [PlaceHolderToken] + idx2word
        for word in idx2word[1:]:
            embeddings_list.append(word_vectors[word])
            word2idx[word] = len(word2idx)
    else:
        idx2word = [PlaceHolderToken]
        for word, vec in word_vectors.items():
            word2idx[word] = len(word2idx)
            idx2word.append(word)
            embeddings_list.append(vec)
    embeddings = torch.FloatTensor(embeddings_list)

    return embeddings, word2idx, idx2word

def determine_file_name():
    meta = f'_{Language}{Paradigm}'
    meta += f'_{MaxSupervision}sup'
    if OrthoAlg:
        meta += '_surface'
        return meta

    meta += f'_{DeviationFrom}'
    meta = meta.replace('+-', '-')
    meta += f'_enhance-{ExpansionIterations}'
    vec_type = os.path.basename(PreTrainedVecsPath).split('.')[0]
    meta += f"_{vec_type}"
    if '.mini.' in PreTrainedVecsPath:
        meta += '_minidict'
    return meta

def make_experiment_dir():
    meta = f'_{Language}{Paradigm}'
    meta += f'_{MaxSupervision}sup'
    if OrthoAlg:
        meta += '_surface'
    else:
        meta += f'_enh{ExpansionIterations}'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = timestamp + meta
    if len(sys.argv) == 2:
        meta += '_' + sys.argv[1]
    os.mkdir(os.path.join(OutputsDir, meta))
    return meta


def scores_to_dict(scores, idx2word: list, idx2label: list, labels_to_merge: List[set]):
    '''
    This method is only for backward compatibility
    :param scores: torch.sparse.Tensor
    :return:
    '''
    indicator = set().union(*labels_to_merge)
    new_mergers = {}
    for s in labels_to_merge:
        for label in s:
            new_mergers[label] = s

    scores = scores.coalesce()
    new_scores = {}
    for i, (word_idx, label_idx) in enumerate(scores.indices().t()):
        value = scores.values()[i]
        word = idx2word[word_idx]
        if word == PlaceHolderToken:
            continue
        label = idx2label[label_idx]
        if label in indicator:
            label = new_mergers[label]
        else:
            label = {label}
        if word not in new_scores:
            new_scores[word] = {}
        for l in label:
            new_scores[word][l] = float(value)
    return new_scores

def frequency_filter(scores, idx2word, upper_cut, lower_cut=0):
    frequent_words = []
    with open(FrequencyPath, encoding='utf8') as f:
        for i, line in enumerate(f):
            if i < lower_cut:
                continue
            if i == upper_cut:
                break
            frequent_words.append(line.split()[0])
    frequent_words = set(frequent_words)
    scores = {k:v for k,v in scores.items() if k in frequent_words}
    idx2word = [idx2word[0]] + [word for word in idx2word if word in frequent_words]

    return scores, idx2word

def random_filter(scores, idx2word, upper_cut):
    place_holder = idx2word[0]
    idx2word = idx2word[1:]

    chosen_words = set(sample(idx2word, upper_cut))
    scores = {k:v for k,v in scores.items() if k in chosen_words}
    idx2word = [place_holder] + [word for word in idx2word if word in chosen_words]

    return scores, idx2word

def count_cases(labels):
    cases = {k:0 for k in set().union(*labels.values())}
    for word in labels:
        for case in labels[word]:
            cases[case] += 1
    return cases

def load_unimorph_vocab():
    '''
    available only for Spanish
    '''
    path = os.path.join(RootDir, 'vectors', 'es_uni_words.txt')
    with open(path) as f:
        idx2word = [w.strip() for w in f.readlines()]
    word2idx = {w:i for i,w in enumerate(idx2word)}

    print(f'loaded UniMorph vocab. {len(word2idx)} words.')
    return word2idx, idx2word

def write_tagged_words(labels, path, split_corrects=True):
    '''
    :param labels: {word: {labels}} if split_corrects=False, else: a list of 2 such dicts for correct and incorrect tags
    :param path: path to outputted file
    :param split_corrects: if True the outputted file will contain "incorrectly" labeled words at the top
    '''
    if split_corrects:
        correct_labels, incorrect_labels = labels
    else:
        correct_labels, incorrect_labels = {}, labels

    correct_labels = unfold_dict_values(correct_labels)
    incorrect_labels = unfold_dict_values(incorrect_labels)

    with open(path, 'w', encoding='utf8') as f:
        if split_corrects:
            f.write(f'{len(correct_labels)/(len(correct_labels)+len(incorrect_labels)):.4f} definitely correct.\n')
            f.write(f'({len(correct_labels)} out of {len(correct_labels)+len(incorrect_labels)})\n')
            f.write('#######################################\n')
        for tup in correct_labels:
            f.write(f'{tup[0]}\t{tup[1]}\n')
        if split_corrects:
            f.write('#######################################\n')
        for tup in incorrect_labels:
            f.write(f'{tup[0]}\t{tup[1]}\n')

def main(supervision, evaluator, embeddings, word2idx, idx2word):
    exp_dir = make_experiment_dir()
    # meta += '_max'

    # word2idx, idx2word = load_unimorph_vocab()
    # meta += '_univocab'

    print('experiment directory is', exp_dir)

    scores_path = os.path.join(OutputsDir, exp_dir, 'scores')
    if OrthoAlg:
        data_gen = data_generator.SurfaceData(word2idx, BatchSize)
        tagger = tagging.SurfaceTagger(data_gen, supervision)
    else:
        data_gen = data_generator.BatchedData(word2idx, BatchSize)
        enhancer = tagging.EnhanceTagger(data_gen, supervision, embeddings, word2idx, idx2word)
        for i in range(ExpansionIterations):
            print(i)
            enhancer.tag()
            enhancer.update_diffs()
        tagger = tagging.SemanticTagger(data_gen, supervision, embeddings)
    scores = tagger.tag()
    scores = scores_to_dict(scores, idx2word, supervision.idx2label, supervision.labels_to_merge)

    labels_num = len(supervision.labels)
    log_path = os.path.join(OutputsDir, exp_dir, 'log')

    threshold = labels_num//2 if OrthoAlg else 0

    with open(scores_path, 'wb') as f:
        pickle.dump((scores, threshold), f)

    labels = evaluator.tag(scores, threshold)
    correct_labels = {word: labels[word] & evaluator.gold_labels.get(word, set()) for word in labels}
    incorrect_labels = {word: labels[word] - correct_labels[word] for word in labels}
    write_tagged_words([correct_labels, incorrect_labels], os.path.join(OutputsDir, exp_dir, 'tagged_words.txt'))

    # print(len(evaluator.gold_labels))
    # print(len({k:v for k,v in labels.items() if v}))
    evaluator.write_statistics(scores, log_path, 1000, len(idx2word)-1, labels_num, printing=False)

    return scores, exp_dir, threshold

if __name__ == '__main__':
    supervision = supervision_provider(word_vectors, path=AllUnimorphPath)

    embeddings, word2idx, idx2word = rearrange_embeddings(word_vectors)
    del word_vectors

    ps = [AllUnimorphPath]
    evaluator = evaluation_class(ps, idx2word)

    if ExpDir:
        exp_dir = ExpDir
        scores_path = os.path.join(OutputsDir, exp_dir, 'scores')
        with open(scores_path, 'rb') as f:
            scores, threshold = pickle.load(f)
    else:
        scores, exp_dir, threshold = main(supervision, evaluator, embeddings, word2idx, idx2word)
    write_dataset(exp_dir, threshold, supervision.labels_to_merge, evaluator, embeddings, word2idx, scores)
