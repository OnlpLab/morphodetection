# import sys
# import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
# sys.path.append(path)

from hyper_params import *
from evaluator import evaluation_class
from utils import query_dict_by_set
from typing import Set, Tuple, Dict, List
import pickle
import numpy as np
import time
from random import shuffle, choice
from math import ceil
from Levenshtein import distance

avg = lambda x: sum(x)/len(x)

def write_dataset(locked_pairs: Set[Tuple[str,str,str,str]], write_path: str):
    print(os.path.basename(write_path), 'written')
    with open(write_path, 'w', encoding='utf8') as f:
        for tup in locked_pairs:
            f.write('\t'.join(tup)+'\n')

def get_num_examples(scores):
    scores = rank_words(scores)
    lengths = {label: len(words) for label, words in scores.items()}
    lengths = sorted(lengths.values(), reverse=True)
    num_possible = sum([sum([l * j for j in lengths[i + 1:]]) for i, l in enumerate(lengths)])

    if num_possible<TotExamples:
        print(f'{num_possible}-sized data set is created')
        print(f'not enough tagged words for {TotExamples} examples')
    else:
        print(f'there are enough possible examples ({num_possible} to be exact)')
        print(f'a {TotExamples}-sized dataset is created')

    return min(num_possible, TotExamples)

def convert_to_tag2word(labeled_words):
    '''
    convert the labeled_words dict from {word: {labels}} to {label: {words}}
    '''
    new = {}
    for word, labels in labeled_words.items():
        for label in labels:
            if label not in new:
                new[label] = set()
            new[label].add(word)
    return new

def add_copying(labeled_words: Dict[str, Set[str]], scores: Dict[str, Dict[str, int]], examples_num: int):
    '''
    :param labeled_words: {label/s: {word, word, word}}
    :param scores: {word: {label: score}}
    :return: {(word, label1, word, label2)}
    '''
    relevant_labeled_words = {k: v for k, v in labeled_words.items() if ' ' in k}
    if not relevant_labeled_words:
        return set()
    examples_per_relation = int(examples_num/len(relevant_labeled_words))
    relevant_labeled_words = {k: sorted(v, key=lambda x: query_dict_by_set(scores[x], set(k.split())), reverse=True) for k, v in relevant_labeled_words.items()}
    relevant_labeled_words = {k: v[:examples_per_relation] for k, v in relevant_labeled_words.items()}

    new_pairs = set()
    for tags, words in relevant_labeled_words.items():
        tags = tags.split()
        for word in words:
            shuffle(tags)
            new_pairs.add((word, tags[0], word, tags[1]))

    print(len(new_pairs), 'copying pairs added')
    return new_pairs

def rank_words(scores, threshold=0):
    '''
    convert scores from {word: {label:score}} to {label: {word:score}}
    then orders the word by label according to the scores
    :return {{label: [(word,score)]}}
    '''
    scores_per_label = {}
    for word in scores:
        for label in scores[word]:
            if scores[word][label]<threshold:
                continue
            if label not in scores_per_label:
                scores_per_label[label] = {}
            scores_per_label[label][word] = scores[word][label]
    scores_per_label = {k:sorted(v.items(), key=lambda x: x[1], reverse=True) for k,v in scores_per_label.items()}
    return scores_per_label

class DatasetCreator:
    def __init__(self, exp_dir, threshold, labels_to_merge, evaluator: evaluation_class, embeddings, word2idx, scores=None):
        self.dist = 'surface' if OrthoAlg else 'semantic'
        self.threshold = threshold
        self.labels_to_merge = labels_to_merge
        self.evaluator = evaluator
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.exp_dir = exp_dir
        self.scores = scores
        if not self.scores:
            with open(os.path.join(OutputsDir, self.exp_dir, 'scores'), 'rb') as f:
                self.scores = pickle.load(f)
        self.num_examples = get_num_examples(scores)

    def create_dataset(self, copy_examples=CopyExamples):
        scores = self.cap_scores_per_relation()

        labeled_words = self.evaluator.tag(scores, self.threshold)
        labeled_words = self.merge_syncretic(labeled_words)
        labeled_words = convert_to_tag2word(labeled_words)

        pkl_path = os.path.join(OutputsDir, self.exp_dir, 'csls_scores.pkl')
        locked_pairs = self.create_pairs_CSLS(labeled_words, pkl_path, 2)

        locked_pairs = self.unmerge_syncretic(locked_pairs)
        locked_pairs |= add_copying(labeled_words, scores, copy_examples)

        return locked_pairs

    def create_pairs_CSLS(self, labeled_words, pkl_path, K):
        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as f:
                pairs_sorted = pickle.load(f)

        else:
            print('started')
            # labeled_words = {k: list(v) for k,v in labeled_words.items()}
            labeled_words = {k: [w for w in v if w in self.word2idx] for k, v in labeled_words.items()}
            # labeled_word2idx = {k: {w:i for i,w in enumerate(v)} for k,v in labeled_words.items()}
            labeled_embeds = {k: np.stack([self.embeddings[self.word2idx[w]] for w in v]) for k, v in labeled_words.items()}

            average_neighborhoods = {}
            potential_pairmates = {}

            start = time.time()
            for label1 in labeled_words:

                for label2 in labeled_words:
                    if label1 == label2:
                        continue

                    sims = np.dot(labeled_embeds[label1], labeled_embeds[label2].T)
                    highest_idxs = np.argsort(-sims)[:, :K]
                    average_neighborhood = np.mean(-np.sort(-sims)[:, :K], axis=1)

                    for i, word in enumerate(labeled_words[label1]):
                        if word not in average_neighborhoods:
                            average_neighborhoods[word] = {}
                        if word not in potential_pairmates:
                            potential_pairmates[word] = {}

                        average_neighborhoods[word][label2] = average_neighborhood[i]
                        potential_pairmates[word][label2] = {labeled_words[label2][idx] for idx in highest_idxs[i]}

                print(time.time() - start)

            print('phase1, done', time.time() - start)
            start = time.time()

            done_labels = set()
            pairs_unsorted = set()
            for label in labeled_words:
                for word in labeled_words[label]:
                    for label2 in labeled_words:
                        if label2 == label or label2 in done_labels:
                            continue

                        for word2 in potential_pairmates[word][label2]:
                            score = self.CSLS_score(word, word2, average_neighborhoods[word][label2],
                                                    average_neighborhoods[word2][label])
                            if not score:
                                continue
                            pair1 = ((word, label, word2, label2), score)
                            pair2 = ((word2, label2, word, label), score)
                            pairs_unsorted.update((pair1, pair2))

            print(time.time() - start)
            pairs_sorted = sorted(pairs_unsorted, key=lambda x: x[1], reverse=True)

            with open(pkl_path, 'wb') as f:
                pickle.dump(pairs_sorted, f)

        pairs = set([x[0] for x in pairs_sorted][:self.num_examples])

        return pairs

    def CSLS_score(self, word, word2, average_neighborhood_word, average_neighborhood_word2):
        if self.dist == 'semantic':
            vec = self.embeddings[self.word2idx[word]]
            vec2 = self.embeddings[self.word2idx[word2]]
            sim = np.dot(vec, vec2)
            if 1. - sim < 0.001:
                return None

        elif self.dist == 'surface':
            sim = -distance(word, word2)
            if sim == 0:
                return None

        else:
            raise NotImplementedError

        return 2 * sim - average_neighborhood_word - average_neighborhood_word2

    def cap_scores_per_relation(self):
        scores_per_label = rank_words(self.scores, self.threshold)
        cap = ceil(avg([len(v) for v in scores_per_label.values()]))
        scores_per_label = {k: dict(v[:cap]) for k, v in scores_per_label.items()}

        # now I rewind scores to the same {word {label:score}} format
        new_scores = {}
        for label in scores_per_label:
            for word in scores_per_label[label]:
                if word not in new_scores:
                    new_scores[word] = {}
                new_scores[word][label] = scores_per_label[label][word]

        return new_scores

    def merge_syncretic(self, labeled_words):
        # adding the labels omitted due to syncretism
        # if a word has a score for a label in labels_to_merge it is substituted with the multiple corresponding labels with
        # white-space between them
        new_labels = {l: ' '.join(labels_set) for labels_set in self.labels_to_merge for l in labels_set}

        new_labeled_words = {}
        for word, labels in labeled_words.items():
            temp_labels = set()
            for label in labels:
                if label in new_labels:
                    temp_labels.add(new_labels[label])
                else:
                    temp_labels.add(label)
            new_labeled_words[word] = temp_labels

        return new_labeled_words

    @staticmethod
    def unmerge_syncretic(locked_pairs: Set[Tuple[str,str,str,str]], undersample=True):
        pairs_to_add = set()
        pairs_to_remove = set()
        for pair in locked_pairs:
            word1, label1, word2, label2 = pair
            if ' ' in label1 or ' ' in label2:
                pairs_to_remove.add(pair)
                # if ' ' in label1 and ' ' in label2:
                nl1 = label1.split()
                nl2 = label2.split()
                if not undersample:
                    for l1 in nl1:
                        for l2 in nl2:
                            pairs_to_add.add((word1, l1, word2, l2))
                else:
                    pairs_to_add.add((word1, choice(list(nl1)), word2, choice(list(nl2))))

        locked_pairs |= pairs_to_add
        locked_pairs -= pairs_to_remove

        return locked_pairs

def main(exp_dir, threshold, labels_to_merge, evaluator: evaluation_class, embeddings, word2idx, scores=None):
    data_creator = DatasetCreator(exp_dir, threshold, labels_to_merge, evaluator, embeddings, word2idx, scores=scores)

    examples = data_creator.create_dataset()
    write_path = os.path.join(OutputsDir, exp_dir, f'inflec_data.txt')
    write_dataset(examples, write_path)
