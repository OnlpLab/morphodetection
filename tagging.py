from hyper_params import *
import torch
import pickle
import numpy as np
from typing import Dict, List, Set, Tuple
from abc import abstractmethod
import time
import data_generator as dg
import supervision as sp
from initial_version import utils

MaxCheck = 100


class AbstractTagger:
    def __init__(self, data_gen: dg.DataGenerator, supervision: sp.supervision_provider):
        self.data_gen = data_gen
        self.supervision = supervision
        self.coalesce_every = 500

        self.start = 0
        self.average_time = 0

    @abstractmethod
    def tag(self):
        pass

    def print_procedure(self, j):
        print(f'{(j + 1)} batches done')
        pair_time = time.time() - self.start
        print(f'took {pair_time:.2f} seconds')

        self.average_time = (self.average_time * ((j + 1) / self.coalesce_every - 1) + pair_time) / \
                            ((j + 1) / self.coalesce_every)
        print(f'average: {self.average_time:.2f} seconds')

        self.start = time.time()

    def coalesce_procedure(self, scores, tot_idxs, j):
        if len(tot_idxs):
            if tot_idxs.device.type == 'cpu':
                scores += torch.sparse.LongTensor(tot_idxs, torch.ones_like(tot_idxs[0]), scores.shape)
            else:
                assert tot_idxs.device.type=='cuda'
                scores += torch.cuda.sparse.LongTensor(tot_idxs, torch.ones_like(tot_idxs[0]), scores.shape)
            scores = scores.coalesce()
            tot_idxs = []
        self.print_procedure(j)
        return scores, tot_idxs


class SurfaceTagger(AbstractTagger):
    def tag(self):
        print('tagging by surface form')
        print(f'{len(self.data_gen.words) ** 2} word pairs to go over')

        all_edits: Dict[Tuple[str], Set[str]] = self.supervision.get_relevant_edits()
        all_edits: Dict[Tuple[str], List[List[int]]] = \
                                    {k: [[self.supervision.labels[v.split()[0]], self.supervision.labels[v.split()[1]]] for v in vs]
                                                for k,vs in all_edits.items()}

        scores = torch.sparse.LongTensor(self.data_gen.tot_words + 1, len(self.supervision.labels))
        tot_idxs = []

        self.start = time.time()
        for examples, word_pair in enumerate(self.data_gen):
            # the only reason to keep counting non-existing batches is for compatibility in prints
            batches = examples / self.data_gen.batch_size
            if (batches+1) % self.coalesce_every == 0:
                scores, tot_idxs,  = self.coalesce_procedure(scores, tot_idxs, batches)

            if not word_pair:
                continue
            word1, word2 = word_pair
            idx1 = self.data_gen.word2idx[word1]
            idx2 = self.data_gen.word2idx[word2]
            edits = utils.unique_chars(word1, word2)

            if edits in all_edits:
                for label_idxs_pair in all_edits[edits]:
                    tot_idxs.append([idx1, label_idxs_pair[0]])
                    tot_idxs.append([idx2, label_idxs_pair[1]])

        scores, _ = self.coalesce_procedure(scores, tot_idxs, batches)
        return scores

    def coalesce_procedure(self, scores, tot_idxs, j):
        if len(tot_idxs):
            tot_idxs = torch.LongTensor(tot_idxs).t()
        return super().coalesce_procedure(scores, tot_idxs, j)

class SemanticBaseCamp(AbstractTagger):
    def __init__(self, data_gen: dg.DataGenerator, supervision: sp.supervision_provider, embeddings, device=Device):
        super().__init__(data_gen, supervision)
        self.embeddings = embeddings.cuda(device=device)
        self.device = device
        self.batch_size = self.data_gen.batch_size
        self.cat_every = 5

    @abstractmethod
    def tag(self):
        pass

    @abstractmethod
    def load_hp(self, pair_data):
        pass

    def condition_from_words_idxs(self, words1, words2, thresholds, normed_averages):
        words1_mat = torch.nn.functional.embedding(words1, self.embeddings)
        words2_mat = torch.nn.functional.embedding(words2, self.embeddings)
        dif_mat = words1_mat - words2_mat
        dif_mat /= torch.norm(dif_mat, dim=1, keepdim=True)
        cos_dist = 1. - torch.mm(dif_mat, normed_averages.t())
        cond = (cos_dist < thresholds).t()
        return cond

    def idxs_to_tag_from_condition(self, cond, words1, words2, ys):
        x1 = torch.masked_select(words1, cond)
        if len(x1) == 0:
            return []

        x2 = torch.masked_select(words2, cond)
        x = torch.cat((x1, x2))
        y1 = torch.masked_select(ys[:, words1.shape[0]:], cond)
        y2 = torch.masked_select(ys[:, :words1.shape[0]], cond)
        y = torch.cat((y2, y1))
        idxs = torch.stack((x, y))

        return idxs

    def train_pre_process(self):
        self.embeddings = self.embeddings.cuda(device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long).cuda(device=self.device)

        thresholds = []
        normed_averages = []
        ys = []
        for label_pair, pair_data in self.supervision.diffs.items():
            label1, label2 = label_pair.split()
            label1 = self.supervision.labels[label1]
            label2 = self.supervision.labels[label2]
            v, d, n = self.load_hp(pair_data)
            threshold = v + n * d
            thresholds.append(threshold)
            if type(pair_data.average) == np.ndarray:
                normed_average = torch.from_numpy(pair_data.average).float().cuda(device=self.device)
            else:
                normed_average = pair_data.average.cuda(device=self.device)
            normed_average /= normed_average.norm()
            normed_averages.append(normed_average)

            y1 = torch.zeros_like(zeros).fill_(label1)
            y2 = torch.zeros_like(zeros).fill_(label2)
            y = torch.cat((y1, y2))
            ys.append(y)

        thresholds = torch.cuda.FloatTensor(thresholds, device=self.device)
        normed_averages = torch.stack(normed_averages)
        normed_averages = normed_averages.cuda(device=self.device)
        ys = torch.stack(ys)

        return thresholds, normed_averages, ys


class SemanticTagger(SemanticBaseCamp):
    def tag(self):
        print('tagging by semantic representation')
        print(f'{len(self.data_gen.words) ** 2} word pairs to go over')
        scores = torch.cuda.sparse.LongTensor(len(self.data_gen.word2idx) + 1, len(self.supervision.labels), device=self.device)

        thresholds, normed_averages, ys = self.train_pre_process()

        tot_idxs = []

        self.start = time.time()
        for j, (words1, words2) in enumerate(self.data_gen):

            if (j + 1) % self.coalesce_every == 0:
                scores, tot_idxs = self.coalesce_procedure(scores, tot_idxs, j)

            cond = self.condition_from_words_idxs(words1, words2, thresholds, normed_averages)
            idxs = self.idxs_to_tag_from_condition(cond, words1, words2, ys)

            if not len(idxs):
                continue

            tot_idxs.append(idxs)
            # torch.cat takes TOO MUCH time when the list is too long,
            # so there's a need to cat the tot_idxs list every now and then
            if (j + 1) % self.cat_every == 0:
                tot_idxs = [torch.cat(tot_idxs, dim=1)]

        scores, _ = self.coalesce_procedure(scores, tot_idxs, j)

        return scores

    def coalesce_procedure(self, scores, tot_idxs, j):
        if len(tot_idxs):
            tot_idxs = torch.cat(tot_idxs, dim=1)
        return super(SemanticTagger, self).coalesce_procedure(scores, tot_idxs, j)

    def load_hp(self, pair_data):
        if DeviationFrom == 'avg':
            v = pair_data.avg_cos
        elif DeviationFrom == 'max':
            v = pair_data.max
        else:
            raise NotImplementedError
        return v


class EnhanceTagger(SemanticBaseCamp):
    def __init__(self, data_gen: dg.DataGenerator, supervision: sp.supervision_provider, embeddings, word2idx, idx2word, device=Device):
        super().__init__(data_gen, supervision, embeddings, device)
        self.words_checked = 0
        self.words_verified = 0
        self.edit_diffs = self.prepare_edit_diffs()
        self.words_to_add = {}
        self.word2idx = word2idx
        self.idx2word = idx2word

    def tag(self):
        print('extracting semi-gold examples')

        thresholds, normed_averages, ys = self.train_pre_process()

        self.start = time.time()
        for j, (words1, words2) in enumerate(self.data_gen):
            if (j + 1) % self.coalesce_every == 0:
                self.print_procedure(j)

            cond = self.condition_from_words_idxs(words1, words2, thresholds, normed_averages)
            idxs = self.idxs_to_tag_from_condition(cond, words1, words2, ys)
            if not len(idxs):
                continue

            assert idxs.shape[1]%2==0

            # a little reshaping to separate idxs into 2 lists of corresponding word pairs and their label pairs
            half_length = idxs.shape[1]//2
            idxs = torch.stack((idxs[:, :half_length], idxs[:, half_length:]))
            caught_words = idxs[:, 0]   # shape: [2, ?]
            caught_labels = idxs[:, 1]  # shape: [2, ?]
            self.words_checked += min(caught_words.shape[1], MaxCheck)

            self.find_pairs_with_same_edits(caught_words[:, :MaxCheck], caught_labels[:, :MaxCheck])

    def find_pairs_with_same_edits(self, caught_words, caught_labels):

        for i in range(caught_words.shape[1]):
            word_slice = caught_words[:, i].view(-1,1)     # torch.LongTensor, shape=[2,1]
            words = [self.idx2word[x] for x in word_slice]
            if len(set(words[0]) & set(words[1]))<2:
                continue
            res = utils.unique_chars(*words)
            script = res[:2]
            common_chars = res[2]

            labels_slice = tuple(caught_labels[:, i].tolist())
            edit_ref, minimum_chars = self.edit_diffs[labels_slice]

            if script in edit_ref and common_chars >= minimum_chars:
                self.words_verified += 1
                rev_labels_slice = labels_slice[::-1]
                if labels_slice not in self.words_to_add:
                    self.words_to_add[labels_slice] = []
                    self.words_to_add[rev_labels_slice] = []

                self.words_to_add[labels_slice].append(word_slice)
                self.words_to_add[rev_labels_slice].append(torch.flip(word_slice, (0,)))

    def update_diffs(self):
        words_to_add = {' '.join([self.supervision.idx2label[i] for i in k]):v for k,v in self.words_to_add.items()}
        words_to_add = {k:torch.cat(v, dim=1) for k,v in words_to_add.items() if k in self.supervision.diffs.keys()}

        for label_pair in self.supervision.diffs:
            label1, label2 = label_pair.split()
            lab1_idxs = [self.word2idx.get(tag_word.word) for tag_word in self.supervision.words_matrix[self.supervision.labels[label1]]]
            lab2_idxs = [self.word2idx.get(tag_word.word) for tag_word in self.supervision.words_matrix[self.supervision.labels[label2]]]
            cond = [lab1_idxs[i] is not None and lab2_idxs[i] is not None and lab1_idxs[i]!=lab2_idxs[i] for i in range(len(lab1_idxs))]
            lab1_idxs = [l for i, l in enumerate(lab1_idxs) if cond[i]]
            lab2_idxs = [l for i, l in enumerate(lab2_idxs) if cond[i]]

            all_words = torch.cat((words_to_add.get(label_pair, torch.cuda.LongTensor(device=self.device)),
                                   torch.cuda.LongTensor([lab1_idxs, lab2_idxs], device=self.device)), dim=1)
            all_vecs = torch.nn.functional.embedding(all_words, self.embeddings)
            dif_vecs = (all_vecs[0] - all_vecs[1]).cpu().numpy()
            if len(dif_vecs)==1:
                average = dif_vecs.flatten()
                stats = (0.25, 0.3, 0.2)
            else:
                average = np.average(dif_vecs, axis=0)
                coses = [utils.cosine_dist(v, average) for v in dif_vecs]
                stats = (np.average(coses), np.max(coses), np.min(coses))
            self.supervision.diffs[label_pair] = sp.aggregated_pair(average, *stats)

        self.words_to_add = {}

    def prepare_edit_diffs(self):
        '''
        transform the edit_diffs dict from keyed by string to keyed by tuples of ints
        :param self.supervision.edit_diffs: keys: pairs of strings with blank space as separator.
                            values: tuples of "edit script"s and minimal common chars.
        :param self.supervision.labels:
        :return:
        '''
        return {tuple(self.supervision.labels[x] for x in k.split()): v for k, v in self.supervision.edit_diffs.items()}

    def load_hp(self, pair_data):
        if DeviationFromEnhance == 'avg':
            v = pair_data.avg_cos
        elif DeviationFromEnhance == 'max':
            v = pair_data.max
        else:
            raise NotImplementedError
        return v

    def print_procedure(self, j):
        self.words_to_add = {k:[torch.cat(v, dim=1)] for k,v in self.words_to_add.items()}
        super().print_procedure(j)
        print(f'{self.words_verified} added, out of {self.words_checked}.')
        self.words_checked = 0
        self.words_verified = 0
        lengths = [x[0].shape[1] for x in self.words_to_add.values()]
        if lengths:
            print(f'debug words_to_add: max {max(lengths)}, min {min(lengths)}, avg {sum(lengths)/len(lengths)}, sum {sum(lengths)}')
