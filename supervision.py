from collections import namedtuple
from hyper_params import *
from utils import *
from math import ceil

tagged_word = namedtuple('tagged_word', ['word', 'lemma', 'label'])
aggregated_pair = namedtuple('aggregated_pair', ['average', 'avg_cos', 'max', 'min'])

class supervision_provider():
    def __init__(self, word_vectors, path, num_supervision=MaxSupervision):
        self.num_supervision = num_supervision
        self.labels, self.lemmas, self.words_matrix, self.idx2label, self.labels_to_merge\
            = self.load_words(path, word_vectors)
        self.diffs = self.create_diffs(word_vectors)
        self.edit_diffs = self.create_edit_diffs()

    def load_words(self, path, word_vectors):
        '''
        :param path: path for text file from unimorph
        :return: words: {word: tagged_word(word, lemma, label)}
                 lemmas: {lemma: idx}
                 labels: {label: idx}
                 matrix: list of lists queryable by label and lemma
                         type(matrix[labels[label]][lemmas[lemma]]) == tagged_word
        '''
        words = {}  # {label: {words of this label from all lemmas}} we need that to check whether 2 labels are syncretic
        lemmas = {}
        labels = {}
        idx2label = []

        chosen_lemmas = choose_covered_lemmas(word_vectors.keys(), unimorph_path=path, num_supervision=self.num_supervision)
        print(chosen_lemmas)

        #first pass, just to count how many lemmas and labels are there
        with open(path, encoding='utf8') as f:
            for line in f:
                if not self.valid(line):
                    continue
                lemma, word, label = line.split()

                if lemma not in chosen_lemmas:
                    continue
                if not label.startswith(Paradigm):
                    continue

                if Language=='ru':
                    if label.startswith('V.PTCP'):
                        continue
                    if Paradigm == 'N':
                        if 'ANIM' in label:
                            continue
                        elif 'INAN' in label:
                            label = label.replace(';INAN;', ';')
                    if 'FUT' in label:
                        label = label.replace(';FUT;', ';PRS;')

                if Language == 'es' and Paradigm == 'V':
                    # excluding most of the verb-se reflexive forms (that have the same lemma in unimorph for some reason)
                    if word.endswith(('rse', 'dose', 'monos', 'nse', 'ate', 'ete', 'aos', 'eos', 'íos')):
                        continue

                if 'NEG' in label:  # mostly for turkish and latvian
                    continue

                if lemma not in lemmas and len(lemmas)<self.num_supervision:
                    lemmas[lemma] = len(lemmas)
                if lemma not in lemmas and len(lemmas)==self.num_supervision:
                    continue

                # if label not in labels:
                #     labels[label] = len(labels)
                #     idx2label.append(label)

                if label not in words:
                    words[label] = set()
                words[label].add(word)

        # turkish unimorph does not include the lemmas themselves in the tables
        if Language == 'tr':
            label = 'V;NFIN'
            words[label] = set(lemmas.keys())

        # now we check for labels with identical entries (syncretism)
        words_list = list(words.items())
        only_words_list = [x[1] for x in words_list]

        no_dupes = [x for n, x in enumerate(words_list) if x[1] not in only_words_list[:n]]
        dupes = [x for x in words_list if x not in no_dupes]
        only_words_list = [x[1] for x in no_dupes]

        labels_to_merge = {}
        for label, its_words in dupes:
            other_label = no_dupes[only_words_list.index(its_words)][0]
            if other_label not in labels_to_merge:
                labels_to_merge[other_label] = set()
            labels_to_merge[other_label].add(label)
        labels_to_merge = [{k}.union(v) for k,v in labels_to_merge.items()]

        for label, _ in no_dupes:
            labels[label] = len(labels)
            idx2label.append(label)

        # second pass, to load all the words and fill the matrix
        # access: matrix[label index][lemma index]
        matrix = [[tagged_word(None, None, None)]*len(lemmas) for _ in range(len(labels))]

        with open(path, encoding='utf8') as f:
            for line in f:
                if not self.valid(line):
                    continue
                lemma, word, label = line.split()
                if not label.startswith(Paradigm):
                    continue
                if Language=='ru':
                    if label.startswith('V.PTCP'):
                        continue
                    if Paradigm == 'N':
                        if 'ANIM' in label:
                            continue
                        elif 'INAN' in label:
                            label = label.replace(';INAN;', ';')
                    if 'FUT' in label:
                        label = label.replace(';FUT;', ';PRS;')
                if lemma in lemmas and label in labels:
                    # words[word] = tagged_word(word, lemma, label)
                    matrix[labels[label]][lemmas[lemma]] = tagged_word(word, lemma, label)
        if Language=='tr':
            for lemma in lemmas:
                matrix[labels['V;NFIN']][lemmas[lemma]] = tagged_word(lemma, lemma, 'V;NFIN')

        # assertions
        # rand_idxs = [5,2,15]
        # for label_idx in rand_idxs:
        #     label_idx = label_idx%len(labels)
        #     for lemma_idx in rand_idxs:
        #         lemma_idx = lemma_idx % len(lemmas)
        #         assert lemmas[matrix[label_idx][lemma_idx].lemma] == lemma_idx
        #         assert labels[matrix[label_idx][lemma_idx].label] == label_idx

        return labels, lemmas, matrix, idx2label, labels_to_merge

    def create_diffs(self, word_vectors):
        diffs = {}
        included_label_pairs = []
        for label1 in self.labels.keys():
            label1_vecs = [word_vectors.get(w.word, None) for w in self.words_matrix[self.labels[label1]]]
            for label2 in self.labels.keys():
                if label1 != label2 and {label1, label2} not in included_label_pairs:
                    label2_vecs = [word_vectors.get(w.word, None) for w in self.words_matrix[self.labels[label2]]]
                    dif_vecs = [special_deduction(label1_vecs[i],label2_vecs[i]) for i in range(len(label1_vecs))]
                    # first condition in below if is to exclude pairs where one of the words has no pre-trained vector
                    # the other condition is for cases where both words are the same (syncretism)
                    dif_vecs = [i for i in dif_vecs if i is not None and np.any(i != np.zeros_like(i))]

                    # in cases of syncretism, or lack of evidence - do nothing
                    if len(dif_vecs) > MinPairsToConsider:
                        average = np.average(dif_vecs, axis=0)
                        coses = [cosine_dist(v, average) for v in dif_vecs]
                        stats = (np.average(coses), np.max(coses), np.min(coses))
                        diffs[label1+' '+label2] = aggregated_pair(average, *stats)
                        # add pair to memory as a set, to avoid double work (and double scoring later on)
                        included_label_pairs.append({label1, label2})
                    elif len(dif_vecs) == 1:
                        average = dif_vecs[0]
                        stats = (0.25, 0.3, 0.2)
                        diffs[label1+' '+label2] = aggregated_pair(average, *stats)
                        # add pair to memory as a set, to avoid double work (and double scoring later on)
                        included_label_pairs.append({label1, label2})

        return diffs

    def valid(self, line):
        if not line:
            return False
        if not len(line.split()) == 3:
            return False
        return True

    def create_edit_diffs(self):
        edit_diffs = {}
        for label_pair in self.diffs.keys():
            label1, label2 = label_pair.split()
            '''
            reminder!
            self.word_matrix: list of lists queryable by label and lemma
                    type(self.word_matrix[self.labels[label]][self.lemmas[lemma]]) == tagged_word
            '''
            orig_words1 = [x.word for x in self.words_matrix[self.labels[label1]]]
            orig_words2 = [x.word for x in self.words_matrix[self.labels[label2]]]
            words1 = [orig_words1[i] for i in range(len(orig_words1)) if orig_words1[i] and orig_words2[i]]
            words2 = [orig_words2[i] for i in range(len(orig_words2)) if orig_words1[i] and orig_words2[i]]
            uni_chars = set([unique_chars(words1[i], words2[i]) for i in range(len(words1))])
            common_threshold = min([x[2] for x in uni_chars])
            uni_chars = {x[0:2] for x in uni_chars}
            rev_uni_chars = {(x[1], x[0]) for x in uni_chars}

            edit_diffs[label_pair] = (uni_chars, common_threshold)
            edit_diffs[label2+' '+label1] = (rev_uni_chars, common_threshold)

        return edit_diffs

    def get_relevant_edits(self):
        '''
        restructures the edit diffs. from {label_pair: {edits}} to {edit: {label_pairs}}
        '''
        res = {}
        for label_pair in self.edit_diffs:
            for edit in self.edit_diffs[label_pair][0]:
                if edit not in res:
                    res[edit] = set()
                res[edit].add(label_pair)
        return res

    def output_as_dataset(self, size=None):
        merged_labels = {l:l_set for l_set in self.labels_to_merge for l in l_set}

        train_pairs = set()
        for lemma in self.lemmas:
            relevant_words = [wl[self.lemmas[lemma]] for wl in self.words_matrix
                              if wl[self.lemmas[lemma]].word is not None]
            for word1 in relevant_words:
                for word2 in relevant_words:
                    if word1==word2:
                        continue
                    for label1 in merged_labels.get(word1.label, {word1.label}):
                        for label2 in merged_labels.get(word2.label, {word2.label}):
                            train_pairs.add((word1.word, label1, word2.word, label2))
        if not size or len(train_pairs)<size:
            return train_pairs

        per_relation = {}
        for pair in train_pairs:
            relation = pair[1] + ' ' + pair[3]
            if relation not in per_relation:
                per_relation[relation] = []
            per_relation[relation].append(pair)
        per_relation = {k:v[:ceil(size/len(per_relation))] for k,v in per_relation.items()}

        return set().union(*[set(v) for v in per_relation.values()])


# if __name__ == '__main__':
#     from initial_version.all_words_pool import word_vectors
#
#     sp = supervision_provider(word_vectors, AllUnimorphPath)
#     pairs = sp.output_as_dataset()
#     del word_vectors
#     p=1
