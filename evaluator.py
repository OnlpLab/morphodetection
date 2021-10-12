from hyper_params import *
from utils import *


class evaluation_class():
    def __init__(self, answers_path, idx2word):
        self.answers_path = answers_path
        self.gold_labels, self.gold_lemmas_by_word, self.gold_lemmas_by_lemma = self.create_gold(idx2word)

    def create_gold(self, idx2word):
        '''
        :param idx2word: list of all words in play
        :return: {word: {gold_labels} }
                 {word: lemma}
        '''
        idx2word = set(idx2word)
        gold_labels = {}
        gold_lemmas_by_word = {}
        gold_lemmas_by_lemma = {}
        for fp in self.answers_path:
            with open(fp, encoding='utf8') as f:
                for line in f:
                    if len(line.split()) != 3:
                        continue
                    lemma, word, tags = line.split()
                    if Language == 'ru':
                        if tags.startswith('V.PTCP'):
                            continue
                        if Paradigm == 'N':
                            if 'ANIM' in tags:
                                continue
                            elif 'INAN' in tags:
                                tags = tags.replace(';INAN;', ';')
                    if word in idx2word:
                        if word in gold_labels:
                            gold_labels[word].add(tags)
                        else:
                            gold_labels[word] = {tags}

                        if word in gold_lemmas_by_word:
                            gold_lemmas_by_word[word].add(lemma)
                        else:
                            gold_lemmas_by_word[word] = {lemma}
                        if lemma in gold_lemmas_by_lemma:
                            gold_lemmas_by_lemma[lemma].add(word)
                        else:
                            gold_lemmas_by_lemma[lemma] = {word}

        # gold_labels = {w:set() for w in answers_path.words.keys()}
        # for tagged_word in answers_path.words.values():
        #     word = tagged_word.word
        #     label = tagged_word.label
        #     gold_labels[word].add(label)
        return gold_labels, gold_lemmas_by_word, gold_lemmas_by_lemma

    def tag(self, scores, scoring_threshold):
        '''
        :param scores: dict {word: {label: score} }
        :return: dict {word: {labels} }
        '''
        labels = {}
        for word in scores:
            word_scores = scores[word]
            word_scores = {l:s for l,s in word_scores.items() if s>=scoring_threshold}
            word_labels = set(word_scores.keys())
            labels[word] = word_labels

        return labels

    def evaluate(self, labels, total_words_num, total_unimorph_labels_num, gold_labels=None):

        if not gold_labels:
            gold_labels = self.gold_labels

        total = total_unimorph_labels_num*total_words_num
        total_positive = sum([len(ls) for ls in labels.values()])
        total_negative = total-total_positive

        if total_positive == 0:
            return None, None, None, None

        correct_labels = {word: labels[word]&gold_labels.get(word, set()) for word in labels}
        incorrect_labels = {word: labels[word]-correct_labels[word] for word in labels}

        true_positive = sum([len(ls) for ls in correct_labels.values()])
        false_positive = sum([len(ls) for ls in incorrect_labels.values()])

        assert total_positive == true_positive+false_positive

        missed_labels = {word: gold_labels[word]-labels.get(word, set()) for word in gold_labels}

        false_negative = sum([len(ls) for ls in missed_labels.values()])
        true_negative = total_negative - false_negative

        assert total == true_positive+true_negative+false_negative+false_positive
        assert sum([len(ls) for ls in gold_labels.values()]) == true_positive + false_negative

        precision = true_positive / total_positive
        recall = true_positive / (true_positive + false_negative)
        accuracy = (true_positive + true_negative) / total

        if not precision and recall and accuracy and total_positive:
            print('oy!')

        return precision, recall, accuracy, total_positive

    def write_statistics(self, scores, log_path, max_threshold, len_word_vectors, labels_num, printing=True):
        with open(log_path, 'w') as logfile:
            # todo write some meta data

            for scoring_threshold in range(1, max_threshold):
                labels = self.tag(scores, scoring_threshold)
                precision, recall, accuracy, total_tagged = self.evaluate(labels, len_word_vectors, labels_num)

                if precision is None or precision+recall==0:
                    continue

                f1 = 2 * precision * recall / (precision + recall)
                if printing:
                    print('precision {:.2f}'.format(precision))
                    print('recall {:.2f}'.format(recall))
                    print('F1 score {:.2f}'.format(f1))

                logfile.write('scoring_threshold - {} '.format(scoring_threshold))
                # logfile.write('precision {0:.2f}, recall {1:.2f}, F1 score {2:.2f}\n'.format(precision, recall, f1))
                logfile.write(f'precision {precision:.2f}, recall {recall:.2f}, F1 score {f1:.2f}\n')

        print(os.path.basename(log_path), 'written')

# if __name__ == '__main__':
#     supervision = supervision_provider(word_vectors)
#     # trainer = training_class(word_vectors)
#
#     scores = {'comer':   {'V;NFIN': 55,         # correct
#                           'V;IND;FUT;3;SG': 2},
#               'coman':   {'V;POS;IMP;3;PL': 60, # correct
#                           'V;SBJV;PRS;3;PL': 58,# correct
#                           'V;NFIN': 55},
#               'como':    {'V;IND;PRS;1;SG': 30},# correct
#               'hablarás':{'V;IND;FUT;2;SG': 49},# correct
#               'haremos' :{'V;IND;FUT;1;PL': 52, # correct
#                           'V;COND;1;PL': 54},
#               'llegase' :{'V;SBJV;PST;1;SG': 58,# correct
#                           'V;IND;PRS;1;PL': 12}
#               }
#     evaluator = evaluation_class(AnswersPath, word_vectors)
#
#     labels_num = len(supervision.labels)
#     scoring_threshold = 40
#     # for scoring_threshold in range(labels_num):
#     print('scoring threshold ', scoring_threshold)
#     labels = evaluator.tag(scores, scoring_threshold)
#     precision, recall, accuracy = evaluator.evaluate(labels, len(word_vectors), labels_num)
#
#     print('precision {:.2f}'.format(precision))
#     print('recall {:.2f}'.format(recall))
#     print('accuracy {:.2f}'.format(accuracy))

