from hyper_params import *
from abc import abstractmethod
from typing import Dict
import torch
from math import ceil
from random import shuffle
from copy import deepcopy


class DataGenerator:
    def __init__(self, word2idx: Dict[str, int], batch_size):
        self.word2idx = word2idx
        self.words = deepcopy(word2idx)
        self.words.pop(PlaceHolderToken, None)
        self.batch_size = batch_size

        self.generator = self.create_gen()

    @abstractmethod
    def create_gen(self):
        pass

    def __iter__(self):
        self.generator = self.create_gen()
        return self

    def __next__(self):
        return next(self.generator)


class SurfaceData(DataGenerator):
    def __init__(self, word2idx: Dict[str, int], batch_size):
        super().__init__(word2idx, batch_size)

        print(f'{len(self.words)} words in original dictionary.\nreducing amount to {LimitWordsForSurface}.')
        self.tot_words = len(self.words)
        self.words = list(self.words.keys())
        shuffle(self.words)
        self.words = [w for w in self.words if len(w) > OverlappedLetters][:LimitWordsForSurface]

    def create_gen(self):
        for word1 in self.words:
            set_word1 = set(word1)
            for word2 in self.words:
                set_word2 = set(word2)
                cross = set_word1 & set_word2
                if len(cross) <= OverlappedLetters:
                    yield None

                yield word1, word2


class BatchedData(DataGenerator):
    def __init__(self, word2idx: Dict[str, int], batch_size, device=Device):
        super().__init__(word2idx, batch_size)
        self.device = device
        self.words = list(self.words.values())

        self.padded_words = torch.cuda.LongTensor(self.words + [0]*(self.batch_size - (len(self.words) % self.batch_size)),
                                                  device=self.device)
        self.words = torch.cuda.LongTensor(self.words, device=self.device)

    def create_gen(self):
        num_batches_per_word = ceil(len(self.words) / self.batch_size)
        for word1 in self.words:
            words1 = torch.cuda.LongTensor(self.batch_size, device=self.device).fill_(word1)
            for i in range(num_batches_per_word):
                yield words1, self.padded_words[i * self.batch_size:(i + 1) * self.batch_size]
