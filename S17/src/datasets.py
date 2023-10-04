import random
import re
from collections import Counter
from os.path import exists

import torch
from torch.utils.data import DataLoader, Dataset


class SentencesDataset(Dataset):
    def __init__(self, sentences, vocab, seq_len):
        dataset = self
        dataset.sentences = sentences
        dataset.seq_len = seq_len

        # Create vocabulary
        dataset.vocab = vocab + ["<ignore>", "<oov>", "<mask>"]
        dataset.vocab = {e: i for i, e in enumerate(dataset.vocab)}
        dataset.rvocab = {i: e for i, e in enumerate(dataset.vocab)}

        # Special Tags
        dataset.IGNORE_IDX = dataset.vocab["<ignore>"]
        dataset.OOV_IDX = dataset.vocab["<oov>"]
        dataset.MASK_IDX = dataset.vocab["<mask>"]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx, p_random_mask=0.15):
        dataset = self

        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(idx % len(dataset)))
            idx += 1

        # Ensure that the sequence is of length seq_len
        s = s[: dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for _ in range(dataset.seq_len - len(s))]

        # Apply random masking
        s = [
            (dataset.MASK_IDX, w)
            if random.random() < p_random_mask
            else (w, dataset.IGNORE_IDX)
            for w in s
        ]

        return {
            "input_ids": torch.Tensor([w[0] for w in s]).long(),
            "target_ids": torch.Tensor([w[1] for w in s]).long(),
        }

    def get_sentence_idx(self, idx):
        dataset = self

        # Get sentence
        sentence = dataset.sentences[idx]

        sentence = [
            dataset.vocab[w] if w in dataset.vocab else dataset.OOV_IDX
            for w in sentence
        ]
        return sentence


def prepare_sentences_dataset(dataset_path, vocab_path, vocab_size):
    print("Loading dataset from", dataset_path)
    sentences = open(dataset_path).read().lower().split("\n")

    # Tokenize
    print("Tokenizing dataset!")
    special_chars = ",?;.:/*!+-()[]{}\"'&"
    sentences = [
        re.sub(f"[{re.escape(special_chars)}]", " \g<0> ", s).split(" ")
        for s in sentences
    ]

    sentences = [[w for w in s if len(w)] for s in sentences]

    # Create vocabulary

    if not exists(vocab_path):
        print("Creating vocabulary!")
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(vocab_size)
        vocab = [w[0] for w in vocab]
        open(vocab_path, "w+").write("\n".join(vocab))
    else:
        print("Loading vocabulary from", vocab_path)
        vocab = open(vocab_path).read().split("\n")

    return sentences, vocab
