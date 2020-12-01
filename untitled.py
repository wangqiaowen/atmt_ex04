import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import collections

import torch
import torch.nn as nn

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY
import re

import pickle

SPACE_NORMALIZER = re.compile("\s+")

def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('BPE Data pre-processing')

    # Add data arguments
    parser.add_argument('--data', default='/Users/wangqiaowen/atmt/baseline/bpe', help='path to data directory')
    parser.add_argument('--bpe-dropout-data', default='/Users/wangqiaowen/atmt/baseline/bpe/bpe_dropout', help='path to data')
    parser.add_argument('--source-lang', default='de', help='source language')
    parser.add_argument('--target-lang', default='en', help='target language')

    parser.add_argument('--train-prefix', default='/Users/wangqiaowen/atmt/baseline/bpe/bpe_dropout/bpe_train_dropout', help='train file prefix')
    parser.add_argument('--tiny-train-prefix', default='/Users/wangqiaowen/atmt/baseline/bpe/bpe_tiny_train', help='tiny train file prefix')
    parser.add_argument('--valid-prefix', default='/Users/wangqiaowen/atmt/baseline/bpe/bpe_dropout/bpe_valid', help='valid file prefix')
    parser.add_argument('--test-prefix', default='/Users/wangqiaowen/atmt/baseline/bpe/bpe_test', help='test file prefix')
    parser.add_argument('--dropout-prefix', default='/Users/wangqiaowen/atmt/baseline/bpe/bpe_train_dropout', help='test file prefix')

    return parser.parse_args()

def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def main(args): 

	src_dict = Dictionary.load(os.path.join(args.data, 'vocab_train.{:s}'.format(args.source_lang)))
	tgt_dict = Dictionary.load(os.path.join(args.data, 'vocab_train.{:s}'.format(args.target_lang)))

	def make_split_datasets(lang, dictionary):
		if args.train_prefix is not None:
			make_binary_dataset(args.train_prefix + '.' + lang, os.path.join(args.bpe_dropout_data, 'bpe_train_dropout_pkl.' + lang), dictionary)

		"""
		To use BPE-dropout, please comment out the next 6 lines of commented code before traning after data preprocessing.
		To use BPE without BPE-dropout, please uncomment the next 6 lines of commented code during data preprocessing.

		"""
		# if args.tiny_train_prefix is not None:
		# 	make_binary_dataset(args.tiny_train_prefix + '.' + lang, os.path.join(args.data, 'bpe_tiny_train_dropout_pkl.' + lang), dictionary)
		# if args.valid_prefix is not None:
		# 	make_binary_dataset(args.valid_prefix + '.' + lang, os.path.join(args.bpe_dropout_data, 'bpe_valid_dropout_pkl.' + lang), dictionary)
		# if args.test_prefix is not None:
		# 	make_binary_dataset(args.test_prefix + '.' + lang, os.path.join(args.data, 'bpe_test_dropout_pkl.' + lang), dictionary)
	make_split_datasets(args.source_lang, src_dict)
	make_split_datasets(args.target_lang, tgt_dict)


def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()

    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))

if __name__ == '__main__' : 

	args = get_args()
	main(args)
















