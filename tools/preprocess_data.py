# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.data.indexed_dataset import best_fitting_dtype
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            # NOTE: 在这里添加 EOD
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    # 训练数据集 json 的路径
    group.add_argument('--input', type=str,
                       help='Path to input JSON')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    # 是否将文档拆分成句子，BERT 最好是拆分
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    # tokenizer类型：GPT: GPT2BPETokenizer
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    # 某些分布式训练框架要求词汇表大小能被某个数整除
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach. This value must be greater than'
                            ' the initial size of the tokenizer. If this argument is used the value of '
                            '`make-vocab-size-divisible-by` will be ignored.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    # 在每个子进程中执行初始化，初始化 Encoder.tokenizer 和 Encoder.splitter 用于 encoder.encode 使用
    '''
    在多进程环境下，主进程和子进程有独立的内存空间。
    即使在主进程中创建了 tokenizer 对象，它也不会自动被共享到每个子进程中。
    子进程的内存是隔离的，它们无法访问主进程的变量或对象。
    即使在主进程中正确初始化了 tokenizer，子进程仍然无法使用它，除非显式地通过 initializer 在每个子进程中进行初始化。
    '''
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    '''
    imap 会按顺序逐一传递给定的输入数据（fin 中的每一行）给 encoder.encode 函数处理
    每次会将 25 行数据传递给一个工作进程进行处理
    pool.imap 创建一个 惰性迭代器
    - 不会立即执行 encoder.encode 方法，而是会创建一个迭代器，延迟执行实际的编码工作
    - pool.imap() 会返回一个迭代器，该迭代器会异步地将任务（即 encoder.encode）分发给 multiprocessing.Pool 中的多个子进程。
    - 当迭代 encoded_docs 时，它会依次将 fin 文件中的数据行（每一行传递给 encoder.encode）传递给池中的工作进程，执行编码操作。
    '''
    encoded_docs = pool.imap(encoder.encode, fin, 25)
    #encoded_docs = map(encoder.encode, fin)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                    key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                    key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],  # 'meg-gpt2_text_document.bin'
                                                     impl=args.dataset_impl,  # 'mmap'
                                                     dtype=best_fitting_dtype(tokenizer.vocab_size))

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        # 正式多进程处理数据，并把 tokenize 后的 np.array 数据写入到文件中
        # 文件每行是一个 np.array，包含一个文档的 token id
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                # add_item 就是做了一个写入操作
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()  # 把 jsonl 数据条数记录到 idx 文件里，[0, jsonl数据条数]
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                f"({i/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)

    for key in args.json_keys:
        # 保存数据索引，每一行是一个文档的长度
        builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    '''
    python3 tools/preprocess_data.py \
        --input oscar-1GB.jsonl \
        --output-prefix meg-gpt2 \
        --vocab gpt2-vocab.json \
        --dataset-impl mmap \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file gpt2-merges.txt \
        --append-eod \
        --workers 8

    '''
    # import sys
    # sys.argv = [
    #     'preprocess_data.py', 
    #     '--input', '/data/cheyujie/github/datasets/oscar-1GB.jsonl', 
    #     '--output-prefix', 'meg-gpt2',
    #     '--vocab', '/data/cheyujie/github/datasets/gpt2-vocab.json', 
    #     '--dataset-impl', 'mmap',
    #     '--tokenizer-type', 'GPT2BPETokenizer',
    #     '--merge-file', '/data/cheyujie/github/datasets/gpt2-merges.txt',
    #     '--append-eod',
    #     '--workers', '8'
    # ]

    main()
