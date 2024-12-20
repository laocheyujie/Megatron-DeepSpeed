# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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

"""Tokenization classes for OpenAI GPT."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import os
import regex as re
from io import open

from megatron import logging

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE
    # tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


logger = logging.get_logger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
}
PRETRAINED_MERGES_ARCHIVE_MAP = {
    'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'gpt2': 1024,
}
VOCAB_NAME = 'vocab.json'
MERGES_NAME = 'merges.txt'
SPECIAL_TOKENS_NAME = 'special_tokens.txt'


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.

    bytes_to_unicode 返回的是一个字典，该字典将每个字节（0-255 范围）映射到一个 Unicode 字符。
    这是一个可逆的转换，字节可以被转换为 Unicode 字符，也可以通过反向映射进行逆向转换。
    """
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).

    用于从一个单词中提取出所有相邻的字符对（bigram）。
    它通过遍历单词的字符，组合出相邻字符对，并将这些对存储在一个集合中。
    在 BPE 分词中，这些字符对是后续合并操作的基础。BPE 算法会根据字符对出现的频率进行合并，从而优化词汇表并提高分词效率。
    """
    pairs = set()  # 初始化一个空的集合，用来存储相邻字符对
    prev_char = word[0]  # 取单词的第一个字符作为初始字符
    for char in word[1:]:  # 从第二个字符开始遍历
        pairs.add((prev_char, char))  # 将前一个字符和当前字符的元组添加到 pairs 集合中
        prev_char = char  # 更新 prev_char 为当前字符
    return pairs  # 返回所有的字符对集合


class GPT2Tokenizer(object):
    """
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level BPE
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
            merges_file = PRETRAINED_MERGES_ARCHIVE_MAP[pretrained_model_name_or_path]
            special_tokens_file = None
        else:
            vocab_file = os.path.join(pretrained_model_name_or_path, VOCAB_NAME)
            merges_file = os.path.join(pretrained_model_name_or_path, MERGES_NAME)
            special_tokens_file = os.path.join(pretrained_model_name_or_path, SPECIAL_TOKENS_NAME)
            if not os.path.exists(special_tokens_file):
                special_tokens_file = None
            else:
                logger.info("loading special tokens file {}".format(special_tokens_file))
        # redirect to the cache, if necessary
        try:
            from .file_utils import cached_path
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
            resolved_merges_file = cached_path(merges_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find files {} and {} "
                "at this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    pretrained_model_name_or_path,
                    vocab_file, merges_file))
            return None
        if resolved_vocab_file == vocab_file and resolved_merges_file == merges_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
            logger.info("loading merges file {}".format(merges_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
            logger.info("loading merges file {} from cache at {}".format(
                merges_file, resolved_merges_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        if special_tokens_file and 'special_tokens' not in kwargs:
            special_tokens = open(special_tokens_file, encoding='utf-8').read().split('\n')[:-1]
        else:
            special_tokens = kwargs.pop('special_tokens', [])
        tokenizer = cls(
            resolved_vocab_file,
            resolved_merges_file,
            special_tokens=special_tokens,
            *inputs,
            **kwargs)
        return tokenizer


    def __init__(self, vocab_file, merges_file, errors='replace',
                 special_tokens=None, max_len=None, max_token_len_cache=9):
        """
        max_token_len_cache determines whether a normalized token will be cached. It tries to only store shorter tokens in the cache, 
        with the heuristic that they are more frequent. Increasing this may make tokenization faster but will also take more memory. 
        The upper bound of the normalized token cache is fixed at 1_000_000 tokens.

        max_token_len_cache: bpe时，只缓存长度小于等于max_token_len_cache的token
        """
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # 存储了字符对的合并优先级，排名越靠前的字符对越优先合并。
        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        # 会从输入的文本中提取出字符、单词、标点符号等各种 token
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)
        self.max_token_len_cache = max_token_len_cache

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i)
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        logger.info("Special tokens {}".format(self.special_tokens))
    
    @lru_cache(1_000_000)
    def bpe(self, token):
        '''
        bpe 函数通过逐步合并频繁出现的字符对，将输入的单词（或子词）分解为更小的单元，直到无法继续合并为止。
        这个过程使得分词能够在字符级别进行优化，减少了词汇表大小，同时也能够处理未见过的词
        lover -> l o v e r -> l o v er -> l o ver -> l over
        '''
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # 找到最频繁出现的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                # merge 表里没出现过就退出
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except BaseException:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        return word
      
    @lru_cache(1_000_000)
    def normalize_token_and_cache(self, token):
        return self.normalize_token(token)
    
    def normalize_token(self, token):
        # '爱' -> [231, 136, 177] -> ['ç', 'Ī', '±']
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        ret = [bpe_token for bpe_token in self.bpe(token).split(' ')]
        return ret

    def tokenize(self, text):
        """ Tokenize a string. 
        将输入的文本（字符串）转换为对应的 token 列表。
        I love you -> re: ['I', 'love', 'you'] -> bpe_tokens: ['I', 'l', 'ove', 'you]
        """
        # max_token_len_cache：表示缓存 token 时的最大长度，用于控制缓存的大小，避免缓存过长的 token 以占用过多内存
        max_token_len_cache = self.max_token_len_cache
        bpe_tokens = []
        # 提取文本 text 中的 token
        if sys.version_info[0] == 2:
          for token in re.findall(self.pat, text):        
              token = ''.join(self.byte_encoder[ord(b)] for b in token)
              bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
          return bpe_tokens
        for token in re.findall(self.pat, text):
          if len(token) <= max_token_len_cache:
              bpe_tokens.extend(self.normalize_token_and_cache(token))
          else: 
              bpe_tokens.extend(self.normalize_token(token))
        # 返回 bpe_tokens 列表，包含了对文本进行分词后得到的所有 tokens
        return bpe_tokens
      
    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            # 确保输入的 token 序列不会超出最大的 sequence length
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(
                    len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        # 1. tokenize: 
        #   1.1 根据 re 将 text 分割为 tokens 列表
        #   1.2 将每个 token 转换为 BPE tokens 列表: 将 token 拆分为更小的子词单元
        # 2. convert_tokens_to_ids: 将 bpe tokens 列表根据转换为 ids 列表
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(vocab_path):
            logger.error("Vocabulary path ({}) should be a directory".format(vocab_path))
            return
        vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        merge_file = os.path.join(vocab_path, MERGES_NAME)
        special_tokens_file = os.path.join(vocab_path, SPECIAL_TOKENS_NAME)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        index = len(self.encoder)
        with open(special_tokens_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.special_tokens.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving special tokens vocabulary to {}: BPE indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(special_tokens_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1

        return vocab_file, merge_file, special_tokens_file
