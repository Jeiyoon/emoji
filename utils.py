"""
@author: eagle705
https://github.com/eagle705/pytorch-bert-crf-ner/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import torch
from pathlib import Path

from tensorflow import keras
import numpy as np
from konlpy.tag import Twitter
from collections import Counter
from threading import Thread

import six
from torch import nn

class Config:
    def __init__(self, json_path):
        with open(json_path, mode='r') as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, mode='w') as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path):
        with open(json_path, mode='r') as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

class CheckpointManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir

    def save_checkpoint(self, state, filename):
        torch.save(state, self._model_dir / filename)

    def load_checkpoint(self, filename):
        state = torch.load(self._model_dir / filename, map_location=torch.device('cpu'))
        return state

class SummaryManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename):
        with open(self._model_dir / filename, mode='w') as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename):
        with open(self._model_dir / filename, mode='r') as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary):
        self._summary.update(summary)

    def reset(self):
        self._summary = {}

    @property
    def summary(self):
        return self._summary



class Vocabulary(object):
    """Vocab Class"""

    def __init__(self, token_to_idx=None):

        self.token_to_idx = {}
        self.idx_to_token = {}
        self.idx = 0

        self.PAD = self.padding_token = "[PAD]"
        self.START_TOKEN = "<S>"
        self.END_TOKEN = "<T>"
        self.UNK = "[UNK]"
        self.CLS = "[CLS]"
        self.MASK = "[MASK]"
        self.SEP = "[SEP]"
        self.SEG_A = "[SEG_A]"
        self.SEG_B = "[SEG_B]"
        self.NUM = "<num>"

        self.cls_token = self.CLS
        self.sep_token = self.SEP

        self.special_tokens = [self.PAD,
                               self.START_TOKEN,
                               self.END_TOKEN,
                               self.UNK,
                               self.CLS,
                               self.MASK,
                               self.SEP,
                               self.SEG_A,
                               self.SEG_B,
                               self.NUM]
        self.init_vocab()

        if token_to_idx is not None:
            self.token_to_idx = token_to_idx
            self.idx_to_token = {v: k for k, v in token_to_idx.items()}
            self.idx = len(token_to_idx) - 1

            # if pad token in token_to_idx dict, get pad_id
            if self.PAD in self.token_to_idx:
                self.PAD_ID = self.transform_token2idx(self.PAD)
            else:
                self.PAD_ID = 0

    def init_vocab(self):
        for special_token in self.special_tokens:
            self.add_token(special_token)
        self.PAD_ID = self.transform_token2idx(self.PAD)

    def __len__(self):
        return len(self.token_to_idx)

    def to_indices(self, tokens):
        return [self.transform_token2idx(X_token) for X_token in tokens]

    def add_token(self, token):
        if not token in self.token_to_idx:
            self.token_to_idx[token] = self.idx
            self.idx_to_token[self.idx] = token
            self.idx += 1

    def transform_token2idx(self, token, show_oov=False):
        try:
            return self.token_to_idx[token]
        except:
            if show_oov is True:
                print("key error: " + str(token))
            token = self.UNK
            return self.token_to_idx[token]

    def transform_idx2token(self, idx):
        try:
            return self.idx_to_token[idx]
        except:
            print("key error: " + str(idx))
            idx = self.token_to_idx[self.UNK]
            return self.idx_to_token[idx]

    def build_vocab(self, list_of_str, threshold=1, vocab_save_path="./data_in/token_vocab.json",
                    split_fn=Twitter().morphs):
        """Build a token vocab"""

        def do_concurrent_tagging(start, end, text_list, counter):
            for i, text in enumerate(text_list[start:end]):
                text = text.strip()
                text = text.lower()

                try:
                    tokens_ko = split_fn(text)
                    # tokens_ko = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
                    counter.update(tokens_ko)

                    if i % 1000 == 0:
                        print("[%d/%d (total: %d)] Tokenized input text." % (
                            start + i, start + len(text_list[start:end]), len(text_list)))

                except Exception as e:  # OOM, Parsing Error
                    print(e)
                    continue

        counter = Counter()

        num_thread = 4
        thread_list = []
        num_list_of_str = len(list_of_str)
        for i in range(num_thread):
            thread_list.append(Thread(target=do_concurrent_tagging, args=(
                int(i * num_list_of_str / num_thread), int((i + 1) * num_list_of_str / num_thread), list_of_str,
                counter)))

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        # vocab_report
        print(counter.most_common(10))  # print most common tokens
        tokens = [token for token, cnt in counter.items() if cnt >= threshold]

        for i, token in enumerate(tokens):
            self.add_token(str(token))

        print("len(self.token_to_idx): ", len(self.token_to_idx))

        import json
        with open(vocab_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_idx, f, ensure_ascii=False, indent=4)

        return self.token_to_idx




def keras_pad_fn(token_ids_batch, maxlen, pad_id=0, padding='post', truncating='post'):
    padded_token_ids_batch = pad_sequences(token_ids_batch,
                                            value=pad_id,  # vocab.transform_token2idx(PAD),
                                            padding=padding,
                                            truncating=truncating,
                                            maxlen=maxlen)
    return padded_token_ids_batch

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x



class Tokenizer:
    """ Tokenizer class"""

    def __init__(self, vocab, split_fn, pad_fn, maxlen):
        self._vocab = vocab
        self._split = split_fn
        self._pad = pad_fn
        self._maxlen = maxlen

    # def split(self, string: str) -> list[str]:
    def split(self, string):
        tokens = self._split(string)
        return tokens

    # def transform(self, list_of_tokens: list[str]) -> list[int]:
    def transform(self, tokens):
        indices = self._vocab.to_indices(tokens)
        pad_indices = self._pad(indices, pad_id=0, maxlen=self._maxlen) if self._pad else indices
        return pad_indices

    # def split_and_transform(self, string: str) -> list[int]:
    def split_and_transform(self, string):
        return self.transform(self.split(string))

    @property
    def vocab(self):
        return self._vocab

    def list_of_tokens_to_list_of_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_ids_batch.append([self._vocab.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_list_of_tokens(self, X_str_batch):
        X_token_batch = [self._split(X_str) for X_str in X_str_batch]
        return X_token_batch

    def list_of_tokens_to_list_of_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_ids_batch.append([self._vocab.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_list_token_ids(self, X_str_batch):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        X_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_token_batch)

        return X_ids_batch

    def list_of_string_to_arr_of_pad_token_ids(self, X_str_batch, add_start_end_token=False):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        # print("X_token_batch: ", X_token_batch)
        if add_start_end_token is True:
            return self.add_start_end_token_with_pad(X_token_batch)
        else:
            X_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_token_batch)
            pad_X_ids_batch = self._pad(X_ids_batch, pad_id=self._vocab.PAD_ID, maxlen=self._maxlen)

        return pad_X_ids_batch

    def list_of_tokens_to_list_of_cls_sep_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_tokens = [self._vocab.cls_token] + X_tokens + [self._vocab.sep_token]
            X_ids_batch.append([self._vocab.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_arr_of_cls_sep_pad_token_ids(self, X_str_batch):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        X_ids_batch = self.list_of_tokens_to_list_of_cls_sep_token_ids(X_token_batch)
        pad_X_ids_batch = self._pad(X_ids_batch, pad_id=self._vocab.PAD_ID, maxlen=self._maxlen)

        return pad_X_ids_batch

    def list_of_string_to_list_of_cls_sep_token_ids(self, X_str_batch):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        X_ids_batch = self.list_of_tokens_to_list_of_cls_sep_token_ids(X_token_batch)

        return X_ids_batch

    def add_start_end_token_with_pad(self, X_token_batch):
        dec_input_token_batch = [[self._vocab.START_TOKEN] + X_token for X_token in X_token_batch]
        dec_output_token_batch = [X_token + [self._vocab.END_TOKEN] for X_token in X_token_batch]

        dec_input_token_batch = self.list_of_tokens_to_list_of_token_ids(dec_input_token_batch)
        pad_dec_input_ids_batch = self._pad(dec_input_token_batch, pad_id=self._vocab.PAD_ID, maxlen=self._maxlen)

        dec_output_ids_batch = self.list_of_tokens_to_list_of_token_ids(dec_output_token_batch)
        pad_dec_output_ids_batch = self._pad(dec_output_ids_batch, pad_id=self._vocab.PAD_ID, maxlen=self._maxlen)
        return pad_dec_input_ids_batch, pad_dec_output_ids_batch

    def decode_token_ids(self, token_ids_batch):
        list_of_token_batch = []
        for token_ids in token_ids_batch:
            token_token = [self._vocab.transform_idx2token(token_id) for token_id in token_ids]
            # token_token = [self._vocab[token_id] for token_id in token_ids]
            list_of_token_batch.append(token_token)
        return list_of_token_batch


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 7,
                 dr_rate = None,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))

        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler

        return self.classifier(out)