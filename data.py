# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import pandas as pd
import librosa
import glob
import os
import string
import itertools


__author__ = 'buriburisuri@gmail.com'


class VCTK(object):

    def __init__(self, batch_size=16, data_path='asset/data/'):

        @tf.sg_producer_func
        def _load_mfcc(src_list):
            lab, wav = src_list  # label, wave_file
            # decode string to integer
            lab = np.fromstring(lab, np.int)
            # load wave file
            wav, sr = librosa.load(wav, mono=True)
            # mfcc
            mfcc = librosa.feature.mfcc(wav, sr)
            # return result
            return lab, mfcc

        # load corpus
        labels, wave_files = self._load_corpus(data_path)

        # to constant tensor
        label = tf.convert_to_tensor(labels)
        wave_file = tf.convert_to_tensor(wave_files)

        # create queue from constant tensor
        label, wave_file = tf.train.slice_input_producer([label, wave_file], shuffle=True)

        # decode wave file
        label, mfcc = _load_mfcc(source=[label, wave_file], dtypes=[tf.sg_intx, tf.sg_floatx],
                                 capacity=128, num_threads=32)

        # create batch queue with dynamic pad
        batch_queue = tf.train.batch([label, mfcc], batch_size,
                                     shapes=[(None,), (20, None)],
                                     num_threads=32, capacity=batch_size*48,
                                     dynamic_pad=True)

        # split data
        self.label, self.mfcc = batch_queue
        # batch * time * dim
        self.mfcc = self.mfcc.sg_transpose(perm=[0, 2, 1])

        # calc total batch count
        self.num_batch = len(labels) // batch_size

        # print info
        tf.sg_info('VCTK corpus loaded.(total data=%d, total batch=%d)' % (len(labels), self.num_batch))

    def _load_corpus(self, data_path):

        # read meta-info
        df = pd.read_table(data_path + 'speaker-info.txt', usecols=['ID', 'AGE', 'GENDER', 'ACCENTS'],
                           index_col=False, delim_whitespace=True)

        # make file ID
        file_ids = []
        for d in [data_path + 'txt/p%d/' % uid for uid in df.ID.values]:
            file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

        # make wave file list
        wav_files = [data_path + 'wav48/%s/' % f[:4] + f + '.wav' for f in file_ids]

        # exclude extremely short wave files
        file_id, wav_file = [], []
        for i, w in zip(file_ids, wav_files):
            if os.stat(w).st_size > 240000:  # at least 5 seconds
                file_id.append(i)
                wav_file.append(w)

        # read label sentence
        sents = []
        for f in file_id:
            # remove punctuation, to lower, clean white space
            s = ' '.join(open(data_path + 'txt/%s/' % f[:4] + f + '.txt').read()
                         .translate(None, string.punctuation).lower().split())
            # append byte code
            sents.append([ord(ch) for ch in s])

        # make vocabulary
        self.index2byte = [0] + list(np.unique(list(itertools.chain(*sents))))  # add <EMP> token
        self.byte2index = {}
        for i, b in enumerate(self.index2byte):
            self.byte2index[b] = i
        self.voca_size = len(self.index2byte)
        self.max_len = np.max([len(s) for s in sents])

        # byte to index label
        label = []
        for s in sents:
            # save as string for variable-length support.
            label.append(np.asarray([self.byte2index[ch] for ch in s]).tostring())

        return label, wav_file

    def print_index(self, indices):
        for i, index in enumerate(indices):
            str_ = ''
            for ch in index:
                if ch > 0:
                    str_ += unichr(self.index2byte[ch])
                elif ch == 0:  # <EOS>
                    break
            print str_