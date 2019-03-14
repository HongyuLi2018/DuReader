# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
import io


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """

    def __init__(self,
                 max_p_num,
                 max_p_len,
                 max_q_len,
                 train_files=[],
                 dev_files=[],
                 test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files

        self.train_set, self.dev_set, self.test_set = None, None, None
        if train_files:
            self.train_set = self._load_dataset(train_files, train=True)
            # self.logger.info('Train set size: {} questions.'.format(
            #     len(self.train_set)))

        if dev_files:
            self.dev_set = self._load_dataset(dev_files)
            # self.logger.info('Dev set size: {} questions.'.format(
            #     len(self.dev_set)))

        if test_files:
            self.test_set = self._load_dataset(test_files)
            # self.logger.info('Test set size: {} questions.'.format(
            #     len(self.test_set)))

    def _load_dataset(self, data_files, train=False):
        """
        Loads the dataset
        Args:
            data_files: list of data files to load
        """
        for data_file in data_files:
            with io.open(data_file, 'r', encoding='utf-8') as fin:
                for lidx, line in enumerate(fin):
                    sample = json.loads(line.strip())
                    if train:
                        if len(sample['answer_spans']) == 0:
                            continue
                        if sample['answer_spans'][0][1] >= self.max_p_len:
                            continue

                    if 'answer_docs' in sample:
                        sample['answer_passages'] = sample['answer_docs']

                    sample['question_tokens'] = sample['segmented_question']

                    sample['passages'] = []
                    for d_idx, doc in enumerate(sample['documents']):
                        if train:
                            most_related_para = doc['most_related_para']
                            sample['passages'].append({
                                'passage_tokens':
                                doc['segmented_paragraphs'][most_related_para],
                                'is_selected': doc['is_selected']
                            })
                        else:
                            most_related_para = doc['most_related_para']
                            sample['passages'].append(
                                {'passage_tokens': doc['segmented_paragraphs'][most_related_para]}
                            )
                    yield sample

    def _reset_dataset(self, set_name):
        """reset dataset after each epoch"""
        # shuffle dataset before re-loading
        if set_name == 'train':
            for filename in self.train_files:
                def _system_run(cmd):
                    self.logger.info("System command beginning: {}".format(cmd))
                    os.system(cmd)
                    self.logger.info("System command endding: {}".format(cmd))

                # shuf into .shuffle
                _system_run("shuf {0} -o {0}.shuffle".format(filename))
                # mv back into file
                _system_run("mv {0}.shuffle {0}".format(filename))

        if set_name == 'train':
            self.train_set = self._load_dataset(self.train_files, True)
        elif set_name == 'dev':
            self.dev_set = self._load_dataset(self.dev_files, False)
        elif set_name == 'test':
            self.test_set = self._load_dataset(self.test_files, False)
        else:
            raise RuntimeError

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {
            'raw_data': [data[i] for i in indices],
            'question_token_ids': [],
            'question_length': [],
            'passage_token_ids': [],
            'passage_length': [],
            'start_id': [],
            'end_id': [],
            'passage_num': []
        }
        max_passage_num = max(
            [len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            count = 0
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    count += 1
                    batch_data['question_token_ids'].append(sample[
                        'question_token_ids'][0:self.max_q_len])
                    batch_data['question_length'].append(
                        min(len(sample['question_token_ids']), self.max_q_len))
                    passage_token_ids = sample['passages'][pidx][
                        'passage_token_ids'][0:self.max_p_len]
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(
                        min(len(passage_token_ids), self.max_p_len))
            # record the start passage index of current sample
            passade_idx_offset = sum(batch_data['passage_num'])
            batch_data['passage_num'].append(count)
            gold_passage_offset = 0
            if 'answer_passages' in sample and len(sample['answer_passages']) and \
                    sample['answer_passages'][0] < len(sample['documents']):
                for i in range(sample['answer_passages'][0]):
                    gold_passage_offset += len(batch_data['passage_token_ids'][
                        passade_idx_offset + i])
                start_id = min(sample['answer_spans'][0][0], self.max_p_len)
                end_id = min(sample['answer_spans'][0][1], self.max_p_len)
                batch_data['start_id'].append(gold_passage_offset + start_id)
                batch_data['end_id'].append(gold_passage_offset + end_id)
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self._load_dataset(self.train_files + self.dev_files + self.test_files)
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(
                set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        pass

    def gen_mini_batches(self, set_name, batch_size, pad_id, vocab, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            vocab: vocabulary
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(
                set_name))

        self._reset_dataset(set_name)

        batch_data_set = []
        for sample in data:
            if len(batch_data_set) > 0 and len(batch_data_set) == batch_size:
                batch_indices = np.arange(batch_size)
                yield self._one_mini_batch(batch_data_set, batch_indices, pad_id)
                batch_data_set = []
            sample['question_token_ids'] = \
                vocab.convert_to_ids(sample['question_tokens'])
            for passage in sample['passages']:
                passage['passage_token_ids'] = \
                    vocab.convert_to_ids(passage['passage_tokens'])
            batch_data_set.append(sample)

        if len(batch_data_set) > 0:
            batch_indices = np.arange(len(batch_data_set))
            yield self._one_mini_batch(batch_data_set, batch_indices, pad_id)

