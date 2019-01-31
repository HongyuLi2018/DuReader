#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
This module upgrades the basic BIDAF/match-LSTM model with a simplified verification
 technique based on the VNET (see https://arxiv.org/abs/1805.02220)
"""

import os
import time
import logging
import json

import numpy as np
import tensorflow as tf

from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
from layers import func
from rc_model import RCModel


class VerifiedRCModel(RCModel):
    """Basic models with passage verification"""
    def __init__(self, vocab, args):
        self.beta = args.beta
        super(VerifiedRCModel, self).__init__(vocab, args)

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._verify()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        super(VerifiedRCModel, self)._setup_placeholders()
        self.para_label = tf.placeholder(tf.int32, [None])
        self.p_number = tf.placeholder(tf.int32, [None])
        self.p_mask = tf.sequence_mask(self.p_length, maxlen=tf.shape(self.p)[1], dtype=tf.float32)
        self.q_mask = tf.sequence_mask(self.q_length, maxlen=tf.shape(self.q)[1], dtype=tf.float32)
        self.pp_mask = tf.sequence_mask(
            self.p_number, maxlen=tf.shape(self.q)[0] // tf.shape(self.start_label)[0],
            dtype=tf.float32
        )

    def _verify(self):
        """
        Paragraph verification
        """
        with tf.variable_scope('qp_verify'):
            q = self.sep_q_encodes
            match = self.fuse_p_encodes
            d = self.hidden_size

            init = func.summ(
                q, d, mask=self.q_mask, keep_prob=self.dropout_keep_prob, is_train=self.use_dropout
            )
            psgs = func.summ(
                match, d, mask=self.p_mask, init=init, keep_prob=self.dropout_keep_prob,
                is_train=self.use_dropout, scope="summ2"
            )
            psgs = tf.reshape(
                psgs, [tf.shape(self.start_label)[0], -1, psgs.shape[-1].value]
            )
            psgs = func.dense(psgs, 1, True, scope="v0")
            self.reshaped_ans_verif_logit = tf.reshape(
                psgs, [tf.shape(self.start_label)[0], -1]
            )
            self.reshaped_ans_verif_logit += (1. - self.pp_mask) * (-1e9)
            # self.reshaped_ans_verif_score = tf.nn.softmax(self.reshaped_ans_verif_logit, 1)
            self.reshaped_ans_verif_score = tf.nn.sigmoid(self.reshaped_ans_verif_logit)

    def _compute_loss(self):
        """
        The loss function combines the boundary loss and the verification loss
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelihood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        def ce_loss(probs, labels, mask, epsilon=1e-9, scope=None):
            """cross entropy loss"""
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1])
                all_losses = labels * tf.log(probs + epsilon) + (1 - labels) * tf.log(1 - probs + epsilon)
                all_losses *= mask
                losses = - tf.reduce_sum(all_losses, -1) / (tf.reduce_sum(mask, -1) + epsilon)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.boundary_loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        self.verify_loss = tf.reduce_mean(
            ce_loss(self.reshaped_ans_verif_score, self.para_label, self.pp_mask)
        )
        self.loss = self.boundary_loss + self.beta * self.verify_loss
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.para_label: batch['ans_para_id'],
                         self.p_number: batch['passage_number'],
                         self.dropout_keep_prob: dropout_keep_prob}
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.para_label: batch['ans_para_id'],
                         self.p_number: batch['passage_number'],
                         self.dropout_keep_prob: 1.0}
            start_probs, end_probs, verif_scores, loss = self.sess.run(
                [self.start_probs, self.end_probs, self.reshaped_ans_verif_score, self.loss],
                feed_dict
            )

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sidx in range(len(batch['raw_data'])):
                sample = batch['raw_data'][sidx]
                start_prob = start_probs[sidx]
                end_prob = end_probs[sidx]
                verif_score = verif_scores[sidx]
                best_answer = self.find_best_verified_answer(
                    sample, start_prob, end_prob, verif_score, padded_p_len,
                )
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                        'question_type': sample['question_type'],
                                        'answers': sample['answers'],
                                        'entity_answers': [[]],
                                        'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_verified_answer(self, sample, start_prob, end_prob, verif_score, padded_p_len):
        """
        Finds the best answer with verification for a sample given start_prob, end_prob and
         verif_score for each sample.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            score *= verif_score[p_idx]
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer
