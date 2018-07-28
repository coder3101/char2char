"""
    Copyright 2018 Ashar <ashar786khan@gmail.com>
 
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import json
import os.path

import numpy as np
import tensorflow as tf
from tensorflow.nn.rnn_cell import (BasicLSTMCell, BasicRNNCell,
                                    DropoutWrapper, GRUCell, MultiRNNCell)
from tqdm import tqdm

from reader import FilePipeline
from utilits import preditions_to_string, rnn_placeholders


class CharToChar():
    def __init__(self,
                 name,
                 units,
                 train_batch,
                 hot_dimen,
                 num_layers=3,
                 cell_type='lstm',
                 in_keep_prob_val=0.7,
                 out_keep_prob_val=0.7,
                 learning_rate=1e-2,
                 optimizer='adam',
                 use_grad_clip=False,
                 grad_clip_val=5.0):

        self.name = name
        self.units = units
        self.train_batch = train_batch
        self.num_layers = num_layers
        self.in_keep_prob_val = in_keep_prob_val
        self.out_keep_prob_val = out_keep_prob_val
        self.cell_type = cell_type
        self.hot_dimen = hot_dimen
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.grad_clip_val = grad_clip_val
        self.use_grad_clip = use_grad_clip

        self.multi_cell = None
        self.initial_state = None
        self.input_placeholder = None
        self.output_placeholder = None
        self.outputs_raw = None
        self.logits = None
        self.predictions = None
        self.entropy_loss = None
        self.grad_update = None
        self.final_state = None
        self.sess = None
        self.init = None
        self.zero_state = None
        self.in_keep_prob = None
        self.out_keep_prob = None

        self.is_training_done = False

        tf.reset_default_graph()
        self._build_placeholders()
        self._apply_droput_wrapper()
        self._static_unroll()
        self._reshape_and_unstack()
        self._build_optimizer_and_finalize_graph()

    def __new_cell(self):
        if self.cell_type == 'lstm':
            return BasicLSTMCell(self.units)
        elif self.cell_type == 'rnn':
            return BasicRNNCell(self.units)
        else:
            return GRUCell(self.units)

    def _apply_droput_wrapper(self):
        cells = []
        for _ in range(self.num_layers):
            cell = self.__new_cell()
            cell = DropoutWrapper(
                cell, input_keep_prob=self.in_keep_prob, output_keep_prob=self.out_keep_prob)
            cells.append(cell)
        self.multi_cell = MultiRNNCell(cells)

        self.initial_state = rnn_placeholders(
            self.multi_cell.zero_state(self.batch_size, tf.float32))

        self.zero_state = self.multi_cell.zero_state(
            self.batch_size, tf.float32)

    def _build_placeholders(self):
        self.batch_size = tf.placeholder(tf.int32, [])

        self.input_placeholder = tf.placeholder(
            tf.float32, shape=[None, self.hot_dimen])

        self.output_placeholder = tf.placeholder(
            tf.float32, shape=[None, self.hot_dimen])

        self.in_keep_prob = tf.placeholder(tf.float32, [])

        self.out_keep_prob = tf.placeholder(tf.float32, [])

    def _static_unroll(self):
        self.outputs_raw, self.final_state = tf.nn.static_rnn(
            cell=self.multi_cell,
            inputs=[self.input_placeholder],
            dtype=tf.float32,
            initial_state=self.initial_state
        )
        self.outputs_raw = self.outputs_raw[0]

    def get_shared_variable(self, var, shape=None):
        with tf.variable_scope('softmax_dense', reuse=tf.AUTO_REUSE):
            v = tf.get_variable(var, shape)
        return v

    def __apply_dense(self, time_step):
        w = self.get_shared_variable('W', shape=[self.units, self.hot_dimen])
        b = self.get_shared_variable('B', shape=[self.hot_dimen])
        return tf.matmul(time_step, w) + b

    def _reshape_and_unstack(self):
        self.logits = self.__apply_dense(self.outputs_raw)
        predictions = tf.nn.softmax(self.logits)
        self.predictions = predictions
        self.entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.output_placeholder, logits=self.logits))

    def _build_optimizer_and_finalize_graph(self):
        opt = None
        if self.optimizer.lower() == 'adam':
            opt = tf.train.AdamOptimizer
        elif self.optimizer.lower() == 'rms':
            opt = tf.train.RMSPropOptimizer
        else:
            opt = tf.train.AdamOptimizer
        if self.use_grad_clip:
            grad_vars = opt.compute_gradients(self.entropy_loss)
            grad_clip_const = tf.constant(
                self.grad_clip_val, name='grad_clipper')
            clipped_grad_var = [(tf.clip_by_value(
                grad, -grad_clip_const, grad_clip_const), var) for grad, var in grad_vars]
            self.grad_update = opt.apply_gradients(clipped_grad_var)
        else:
            self.grad_update = opt(self.learning_rate).minimize(
                loss=self.entropy_loss)

    def start_session(self):
        self.sess = tf.Session()
        return self.sess

    def train(self,
              file_pipe,
              session=None,
              print_loss_after_iterations=50):

        self.sess = session or tf.Session()
        self.init = tf.global_variables_initializer()
        if isinstance(file_pipe, FilePipeline):
            self.is_training_done = True
            assert self.hot_dimen == file_pipe.get_distinct_char_count()
            self.sess.run(self.init)
            state = self.sess.run(self.zero_state, feed_dict={
                                  self.batch_size: self.train_batch,
                                  self.in_keep_prob: self.in_keep_prob_val,
                                  self.out_keep_prob: self.out_keep_prob_val})
            all_epoch_done = False
            i = 0
            p_bar = tqdm(total=file_pipe.get_expected_total_iteration())
            p_bar.update(i)
            p_bar.set_description("Iteration")
            while not all_epoch_done:
                data, all_epoch_done = file_pipe.next_batch()
                feeder = {self.input_placeholder: data[0],
                          self.output_placeholder: data[1],
                          self.initial_state: state,
                          self.batch_size: self.train_batch,
                          self.in_keep_prob: self.in_keep_prob_val,
                          self.out_keep_prob: self.out_keep_prob_val}
                # if i % print_loss_after_iterations == 0:
                #     state, loss, _ = self.sess.run(
                #         [self.final_state, self.entropy_loss, self.grad_update], feed_dict=feeder)
                #     print('At Iteration {} = {}'.format(i, loss))
                # else:
                state, _ = self.sess.run(
                    [self.final_state, self.grad_update], feed_dict=feeder)
                i += 1
                p_bar.update(1)
        else:
            raise ValueError(
                "Cannot train the model. file_pipe is not an instance of FilePipeline")

    def recycle(self):
        self.sess.close()

    def sample(self, f_pipe, seq_len=5, save_as_file=None):
        if not self.is_training_done:
            raise ValueError(
                "You must train the model before sampling sequences.")
        else:
            state = self.sess.run(self.initial_state,
                                  feed_dict={
                                      self.batch_size: 1,
                                      self.in_keep_prob: 1.0,
                                      self.out_keep_prob: 1.0})
            inp = np.zeros((1, self.hot_dimen))
            result = []
            for _ in range(seq_len):
                feeder = {self.input_placeholder: inp,
                          self.initial_state: state,
                          self.batch_size: 1,
                          self.in_keep_prob: 1.0,
                          self.out_keep_prob: 1.0}
                inp, state = self.sess.run(
                    [self.predictions, self.final_state], feed_dict=feeder)
                # pylint:disable=E1101
                x = np.random.choice(self.hot_dimen, p=np.squeeze(
                    inp))
                inp = np.zeros((1, self.hot_dimen))
                inp[0, x] = 1
                result.append(x)
            preditions_to_string(f_pipe, result, save_as_file)

    def load_saved_checkpoints(self, version, folder=None):
        saver = tf.train.Saver()
        self.is_training_done = True
        if folder is None:
            saver.restore(self.sess, './saved-v'+str(version)+'/'+self.name)
        else:
            saver.restore(self.sess, './' + folder +
                          str(version)+'/'+self.name)

    def dump_model_checkpoints(self, version, folder=None):
        saver = tf.train.Saver()
        if folder is None:
            saver.save(self.sess, './saved-v' + str(version)+'/'+self.name)
        else:
            saver.save(self.sess, './' + folder + str(version) + '/'+self.name)

    def to_json(self, path='.', file_name=None):
        f_name = file_name or self.name
        config = ["name",
                  "units",
                  "train_batch",
                  "hot_dimen",
                  "num_layers",
                  "cell_type",
                  "in_keep_prob_val",
                  "out_keep_prob_val",
                  "learning_rate",
                  "optimizer",
                  "use_grad_clip",
                  "grad_clip_val"]
        data = {k: v for k, v in self.__dict__.items() if k in config}
        with open(os.path.join(path, f_name)+'.json', mode='w') as f:
            json.dump(data, f)

    @staticmethod
    def from_json(path, file_name):
        data = None
        with open(os.path.join(path, file_name)) as f:
            data = json.load(f)
        return CharToChar(**data)
