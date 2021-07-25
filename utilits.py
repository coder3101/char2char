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

import os

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.compat.v1.nn.rnn_cell import LSTMStateTuple


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def rnn_placeholders(state):
    """Convert RNN state tensors to placeholders with the zero state as default."""
    if isinstance(state, LSTMStateTuple):
        c, h = state
        c = tf.placeholder_with_default(c, c.shape, c.op.name)
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)


def preditions_to_string(pipeline, preds, write_as_file=None):
    string = ''
    i_to_char = pipeline._get_index_list()
    chrs = [i_to_char[p] for p in preds]
    if write_as_file is None:
        print(string.join(chrs))
    else:
        res = string.join(chrs)
        with open('./'+str(write_as_file), 'w') as f:
            f.write(res)
        print('Saved the output on a file :', write_as_file)
