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

import argparse
import os

from model import CharToChar
from reader import FilePipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parser = argparse.ArgumentParser()

    # args for FilePipeline
    parser.add_argument('--file_name',
                        type=str,
                        required=True,
                        help='The Path of the data file in any readable format')

    parser.add_argument('--encoding',
                        type=str,
                        default='utf-8',
                        help='the encoding of the data file.')

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='number of epochs')

    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='minibatch size')

    # args for Char2Char

    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='NAme of this model')

    parser.add_argument('--units', type=int,
                        default=128,
                        help='size of RNN hidden state vector')

    parser.add_argument('--num_layers',
                        type=int,
                        default=3,
                        help='number of layers in the RNN')

    parser.add_argument('--cell_type', type=str, default='lstm',
                        help='which model to use (rnn, lstm or gru).')

    parser.add_argument('--input_dropout_keep_prob',
                        type=float,
                        default=1.0,
                        help=('dropout rate on input layer, default to 1 (no dropout),'
                              'and no dropout if using one-hot representation.'))

    parser.add_argument('--output_dropout_keep_prob',
                        type=float,
                        default=1.0,
                        help=('dropout-keep_prob rate on input layer, default to 1 (no dropout),'
                              'and no dropout if using one-hot representation.'))

    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-2,
                        help='initial learning rate')

    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='adam or rms optimizer')

    # parser.add_argument('--use_grad_clip',
    #                     type=bool
    #                     default=False,
    #                     help='Use grad clipping')

    args = parser.parse_args()
    pipe = build_pipe(args)
    model = build_model(pipe, args)

    model.train(pipe)
    model.dump_model_checkpoints(version=1)
    model.to_json()
    model.recycle()

    print('Training Completed. To Generate Outputs Run the sample.py file')
    exit(0)


def build_pipe(args):
    arg_d = vars(args)
    pipe = FilePipeline(file_name=arg_d['file_name'],
                        encoding=arg_d['encoding'],
                        batch_size=arg_d['batch_size'],
                        epoch=arg_d['epochs'])
    return pipe


def build_model(pipe, args):
    arg_d = vars(args)
    model = CharToChar(name=arg_d['name'],
                       units=arg_d['units'],
                       train_batch=arg_d['batch_size'],
                       hot_dimen=pipe.get_distinct_char_count(),
                       num_layers=arg_d['num_layers'],
                       cell_type=arg_d['cell_type'],
                       in_keep_prob_val=arg_d['input_dropout_keep_prob'],
                       out_keep_prob_val=arg_d['output_dropout_keep_prob'],
                       learning_rate=arg_d['learning_rate'],
                       optimizer=arg_d['optimizer'],
                       use_grad_clip=False,
                       grad_clip_val=5.0)
    return model


if __name__ == "__main__":
    main()
