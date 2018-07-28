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

    parser.add_argument('--output_json',
                        type=str,
                        required=True,
                        help='json output file produced by train.py')

    parser.add_argument('--seq_len',
                        type=int,
                        default=100,
                        help='No of output characters to generate')

    parser.add_argument('--source_file',
                        type=str,
                        required=True,
                        help=('The data file on which the this json was trained on.'))

    args = parser.parse_args()
    arg_d = vars(args)
    model = CharToChar.from_json('.', arg_d['output_json'])
    model.start_session()
    model.load_saved_checkpoints(version=1)
    pipe = FilePipeline(file_name=arg_d['source_file'])
    seq_len = arg_d['seq_len']
    out_file = model.name + '-output.txt'

    model.sample(pipe, seq_len, save_as_file=out_file)
    model.recycle()


if __name__ == "__main__":
    main()
