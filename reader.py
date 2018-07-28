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

import numpy as np


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


class FilePipeline():
    def __init__(self, file_name, encoding='utf-8', batch_size=100, epoch=10):
        self.file = file_name
        self.encoding = encoding
        self._data = None
        self.read_pointer = 0
        self.cur_epoch = 0
        self.tot_epoch = epoch
        self.batch = batch_size

        # Private calls
        self._read()
        self._index_list = self._get_index_list()
        self._char_dict = self._get_vocab_dict()
        self._unique_chars = self.get_distinct_chars()

        # Useful information for user
        print('Reading the file :', file_name)
        print('Encoding : ', encoding)
        print('Unique Characters : ', len(self._unique_chars))
        print('Total Characters :', len(list(self._data)))
        print('Epoch :', epoch)
        print('Batch Size : ', batch_size)
        print('Total Expected iteration : ',
              self.get_expected_total_iteration())

    def _read(self):
        res = None
        with open(self.file, mode='r', encoding=self.encoding) as f:
            res = f.read()
        self._data = res
        return res

    def get_distinct_chars(self):
        chars = set(list(self._data))
        return chars

    def _get_vocab_dict(self, vocabs=None):
        return {char: index for index, char in enumerate(sorted(self.get_distinct_chars()))}

    def _get_index_list(self):
        return [char for char in sorted(self.get_distinct_chars())]

    def _batch_to_one_hot(self, batch):
        index_rep = [self._get_vocab_dict()[char] for char in batch]
        return get_one_hot(np.array(index_rep), len(self._unique_chars))

    def next_data(self):
        data = self._data
        start = self.read_pointer
        stop = start + 1
        if stop+1 > len(data):
            self.read_pointer = 0
            self.cur_epoch += 1
            return (np.squeeze(self._batch_to_one_hot(data[0:1])), np.squeeze(self._batch_to_one_hot(data[1:2])), False)
        else:
            self.read_pointer = stop
            if self.cur_epoch > self.tot_epoch:
                return (self._batch_to_one_hot(data[start:stop]), self._batch_to_one_hot(data[start+1:stop+1]), True)
            return (np.squeeze(self._batch_to_one_hot(data[start:stop])), np.squeeze(self._batch_to_one_hot(data[start+1:stop+1])), False)

    def next_batch(self):
        data = (np.zeros((self.batch, self.get_distinct_char_count())),
                np.zeros((self.batch, self.get_distinct_char_count())))
        for _ in range(self.batch):
            x, y, end = self.next_data()
            data[0][_, :] = x
            data[1][_, :] = y
        return (data, end)

    def get_distinct_char_count(self):
        return len(self.get_distinct_chars())

    def get_expected_total_iteration(self):
        data_len = len(list(self._data))-1
        iteration = (self.tot_epoch*data_len + data_len)//self.batch
        return iteration+1
