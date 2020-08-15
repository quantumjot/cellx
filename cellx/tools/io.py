import os
import json
import hashlib
import numpy as np
from tqdm import tqdm


def _hash_encoding(x: np.array):
    """ create a hash of the encoding """
    # return hash(x.tostring())
    m = hashlib.sha256()
    m.update(x.tostring())
    return m.hexdigest()



class EncodingWriter:
    """ EncodingWriter

    Handler class for writing encoded data
    """
    def __init__(self,
                 path: str):

        if not os.path.exists(path):
            os.makedirs(path)

        self._path = path

        # make some space for the json encoding
        self._json_data = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        json_fn = os.path.join(self._path, 'encodings.json')
        with open(json_fn, 'a') as file:
            json.dump(file, self._json_data)

    def write(self,
              encoding: np.ndarray,
              src_file: str,
              dst_file: str,
              class_label: int = 0,
              metadata: dict = {}):
        """ write out the encoding """

        assert dst_file.endswith('.npz')

        # save the npz file
        np.savez(dst_file, encoding=encoding, class_label=class_label)

        # save out the data to the json dictionary
        data = {'dst_file': dst_file,
                'src_file': src_file,
                'class_label': class_label,
                'hash': _hash_encoding(encoding.tostring())}
        self._json_data[dst_file] = {**data, **metadata}



class EncodingReader:
    """ EncodingReader

    Handler class for reading encoded data

    """
    def __init__(self,
                 filename: str):

        # grab the data from the file
        with open(filename, 'r') as file:
            data = json.load(file)

        # recover the entries in the dictionary
        self._metadata = list(data.values())

        # iterator position
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._idx+=1
        return self[self._idx-1]

    def __len__(self):
        """ return the number of entries """
        return len(self._data.keys())

    def __getitem__(self, idx):
        metadata = self._metadata[idx]

        # load the encoding
        # sanity check that the class label is correct and that the hash matches
        encoded = np.load(metadata['dst_file'])
        endoding = encoded['encoding']
        assert encoded['class_label'] == metadata['class_label']
        assert _hash_encoding(encoding) == metadata['hash']
        return encoding, metadata


    @property
    def statistics(self):
        pass
