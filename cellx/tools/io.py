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

    Params:
        filename: str, a path and filename for the json file storing the
                  metadata

    Usage:

        with EncodingWriter('/path/to/encodings.json') as writer:

            # put your code here to generate the encoding
            src_file = 'GV0800/Pos12/data.tif'
            encoding = some_function_to_generate_encoding(src_file)

            # save the encoding as a destination file:
            dst_file = 'GV0800/Pos12/data_encoded.npz'

            # store metadata, e.g. model parameters used for encoding
            metadata = {'model': 'my_cool_model.h5',
                        'version': '0027'}

            writer.write(encoding,
                         src_file,
                         dst_file,
                         class_label=0,
                         metadata=metadata)

    """
    def __init__(self,
                 filename: str):

        assert filename.endswith('.json')

        # check that the path to the filename exists
        path, _ = os.path.split(filename)
        if not os.path.exists(path):
            os.makedirs(path)

        self._filename = filename

        # make some space for the json encoding
        self._json_data = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open(self._filename, 'w') as file:
            json.dump(self._json_data, file,  separators=(',', ':'), indent=2)

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
                'hash': _hash_encoding(encoding)}
        self._json_data[dst_file] = {**data, **metadata}



class EncodingReader:
    """ EncodingReader

    Handler class for reading encoded data


    Params:
        filename: str, a path to the json file created by the EncodingWriter

    Usage:
        encodings = EncodingReader('/path/to/encodings.json')
        for encoded, metadata in encodings:
            print(encoding, metadata)

    Notes:
        this compares the hash of the loaded encoding with that stored in the
        metadata from the json. this is to ensure that the metadata and the
        encoding match.

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
        if self._idx >= len(self):
            raise StopIteration
        self._idx+=1
        return self[self._idx-1]

    def __len__(self):
        """ return the number of entries """
        return len(self._metadata)

    def __getitem__(self, idx):
        metadata = self._metadata[idx]

        # load the encoding
        encoded = np.load(metadata['dst_file'])
        encoding = encoded['encoding']

        # sanity check that the class label is correct and that the hash matches
        assert encoded['class_label'] == metadata['class_label']
        assert _hash_encoding(encoding) == metadata['hash']
        return encoding, metadata
