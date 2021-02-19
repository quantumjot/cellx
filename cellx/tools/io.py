import enum
import hashlib
import json
import os
import zipfile

import numpy as np
from skimage import io


def _hash_encoding(x: np.array):
    """Create a unique hash of the encoding."""
    # return hash(x.tostring())
    m = hashlib.sha256()
    m.update(x.tostring())
    return m.hexdigest()


class EncodingWriter:
    """EncodingWriter.

    Handler class for writing encoded data.

    Parameters
    ----------
    filename : str
        A path and filename for the json file storing the metadata.

    Usage
    -----

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

    def __init__(self, filename: str):

        assert filename.endswith(".json")

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
        with open(self._filename, "w") as file:
            json.dump(self._json_data, file, separators=(",", ":"), indent=2)

    def write(
        self,
        encoding: np.ndarray,
        src_file: str,
        dst_file: str,
        class_label: int = 0,
        metadata: dict = {},
    ):
        """Write out the encoding."""

        assert dst_file.endswith(".npz")

        # save the npz file
        np.savez(dst_file, encoding=encoding, class_label=class_label)

        # save out the data to the json dictionary
        data = {
            "dst_file": dst_file,
            "src_file": src_file,
            "class_label": class_label,
            "hash": _hash_encoding(encoding),
        }
        self._json_data[dst_file] = {**data, **metadata}


class EncodingReader:
    """EncodingReader.

    Handler class for reading encoded data. This compares the hash of the loaded
    encoding with that stored in the metadata from the json. this is to ensure
    that the metadata and the encoding match.

    Parameters
    ----------
    filename : str
        A path and filename for the json file storing the metadata.

    Usage
    -----
    encodings = EncodingReader('/path/to/encodings.json')
    for encoded, metadata in encodings:
        print(encoding, metadata)

    """

    def __init__(self, filename: str):
        # grab the data from the file
        with open(filename, "r") as file:
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
        self._idx += 1
        return self[self._idx - 1]

    def __len__(self):
        """Return the number of entries."""
        return len(self._metadata)

    def __getitem__(self, idx: int):
        """Get and encoding and metadata."""
        metadata = self._metadata[idx]

        # load the encoding
        encoded = np.load(metadata["dst_file"])
        encoding = encoded["encoding"]

        # sanity check that the class label is correct and that the hash match
        assert encoded["class_label"] == metadata["class_label"]
        assert _hash_encoding(encoding) == metadata["hash"]
        return encoding, metadata

    def load_image(self, idx: int, scale: int = 1, use_cutoff: bool = True):
        """Get the associated image data."""
        metadata = self._metadata[idx]

        # load the raw image data
        image_fn = metadata["src_file"]
        if not image_fn.endswith(".tif"):
            image_fn = image_fn + ".tif"

        image = io.imread(image_fn)

        # get the cutoff
        if use_cutoff:
            cutoff = metadata["cutoff"]
        else:
            cutoff = image.shape[0]

        assert cutoff >= 0 and cutoff <= image.shape[0]
        assert scale >= 0 and scale < image.shape[1]

        # crop the image stack, and make channels last dim
        stack = image[:cutoff, scale, ...]
        stack = np.rollaxis(stack, 1, 4)

        return stack, metadata


def read_annotations(path: str):
    """Read annotations.

    This provides a capability to load the contents of a single, or multiple
    annotation files, aggregating their contents and returning a dictionary of
    state labels.

    Parameters
    ----------
    path : str
        The path to the folder containing the annotation.zip files

    Returns
    -------
    images : np.ndarray
        An array of N image patches of the format N,(Z),X,Y,C
    labels : np.ndarray
        An array of numeric image labels of size (N, )
    states : dict
        A dictionary mapping a string label to the numeric image label
    """

    images = []
    labels = []
    states = {}

    # find the zip files:
    zipfiles = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".zip") and f.startswith("annotation_")
    ]

    if len(zipfiles) == 0:
        raise IOError("Warning, no 'annotation' zip files found.")

    # iterate over the zip files and aggregate the data
    for zfn in zipfiles:

        with zipfile.ZipFile(zfn, "r") as zip_data:
            files = zip_data.namelist()

            # first open the annotation file
            json_fn = [f for f in files if f.endswith(".json")][0]
            with zip_data.open(json_fn) as js:
                json_data = json.load(js)
                _states = json_data["states"]

                # if we have no states defined, use the serialized states to
                # define them
                if not states:
                    states = _states
                    States = enum.Enum("States", states)

                # if they do exist, make sure that they match all other files
                if states != _states:
                    raise Exception("Annotation files are incompatible")

            for state in States:

                numeric_label = state.value
                label = state.name

                patch_files = [
                    f for f in files if f.endswith(".tif") and f.startswith(label)
                ]

                raw_images = [io.imread(zip_data.open(f)) for f in patch_files]
                # images_resized = [resize(img, (64, 64), preserve_range=True) for img in images]

                images += raw_images
                labels += [numeric_label] * len(raw_images)

    images_arr = np.stack(images, axis=0)[..., np.newaxis]
    labels_arr = np.stack(labels, axis=0)

    # sanity check that we have the same number of labels and images
    assert images_arr.shape[0] == labels_arr.shape[0]

    return images_arr, labels_arr, states
