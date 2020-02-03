"""
Miscellaneous functions manage data.

Date: September 2019
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import os
import io
import subprocess
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from tensorflow.keras.utils import to_categorical, Sequence

from audioclas import paths


def load_data_splits(splits_dir, dataset_dir, split_name='train'):
    """
    Load the data arrays from the [train/val/test].txt files.
    Lines of txt files have the following format:
    'relative_path_to_image' 'image_label_number'

    Parameters
    ----------
    dataset_dir : str
        Absolute path to the image folder.
    split_name : str
        Name of the data split to load

    Returns
    -------
    X : Numpy array of strs
        First colunm: Contains 'absolute_path_to_file' to images.
    y : Numpy array of int32
        Image label number
    """
    if '{}.txt'.format(split_name) not in os.listdir(splits_dir):
        raise ValueError("Invalid value for the split_name parameter: there is no `{}.txt` file in the `{}` "
                         "directory.".format(split_name, splits_dir))

    # Loading splits
    print("Loading {} data...".format(split_name))
    split = np.genfromtxt(os.path.join(splits_dir, '{}.txt'.format(split_name)), dtype='str', delimiter=' ')
    X = np.array([os.path.join(dataset_dir, i) for i in split[:, 0]])

    if len(split.shape) == 2:
        y = split[:, 1].astype(np.int32)
    else: # maybe test file has not labels
        y = None

    return X, y


def mount_nextcloud(frompath, topath):
    """
    Mount a NextCloud folder in your local machine or viceversa.
    """
    command = (['rclone', 'copy', frompath, topath])
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn("Error while mounting NextCloud: {}".format(error))
    return output, error


def load_class_names(splits_dir):
    """
    Load list of class names

    Returns
    -------
    Numpy array of shape (N) containing strs with class names
    """
    print("Loading class names...")
    class_names = np.genfromtxt(os.path.join(splits_dir, 'classes.txt'), dtype='str', delimiter='/n')
    return class_names


def load_class_info(splits_dir):
    """
    Load list of class names

    Returns
    -------
    Numpy array of shape (N) containing strs with class names
    """
    print("Loading class info...")
    class_info = np.genfromtxt(os.path.join(splits_dir, 'info.txt'), dtype='str', delimiter='/n')
    return class_info


class data_sequence(Sequence):
    """
    Instance of a Keras Sequence that is safer to use with multiprocessing than a standard generator.
    Check https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    TODO: Add sample weights on request
    """

    def __init__(self, inputs, targets, batch_size, num_classes, shuffle=True):
        """
        Parameters are the same as in the data_generator function
        """
        assert len(inputs) == len(targets)
        assert len(inputs) >= batch_size

        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.inputs) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idxs = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_X = []
        for i in batch_idxs:
            sample = np.load(self.inputs[i])
            if len((sample.shape)) == 2:
                sample = np.expand_dims(sample, axis=0)
            batch_X.append(sample)
        batch_X = np.vstack(batch_X)
        batch_y = to_categorical(self.targets[batch_idxs], num_classes=self.num_classes)
        return batch_X, batch_y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.inputs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def generate_embeddings(model_wrap, file_list):
    new_paths = []
    embed_dir, dataset_dir = paths.get_embeddings_dir(), paths.get_dataset_dir()
    for wav_file in tqdm(file_list):
        with open(wav_file, 'rb') as f:
            tmp_path = os.path.relpath(wav_file, start=dataset_dir)
            tmp_path = tmp_path.split('.')[0] + '.npy'
            tmp_path = os.path.join(embed_dir, tmp_path)

            raw_embeddings = model_wrap.generate_embeddings(f.read())
            embeddings_processed = model_wrap.classifier_pre_process(raw_embeddings)

            path = Path(os.path.dirname(tmp_path))
            path.mkdir(parents=True, exist_ok=True)
            np.save(tmp_path, embeddings_processed)

            new_paths.append(tmp_path)

    if len(set(new_paths)) < len(file_list):
        raise Exception('WAV files should have different names.')

    return np.array(new_paths)


def create_embeddings_names(X):
    """
    Small utility to transform existing txt files with audio names to txt files with embeddings names.
    """
    new_X = np.array([f.split('.')[0] + '.npy' for f in X])
    if not os.path.isfile(new_X[0]):
        raise Exception('Failed to generate the embeddings paths. '
                        'Please check the dataset_directory points to the embeddings files and you can optionally '
                        'provide the embeddings names instead of the original audio files names.')
    return new_X


def json_friendly(d):
    """
    Return a json friendly dictionary (mainly remove numpy data types)
    """
    new_d = {}
    for k, v in d.items():
        if isinstance(v, (np.float32, np.float64)):
            v = float(v)
        elif isinstance(v, (np.ndarray, list)):
            if isinstance(v[0], (np.float32, np.float64)):
                v = np.array(v).astype(float).tolist()
            else:
                v = np.array(v).tolist()
        new_d[k] = v
    return new_d


def file_to_PCM_16bits(read_path, save_path=None):
    """
    Transform audio file to a format readable by scipy, ie. uncompressed PCM 16-bits.
    Support transformation from any format supported by ffmpeg.
    """
    try:
        audiofile = AudioSegment.from_file(read_path)  # it infers the file format
        # file_format = read_path.split('.')[-1]
        # audiofile = AudioSegment.from_file(read_path, file_format)
    except Exception as e:
        raise Exception("""Invalid audio file. Make sure you have FFMPEG installed.""")

    # Apply desired preprocessing
    audiofile.set_sample_width = 2  # set to 16-bits
    audiofile = audiofile[:10*1000]  # keep only the first ten seconds
    # duration = audiofile.duration_seconds
    # audiofile.strip_silence

    save_path = read_path if not save_path else save_path
    audiofile.export(save_path, format="wav")


def bytes_to_PCM_16bits(bytes):
    """
    Transform audio file to a format readable by scipy, ie. uncompressed PCM 16-bits.
    Support transformation from any format supported by ffmpeg.
    """
    try:
        audiofile = AudioSegment.from_file(bytes)
    except Exception as e:
        raise Exception("""Invalid audio file.""")

    # Apply desired preprocessing
    audiofile.set_sample_width = 2  # set to 16-bits
    audiofile = audiofile[:10*1000]  # keep only the first ten seconds
    # duration = audiofile.duration_seconds
    # audiofile.strip_silence

    # Return the results as bytes without writing to disk
    # ref: https://github.com/jiaaro/pydub/issues/270
    buf = io.BytesIO()
    audiofile.export(buf, format="wav")
    return buf.getvalue()


# def transform_to_16_bits(read_path, save_path=None):
#     """
#     Transform WAV file to a format readable by scipy.
#     This old function is dependent on the wavio module, which doesn't need ffmpeg.
#     The downside is that it doesn't support compression.
#     """
#     try:
#         w = wavio.read(read_path)
#     except Exception as e:
#         raise Exception("""Invalid WAV format. Remember Python does not support compressed WAV files.
#                            Try Audacity to decompress your files in bulk.""")
#
#     save_path = read_path if not save_path else save_path
#     wavio.write(data=w.data,
#                 file=save_path,
#                 rate=w.rate,
#                 sampwidth=2)  # samp=2 for 16-bit
