"""
File to run some unit test on the API
"""
import os

from werkzeug.datastructures import FileStorage

from audioclas import paths
from audioclas.api import predict_data, predict_url


def test_predict_url():
    url = 'https://file-examples.com/wp-content/uploads/2017/11/file_example_WAV_1MG.wav'
    args = {'urls': [url], 'filter': []}
    results = predict_url(args)


def test_predict_data():
    # fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'applause.wav')  # uncompressed WAV
    # fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'cat-mad2_compressed.wav')  # compressed WAV
    fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'music_sample.mp3')  # MP3 file

    file = FileStorage(open(fpath, 'rb'))
    args = {'files': [file], 'filter': []}
    results = predict_data(args)
    print(results)


if __name__ == '__main__':
    # test_predict_data()
    # test_predict_url()
