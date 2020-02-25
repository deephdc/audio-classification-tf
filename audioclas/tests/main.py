"""
File to run some unit test on the API
"""
import glob
import json
import os
import shutil
import subprocess
import time
from urllib.parse import quote_plus

from audioclas import paths


module_name = 'audioclas'
test_url = 'https://file-examples.com/wp-content/uploads/2017/11/file_example_WAV_1MG.wav'

data_path = os.path.join(paths.get_base_dir(), 'data')



def copy_files(src, dst, extension):
    files = glob.iglob(os.path.join(src, extension))
    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, dst)


def remove_files(src, extension):
    files = glob.iglob(os.path.join(src, extension))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


# ===========
# Local Tests
# ===========

def test_load():
    print('Testing local: module load ...')
    import audioclas.api


def test_metadata():
    print('Testing local: metadata ...')
    from audioclas.api import get_metadata

    get_metadata()


def test_predict_url():
    print('Testing local: predict url ...')
    from audioclas.api import predict_url

    args = {'urls': [test_url]}
    r = predict_url(args)


def test_predict_data():
    print('Testing local: predict data ...')
    from deepaas.model.v2.wrapper import UploadedFile
    from audioclas.api import predict_data

    fpath = os.path.join(data_path, 'samples', 'applause.wav')  # uncompressed WAV
    # fpath = os.path.join(data_path, 'samples', 'cat-mad2_compressed.wav')  # compressed WAV
    # fpath = os.path.join(data_path, 'samples', 'music_sample.mp3')  # MP3 file
    content_type = 'audio'

    # fpath = os.path.join(data_path, 'samples', 'demo.tar.xz')  # multiple audios
    # content_type='application/x-xz'

    tmp_fpath = os.path.join(data_path, 'samples', 'tmp_file.jpg')
    shutil.copyfile(fpath, tmp_fpath)  # copy to tmp because we are deleting the file after prediction
    file = UploadedFile(name='data', filename=tmp_fpath, content_type=content_type)
    args = {'files': [file]}
    r = predict_data(args)


def test_train():
    print('Testing local: train ...')

    from audioclas.api import get_train_args, train

    copy_files(src=os.path.join(data_path, 'demo-dataset_files', 'demo'),
               dst=os.path.join(data_path, 'dataset_files'),
               extension='*.txt')

    args = get_train_args()
    args_d = {k: v.missing for k, v in args.items()}
    args_d['dataset_directory'] = '"data/samples"'
    out = train(**args_d)

    remove_files(src=os.path.join(data_path, 'dataset_files'),
                 extension='*.txt')

    shutil.rmtree(os.path.join(paths.get_models_dir(), out['modelname']), ignore_errors=True)


# ==========
# CURL Tests
# ==========

def test_curl_load():
    print('Testing curl: module load ...')

    r = subprocess.run('curl -X GET "http://0.0.0.0:5000/v2/models/" -H "accept: application/json"',
                       shell=True, check=True, stdout=subprocess.PIPE).stdout
    r = json.loads(r)
    models = [m['name'] for m in r['models']]
    if module_name not in models:
        raise Exception('Model is not correctly loaded.')


def test_curl_metadata():
    print('Testing curl: metadata ...')

    r = subprocess.run('curl -X GET "http://0.0.0.0:5000/v2/models/{}/" -H "accept: application/json"'.format(module_name),
                       shell=True, check=True, stdout=subprocess.PIPE).stdout
    if r == b'404: Not Found':
        raise Exception('Model is not correctly loaded.')
    r = json.loads(r)


def test_curl_predict_url():
    print('Testing curl: predict url ...')

    r = subprocess.run('curl -X POST "http://0.0.0.0:5000/v2/models/{}/predict/?urls={}" -H "accept: application/json"'.format(module_name,
                                                                                                                               quote_plus(test_url)),
                       shell=True, check=True, stdout=subprocess.PIPE).stdout
    if r == b'404: Not Found':
        raise Exception('Model is not correctly loaded.')
    r = json.loads(r)


def test_curl_train():
    print('Testing curl: train ...')

    copy_files(src=os.path.join(data_path, 'demo-dataset_files'),
               dst=os.path.join(data_path, 'dataset_files'),
               extension='*.txt')

    command = """curl -X POST "http://0.0.0.0:5000/v2/models/audioclas/train/?base_directory=%22.%22&dataset_directory=%22data%2Fsamples%22&num_classes=null&files_to_PCM=true&compute_embeddings=true&mode=%22normal%22&initial_lr=0.001&batch_size=500&epochs=15&ckpt_freq=null&lr_schedule_mode=%22step%22&lr_step_decay=0.1&lr_step_schedule=%5B0.7%2C%200.9%5D&l2_reg=0.0001&use_validation=true&use_early_stopping=false&use_tensorboard=false&use_remote=false" -H "accept: application/json"
    """
    r = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE).stdout

    # """
    # We cannot remove the resulting files as DEEPaaS is asynchronous and therefore we cannot know we the training
    # finishes
    # """
    # remove_files(src=os.path.join(data_path, 'dataset_files'),
    #              extension='*.txt')
    # shutil.rmtree(os.path.join(paths.get_models_dir(), out['modelname']), ignore_errors=True)


if __name__ == '__main__':
    print('Testing locally ...')
    test_load()
    test_metadata()
    test_predict_url()
    test_predict_data()
    test_train()

    print('Testing through CURL ...')
    r = subprocess.run('deepaas-run --listen-ip 0.0.0.0 --nowarm &', shell=True)  # launch deepaas
    time.sleep(20)  # wait for deepaas to be ready
    test_curl_load()
    test_curl_metadata()
    test_curl_predict_url()
    # test_curl_train()
    r = subprocess.run("kill $(ps aux | grep 'deepaas-run' | awk '{print $2}')", shell=True)   # kill deepaas

