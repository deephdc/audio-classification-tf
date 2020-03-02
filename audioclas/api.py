"""
API for the image classification package

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

Notes: Based on https://github.com/indigo-dc/plant-classification-theano/blob/package/plant_classification/api.py

Descriptions:
The API will use the model files inside ../models/api. If not found it will use the model files of the last trained model.
If several checkpoints are found inside ../models/api/ckpts we will use the last checkpoint.

Warnings:
There is an issue of using Flask with Keras: https://github.com/jrosebr1/simple-keras-rest-api/issues/1
The fix done (using tf.get_default_graph()) will probably not be valid for standalone wsgi container e.g. gunicorn,
gevent, uwsgi.
"""

import builtins
from collections import OrderedDict
from datetime import datetime
import json
import magic
import mimetypes
import os
import pkg_resources
import random
import re
import string
from urllib.request import urlretrieve
import warnings

from deepaas.model.v2.wrapper import UploadedFile
import numpy as np
import requests
from tensorflow.keras import backend as K
from webargs import fields
from aiohttp.web import HTTPBadRequest

from audioclas import config, paths, misc
from audioclas.data_utils import load_class_names, load_class_info, bytes_to_PCM_16bits
from audioclas.model import ModelWrapper
from audioclas.train import train_fn


# # Mount NextCloud folders (if NextCloud is available)
# try:
#     mount_nextcloud('ncplants:/data/dataset_files', paths.get_splits_dir())
#     mount_nextcloud('ncplants:/data/images', paths.get_images_dir())
#     #mount_nextcloud('ncplants:/models', paths.get_models_dir())
# except Exception as e:
#     print(e)

# Empty model variables for inference (will be loaded the first time we perform inference)
loaded_ts, loaded_ckpt = None, None
model_wrapper, conf, class_names, class_info = None, None, None, None

# Additional parameters
compressed_extensions = ['zip', 'tar', 'bz2', 'tb2', 'tbz', 'tbz2', 'gz', 'tgz', 'lz', 'lzma', 'tlz', 'xz', 'txz', 'Z', 'tZ']


def load_inference_model(timestamp=None, ckpt_name=None):
    """
    Load a model for prediction.

    Parameters
    ----------
    * timestamp: str
        Name of the timestamp to use. The default is the last timestamp in `./models`.
    * ckpt_name: str
        Name of the checkpoint to use. The default is the last checkpoint in `./models/[timestamp]/ckpts`.
    """
    global loaded_ts, loaded_ckpt
    global model_wrapper, conf, class_names, class_info

    # Set the timestamp
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    timestamp_list.remove('common')  # common files do not count as full model
    if not timestamp_list:
        raise Exception(
            "You have no models in your `./models` folder to be used for inference. "
            "Therefore the API can only be used for training.")
    elif timestamp is None:
        timestamp = timestamp_list[-1]
    elif timestamp not in timestamp_list:
        raise ValueError(
            "Invalid timestamp name: {}. Available timestamp names are: {}".format(timestamp, timestamp_list))
    paths.timestamp = timestamp
    print('Using TIMESTAMP={}'.format(timestamp))

    # Set the checkpoint model to use to make the prediction
    ckpt_list = os.listdir(paths.get_checkpoints_dir())
    ckpt_list = sorted([name for name in ckpt_list if name.endswith('.h5')])
    if not ckpt_list:
        raise Exception(
            "You have no checkpoints in your `./models/{}/ckpts` folder to be used for inference. ".format(timestamp) +
            "Therefore the API can only be used for training.")
    elif ckpt_name is None:
        ckpt_name = ckpt_list[-1]
    elif ckpt_name not in ckpt_list:
        raise ValueError(
            "Invalid checkpoint name: {}. Available checkpoint names are: {}".format(ckpt_name, ckpt_list))
    print('Using CKPT_NAME={}'.format(ckpt_name))

    # Clear the previous loaded model
    K.clear_session()

    # Load the class names and info
    splits_dir = paths.get_ts_splits_dir()
    class_names = load_class_names(splits_dir=splits_dir)
    class_info = None
    if 'info.txt' in os.listdir(splits_dir):
        class_info = load_class_info(splits_dir=splits_dir)
        if len(class_info) != len(class_names):
            warnings.warn("""The 'classes.txt' file has a different length than the 'info.txt' file.
            If a class has no information whatsoever you should leave that classes row empty or put a '-' symbol.
            The API will run with no info until this is solved.""")
            class_info = None
    if class_info is None:
        class_info = ['' for _ in range(len(class_names))]

    # Load training configuration
    conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
    with open(conf_path) as f:
        conf = json.load(f)
        update_with_saved_conf(conf)

    # Load the model
    model_wrapper = ModelWrapper(classifier_model=os.path.join(paths.get_checkpoints_dir(), ckpt_name))

    # Set the model as loaded
    loaded_ts = timestamp
    loaded_ckpt = ckpt_name


def update_with_saved_conf(saved_conf):
    """
    Update the default YAML configuration with the configuration saved from training
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        if group in saved_conf.keys():
            for g_key, g_val in sorted(val.items()):
                if g_key in saved_conf[group].keys():
                    g_val['value'] = saved_conf[group][g_key]

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def update_with_query_conf(user_args):
    """
    Update the default YAML configuration with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                g_val['value'] = json.loads(user_args[g_key])

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def catch_url_error(url_list):

    # Error catch: Empty query
    if not url_list:
        raise ValueError('Empty query')

    for i in url_list:

        # Error catch: Inexistent url
        try:
            url_type = requests.head(i).headers.get('content-type')
        except Exception:
            raise ValueError("Failed url connection: "
                             "Check you wrote the url address correctly.")

        # # Error catch: Wrong formatted urls
        # if url_type.split('/')[0] != 'audio':
        #     raise BadRequest("""Input should be an audio file""")


def catch_localfile_error(file_list):
    """
    No need to check for file formats because we now support all the formats supported by FFMPEG
    """

    # Error catch: Empty query
    if not file_list:
        raise ValueError('Empty query')


def warm():
    load_inference_model()


@catch_error
def predict(**args):

    if (not any([args['urls'], args['files']]) or
            all([args['urls'], args['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if args['files']:
        args['files'] = [args['files']]  # patch until list is available
        return predict_data(args)
    elif args['urls']:
        args['urls'] = [args['urls']]  # patch until list is available
        return predict_url(args)


def predict_url(args):
    """
    Function to predict an url
    """
    catch_url_error(args['urls'])

    # Download files
    args['files'] = []
    for url in args['urls']:
        fname = ''.join(random.choices(string.ascii_lowercase + string.digits, k=15))
        fpath = os.path.join('/tmp', fname)
        urlretrieve(url, fpath)
        f = UploadedFile(name='data', filename=fpath, content_type=magic.from_file(fpath, mime=True))
        args['files'].append(f)

    return predict_data(args)


def predict_data(args):
    """
    Function to predict a local image
    """
    # Check user configuration
    update_with_query_conf(args)
    conf = config.conf_dict

    catch_localfile_error(args['files'])

    # Unpack if needed
    file_format = mimetypes.guess_extension(args['files'][0].content_type)
    if file_format and file_format[1:] in compressed_extensions:
        output_folder = os.path.join('/tmp', os.path.basename(args['files'][0].filename)).split('.')[0] + '_decomp'
        misc.open_compressed(byte_stream=open(args['files'][0].filename, 'rb'),
                             file_format=file_format[1:],
                             output_folder=output_folder)
        filenames = misc.find_audiofiles(folder_path=output_folder)
    else:
        filenames = [f.filename for f in args['files']]

    # Load model if needed
    if loaded_ts != conf['testing']['timestamp'] or loaded_ckpt != conf['testing']['ckpt_name']:
        load_inference_model(timestamp=conf['testing']['timestamp'],
                             ckpt_name=conf['testing']['ckpt_name'])

    # Getting the predictions
    outputs = []
    for fname in filenames:
        try:
            tmp = predict_audio(fpath=fname)
        except Exception as e:
            print(e)
            tmp = {'title': fname,
                   'error': str(e)}
        finally:
            os.remove(fname)
        outputs.append(tmp)

    return outputs


def predict_audio(fpath, merge=True):

    data_bytes = bytes_to_PCM_16bits(fpath)
    pred_lab, pred_prob = model_wrapper.predict(wav_file=data_bytes, merge=merge)

    if merge:
        pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(pred_prob)

    # Formatting the predictions to the required API format
    pred = {'title': os.path.basename(fpath),
            'labels': [class_names[p] for p in pred_lab],
            'probabilities': [float(p) for p in pred_prob],
            'labels_info': [class_info[p] for p in pred_lab],
            'links': {'Google Images': [image_link(class_names[p]) for p in pred_lab],
                      'Wikipedia': [wikipedia_link(class_names[p]) for p in pred_lab]
                      }
            }

    return pred


def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    params = {'tbm':'isch','q':pred_lab}
    link = base_url + requests.compat.urlencode(params)
    return link


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    link = base_url + pred_lab.replace(' ', '_')
    return link


def train(**args):
    """
    Train an image classifier
    """
    update_with_query_conf(user_args=args)
    CONF = config.conf_dict
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    config.print_conf_table(CONF)
    K.clear_session()  # remove the model loaded for prediction
    train_fn(TIMESTAMP=timestamp, CONF=CONF)

    # # Sync with NextCloud folders (if NextCloud is available)
    # try:
    #     mount_nextcloud(paths.get_models_dir(), 'rshare:/models')
    # except Exception as e:
    #     print(e)

    return {'modelname': timestamp}


def populate_parser(parser, default_conf):
    """
    Fill a parser with arguments
    """
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val['help'] if ('help' in gg_keys) else ''
            type = getattr(builtins, g_val['type']) if ('type' in gg_keys) else None
            choices = g_val['choices'] if ('choices' in gg_keys) else None

            # Additional info in help string
            help += '\n' + "<font color='#C5576B'> Group name: **{}**".format(str(group))
            if choices:
                help += '\n' + "Choices: {}".format(str(choices))
            if type:
                help += '\n' + "Type: {}".format(g_val['type'])
            help += "</font>"

            # Create arg dict
            opt_args = {'missing': json.dumps(g_val['value']),
                        'description': help,
                        'required': False,
                        }
            if choices:
                opt_args['enum'] = [json.dumps(i) for i in choices]

            parser[g_key] = fields.Str(**opt_args)

    return parser


def get_train_args():

    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([('general', default_conf['general']),
                                ('model', default_conf['model']),
                                ('preprocessing', default_conf['preprocessing']),
                                ('training', default_conf['training']),
                                ('monitor', default_conf['monitor'])])

    return populate_parser(parser, default_conf)


def get_predict_args():
    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([('testing', default_conf['testing'])])

    # Add options for modelname
    timestamp = default_conf['testing']['timestamp']
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list.remove('common')  # common files do not count as full model
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        timestamp['value'] = ''
    else:
        timestamp['value'] = timestamp_list[-1]
        timestamp['choices'] = timestamp_list

    # Add data and url fields
    parser['files'] = fields.Field(required=False,
                                   missing=None,
                                   type="file",
                                   data_key="data",
                                   location="form",
                                   description="Select the audio file you want to classify.")

    parser['urls'] = fields.Url(required=False,
                                missing=None,
                                description="Select an URL of the audio file you want to classify.")

    # missing action="append" --> append more than one url

    return populate_parser(parser, default_conf)


def get_metadata(distribution_name='audioclas'):
    """
    Function to read metadata
    """

    pkg = pkg_resources.get_distribution(distribution_name)
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par):
                _, value = line.split(": ", 1)
                meta[par] = value

    # Update information with Docker info (provided as 'CONTAINER_*' env variables)
    r = re.compile("^CONTAINER_(.*?)$")
    container_vars = list(filter(r.match, list(os.environ)))
    for var in container_vars:
        meta[var.capitalize()] = os.getenv(var)

    return meta
