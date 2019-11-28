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

import json
import os
import io
import tempfile
import warnings
from datetime import datetime
import pkg_resources
import builtins
import re
import mimetypes
from collections import OrderedDict

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from webargs import fields
from aiohttp.web import HTTPBadRequest

from audioclas import config, paths
from audioclas.train import train_fn
from audioclas.model import ModelWrapper
from audioclas.data_utils import bytes_to_PCM_16bits, mount_nextcloud


# # Mount NextCloud folders (if NextCloud is available)
# try:
#     mount_nextcloud('ncplants:/data/dataset_files', paths.get_splits_dir())
#     mount_nextcloud('ncplants:/data/images', paths.get_images_dir())
#     #mount_nextcloud('ncplants:/models', paths.get_models_dir())
# except Exception as e:
#     print(e)

# Empty model variables for inference (will be loaded the first time we perform inference)
loaded = False
graph, model, conf, class_names, class_info = None, None, None, None, None

# # Additional parameters
top_K = 5  # number of top classes predictions to return


model_wrapper = None


def load_inference_model():
    global model_wrapper

    # Clear the previous loaded model
    K.clear_session()

    # Load new model
    model_wrapper = ModelWrapper()


# def load_inference_model():
#     """
#     Load a model for prediction.
#
#     If several timestamps are available in `./models` it will load `.models/api` or the last timestamp if `api` is not
#     available.
#     If several checkpoints are available in `./models/[timestamp]/ckpts` it will load
#     `.models/[timestamp]/ckpts/final_model.h5` or the last checkpoint if `final_model.h5` is not available.
#     """
#     global loaded, graph, model, conf, class_names, class_info
#
#     # Set the timestamp
#     timestamps = next(os.walk(paths.get_models_dir()))[1]
#     if not timestamps:
#         raise BadRequest(
#             """You have no models in your `./models` folder to be used for inference.
#             Therefore the API can only be used for training.""")
#     else:
#         if 'api' in timestamps:
#             TIMESTAMP = 'api'
#         else:
#             TIMESTAMP = sorted(timestamps)[-1]
#         paths.timestamp = TIMESTAMP
#         print('Using TIMESTAMP={}'.format(TIMESTAMP))
#
#         # Set the checkpoint model to use to make the prediction
#         ckpts = os.listdir(paths.get_checkpoints_dir())
#         if not ckpts:
#             raise BadRequest(
#                 """You have no checkpoints in your `./models/{}/ckpts` folder to be used for inference.
#                 Therefore the API can only be used for training.""".format(TIMESTAMP))
#         else:
#             if 'final_model.h5' in ckpts:
#                 MODEL_NAME = 'final_model.h5'
#             else:
#                 MODEL_NAME = sorted([name for name in ckpts if name.endswith('*.h5')])[-1]
#             print('Using MODEL_NAME={}'.format(MODEL_NAME))
#
#             # Clear the previous loaded model
#             K.clear_session()
#
#             # Load the class names and info
#             splits_dir = paths.get_ts_splits_dir()
#             class_names = load_class_names(splits_dir=splits_dir)
#             class_info = None
#             if 'info.txt' in os.listdir(splits_dir):
#                 class_info = load_class_info(splits_dir=splits_dir)
#                 if len(class_info) != len(class_names):
#                     warnings.warn("""The 'classes.txt' file has a different length than the 'info.txt' file.
#                     If a class has no information whatsoever you should leave that classes row empty or put a '-' symbol.
#                     The API will run with no info until this is solved.""")
#                     class_info = None
#             if class_info is None:
#                 class_info = ['' for _ in range(len(class_names))]
#
#             # Load training configuration
#             conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
#             with open(conf_path) as f:
#                 conf = json.load(f)
#
#             # Load the model
#             model = load_model(os.path.join(paths.get_checkpoints_dir(), MODEL_NAME), custom_objects=utils.get_custom_objects())
#             graph = tf.get_default_graph()
#
#     # Set the model as loaded
#     loaded = True


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


def predict_url(args, merge=True):
    """
    Function to predict an url
    """
    catch_url_error(args['urls'])

    # Load the model if needed
    if model_wrapper is None:
        load_inference_model()

    # Download wav
    url = args['urls'][0]
    resp = requests.get(url, stream=True, allow_redirects=True)

    # Getting the predictions
    data_bytes = resp.content
    data_bytes = bytes_to_PCM_16bits(io.BytesIO(data_bytes))
    try:
        preds = model_wrapper._predict(wav_file=data_bytes,
                                       time_stamp=0)
    except ValueError:
        raise ValueError('Invalid start time: value outside audio clip')

    # Aligning the predictions to the required API format
    label_preds = [{'label_id': p[0], 'label': p[1], 'probability': p[2]} for p in preds]

    # # Filter list
    # if args['filter'] is not None and any(x.strip() != '' for x in args['filter']):
    #     label_preds = [x for x in label_preds if x['label'] in args['filter']]

    labels = [i['label'] for i in label_preds]
    probs = [i['probability'] for i in label_preds]

    return format_prediction(labels, probs)


def predict_data(args, merge=True):
    """
    Function to predict a local image
    """
    catch_localfile_error(args['files'])

    # Load the model if needed
    if model_wrapper is None:
        load_inference_model()

    # Getting the predictions
    fpath = args['files'][0].filename
    data_bytes = bytes_to_PCM_16bits(fpath)
    try:
        preds = model_wrapper._predict(wav_file=data_bytes,
                                       time_stamp=0)
    except ValueError:
        raise ValueError('Invalid start time: value outside audio clip')

    # Aligning the predictions to the required API format
    label_preds = [{'label_id': p[0], 'label': p[1], 'probability': p[2]} for p in preds]

    # # Filter list
    # if args['filter'] is not None and any(x.strip() != '' for x in args['filter']):
    #     label_preds = [x for x in label_preds if x['label'] in args['filter']]

    labels = [i['label'] for i in label_preds]
    probs = [i['probability'] for i in label_preds]

    # if not loaded:
    #     load_inference_model()
    # with graph.as_default():
    #     pred_lab, pred_prob = predict(model=model,
    #                                   X=filenames,
    #                                   conf=conf,
    #                                   top_K=top_K,
    #                                   filemode='local',
    #                                   merge=merge)
    #
    # if merge:
    #     pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(pred_prob)

    return format_prediction(labels, probs)


def format_prediction(labels, probabilities):
    d = {
        "status": "ok",
         "predictions": [],
    }

    for label_id, prob in zip(labels, probabilities):
        name = label_id #class_names[label_id]

        pred = {
            # "label_id": int(label_id),
            "label": name,
            "probability": float(prob),
            "info": {
                "links": {'Google images': image_link(name),
                          'Wikipedia': wikipedia_link(name)}#,
                # 'metadata': class_info[label_id],
            },
        }
        d["predictions"].append(pred)
    return d


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


@catch_error
def train(**user_conf):
    """
    Parameters
    ----------
    user_conf : dict
        Json dict (created with json.dumps) with the user's configuration parameters that will replace the defaults.
        Must be loaded with json.loads()
        For example:
            user_conf={'num_classes': 'null', 'lr_step_decay': '0.1', 'lr_step_schedule': '[0.7, 0.9]', 'use_early_stopping': 'false'}
    """
    CONF = config.CONF

    # Update the conf with the user input
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            g_val['value'] = json.loads(user_conf[g_key])

    # Check the configuration
    try:
        config.check_conf(conf=CONF)
    except Exception as e:
        raise Exception(e)

    CONF = config.conf_dict(conf=CONF)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    config.print_conf_table(CONF)
    K.clear_session() # remove the model loaded for prediction
    train_fn(TIMESTAMP=timestamp, CONF=CONF)

    # Sync with NextCloud folders (if NextCloud is available)
    try:
        mount_nextcloud(paths.get_models_dir(), 'ncplants:/models')
    except Exception as e:
        print(e)


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


# def get_train_args():
#
#     default_conf = config.CONF
#     default_conf = OrderedDict([('general', default_conf['general']),
#                                 ('model', default_conf['model']),
#                                 ('training', default_conf['training']),
#                                 ('monitor', default_conf['monitor']),
#                                 ('dataset', default_conf['dataset']),
#                                 ('augmentation', default_conf['augmentation'])])
#     return get_args(default_conf)


def get_predict_args():
    parser = OrderedDict()

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

    return parser


def get_metadata(distribution_name='audio-classification-tf'):
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
