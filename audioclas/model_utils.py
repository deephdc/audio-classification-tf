"""
Miscellanous functions to handle models.

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import os
import json

from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.layers import Dense

from audioclas import paths


def create_model(CONF, base_model):
    """
    Parameters
    ----------
    CONF : dict
        Contains relevant configuration parameters of the model
    base_model : tf model
        Audio classification model trained for AudioNet
    """
    # Remove last layer of the pre-trained model and add custom layers at the top to adapt it to our problem
    x = base_model.layers[-3].output
    predictions = Dense(CONF['model']['num_classes'],
                        activation='softmax')(x)

    # Full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # # Add L2 reguralization for all the layers in the whole model
    # if CONF['training']['l2_reg']:
    #     for layer in model.layers:
    #         layer.kernel_regularizer = regularizers.l2(CONF['training']['l2_reg'])

    return model, base_model


def save_to_pb(keras_model, export_path):
    """
    Save keras model to protobuf for Tensorflow Serving.
    Source: https://medium.com/@johnsondsouza23/export-keras-model-to-protobuf-for-tensorflow-serving-101ad6c65142

    Parameters
    ----------
    keras_model: Keras model instance
    export_path: str
    """

    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)

    # Build the Protocol Buffer SavedModel at 'export_path'
    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(inputs={"images": keras_model.input},
                                      outputs={"scores": keras_model.output})

    with K.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                             signature_def_map={"predict": signature})

    builder.save()


def export_h5_to_pb(path_to_h5, export_path):
    """
    Transform Keras model to protobuf
    """
    model = load_model(path_to_h5)
    save_to_pb(model, export_path)


def save_conf(conf):
    """
    Save CONF to a txt file to ease the reading and to a json file to ease the parsing.

    Parameters
    ----------
    conf : 1-level nested dict
    """
    save_dir = paths.get_conf_dir()

    # Save dict as json file
    with open(os.path.join(save_dir, 'conf.json'), 'w') as outfile:
        json.dump(conf, outfile, sort_keys=True, indent=4)

    # Save dict as txt file for easier redability
    txt_file = open(os.path.join(save_dir, 'conf.txt'), 'w')
    txt_file.write("{:<25}{:<30}{:<30} \n".format('group', 'key', 'value'))
    txt_file.write('=' * 75 + '\n')
    for key, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            txt_file.write("{:<25}{:<30}{:<15} \n".format(key, g_key, str(g_val)))
        txt_file.write('-' * 75 + '\n')
    txt_file.close()
