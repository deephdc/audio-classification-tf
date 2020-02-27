# Adapted from https://github.com/IBM/MAX-Audio-Classifier/blob/master/core/model.py

import os

import numpy as np
import tensorflow as tf

from audioclas.embeddings import vggish_input, vggish_params, vggish_postprocess, vggish_slim
from audioclas import paths


DEFAULT_EMBEDDING_CHECKPOINT = os.path.join(paths.get_models_dir(), 'common', 'vggish_model.ckpt')
DEFAULT_PCA_PARAMS = os.path.join(paths.get_models_dir(), 'common', 'vggish_pca_params.npz')


class ModelWrapper():
    """
    Contains core functions to generate embeddings and classify them.
    Also contains any helper function required.
    """

    def __init__(self, classifier_model, embedding_checkpoint=DEFAULT_EMBEDDING_CHECKPOINT,
                 pca_params=DEFAULT_PCA_PARAMS):

        # Initialize the classifier model
        self.session_classify = tf.keras.backend.get_session()
        self.classify_model = tf.keras.models.load_model(classifier_model, compile=False)

        # Initialize the vgg-ish embedding model
        self.graph_embedding = tf.Graph()
        with self.graph_embedding.as_default():
            self.session_embedding = tf.Session()
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.session_embedding, embedding_checkpoint)
            self.features_tensor = self.session_embedding.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.session_embedding.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        # Prepare a postprocessor to munge the vgg-ish model embeddings.
        self.pproc = vggish_postprocess.Postprocessor(pca_params)

    def generate_embeddings(self, wav_file):
        """
        Generates embeddings as per the Audioset VGG-ish model.
        Post processes embeddings with PCA Quantization
        Input args:
            wav_file   = /path/to/audio/in/wav/format.wav
        Returns:
            numpy array of shape (x,128) where x is any arbitrary whole number >1.
        """
        examples_batch = vggish_input.wavfile_to_examples(wav_file)
        [embedding_batch] = self.session_embedding.run([self.embedding_tensor],
                                                       feed_dict={self.features_tensor: examples_batch})
        return self.pproc.postprocess(embedding_batch)

    def classify_embeddings(self, processed_embeddings):
        """
        Performs classification on PCA Quantized Embeddings.
        Input args:
            processed_embeddings = numpy array of shape (1,10,128), dtype=float32
        Returns:
            class_scores = Output probabilities for the 527 classes - numpy array of shape (1,527).
        """
        output_tensor = self.classify_model.output
        input_tensor = self.classify_model.input
        class_scores = output_tensor.eval(feed_dict={input_tensor: processed_embeddings}, session=self.session_classify)
        return class_scores

    def predict(self, wav_file, top_K=5, merge=True):
        """Predict function
        If merge=True we use the whole audio to make predictions (ie. we average over predictions of 10s samples).
        If merge=False we return separate predictions for each 10s sample.
        """
        raw_embeddings = self.generate_embeddings(wav_file)
        embeddings_processed = self.classifier_pre_process(raw_embeddings)
        output = self.classify_embeddings(embeddings_processed)

        if merge:
            output = np.mean(output, axis=0)  # take the mean across the batch
            lab = np.argsort(output)[::-1]  # sort labels in descending prob
            lab = lab[:top_K]  # keep only top_K labels
            lab = np.expand_dims(lab, axis=0)  # add extra dimension to make to output have a shape (1, top_k)
            prob = output[lab]
        else:
            lab = np.argsort(output, axis=1)[:, ::-1]  # sort labels in descending prob
            lab = lab[:, :top_K]  # keep only top_K labels
            prob = output[np.repeat(np.arange(len(lab)), lab.shape[1]),
                          lab.flatten()].reshape(lab.shape)  # retrieve corresponding probabilities

        return lab, prob

    @staticmethod
    def classifier_pre_process(embeddings, start_time=0):
        """
        Helper function to make sure input to classifier the model is of standard size.
        * Clips/Crops audio clips embeddings to start at `start_time`
        * Augments audio embeddings to an upper multiple of 10 seconds by repeating in loop.
        * Reshape embeddings to a batch of 10s samples (N, 10, 128)
        * Converts dtype of embeddings from uint8 to float32

        Input args :
            embeddings = numpy array of shape (x,128) where x is any arbitrary whole number >1.
        Returns:
            embeddings = numpy array of shape (N,10,128), dtype=float32.
        """
        # Crop at start time
        embeddings_ts = int(start_time / vggish_params.EXAMPLE_HOP_SECONDS)
        if 0 <= embeddings_ts < embeddings.shape[0]:
            embeddings = embeddings[embeddings_ts:, :]
        else:
            raise ValueError('Invalid start time.')

        # Expand to upper 10s multiple (eg. 13s --> 20s) by repeating in loop
        if embeddings.shape[0] < 10:
            new_embeddings = embeddings
            rep = int(np.ceil(10 / embeddings.shape[0]))
            for _ in range(rep - 1):
                new_embeddings = np.vstack((new_embeddings, embeddings))
        else:
            new_embeddings = np.vstack((embeddings, embeddings[:10, :]))
        embeddings = new_embeddings[:int(new_embeddings.shape[0] // 10) * 10]

        # Prepare batch of 10s samples
        embeddings = embeddings.reshape(-1, 10, 128)

        # Normalize and move from uint8_to_float32
        embeddings = (np.float32(embeddings) - 128.) / 128.

        return embeddings
