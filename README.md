DEEP Open Catalogue: Audio classifier
=====================================

[![Build Status](https://jenkins.indigo-datacloud.eu:/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/audio-classification-tf/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/audio-classification-tf/job/master)

**Author/Mantainer:** [Ignacio Heredia](https://github.com/IgnacioHeredia) (CSIC)

**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has
received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

This is a plug-and-play tool to perform audio classification with Deep Learning. It allows the user to classify their
samples of audio as well as training their own classifier for a custom problem. The classifier is currently
[pretrained](models/default) on the 527 high-level classes from the [AudioSet](https://research.google.com/audioset/) dataset.

You can find more information about it in the [DEEP Marketplace](https://marketplace.deep-hybrid-datacloud.eu/modules/train-an-audio-classifier.html).

![demo](./reports/figures/demo.png)


## Installing this module

### Local installation

> **Requirements**
>
> This project has been tested in Ubuntu 18.04 with Python 3.6.5. Further package requirements are described in the
> `requirements.txt` file.
> - To support a wide range of audio formats we need  to make use of the FFMPEG library. To install it in Linux please run:
>    ```bash
>    apt-get install ffmpeg libavcodec-extra
>    ```
> - It is a requirement to have [Tensorflow>=1.14.0 installed](https://www.tensorflow.org/install/pip) (either in gpu
> or cpu mode). This is not listed in the `requirements.txt` as it [breaks GPU support](https://github.com/tensorflow/tensorflow/issues/7166).

To start using this framework clone the repo and download the [default weights](https://cephrgw01.ifca.es:8080/swift/v1/audio-classification-tf/default.tar.gz):

```bash
git clone https://github.com/deephdc/audio-classification-tf
cd audio-classification-tf
pip install -e .
curl -o ./models/default.tar.gz https://cephrgw01.ifca.es:8080/swift/v1/audio-classification-tf/default.tar.gz
cd models && tar -zxvf default.tar.gz && rm default.tar.gz 
```
now run DEEPaaS:
```
deepaas-run --listen-ip 0.0.0.0
```
and open http://0.0.0.0:5000/ui and look for the methods belonging to the `audioclas` module.

### Docker installation

We have also prepared a ready-to-use [Docker container](https://github.com/deephdc/DEEP-OC-audio-classification-tf) to
run this module. To run it:

```bash
docker search deephdc
docker run -ti -p 5000:5000 -p 6006:6006 -p 8888:8888 deephdc/deep-oc-audio-classification-tf
```

Now open http://0.0.0.0:5000/ui and look for the methods belonging to the `audioclas` module.


## Train an audio classifier

You can train your own audio classifier with your custom dataset. For that you have to:

### 1.1 Prepare the audio files

Put your images in the`./data/audios` folder. If you have your data somewhere else you can use that location by setting
 the `dataset_directory` parameter in the training args. 
Please use a standard audio format (like `.mp3` or `.wav`).

> **Note** The classifier works on 10s samples. So if an audio file is longer/shorter than 10s it will be concatenated
> in loop to make it last a multiple of 10s. So a 22 seconds audio file will create 3 embeddings: `*-0.npy`, `*-1.npy`
> and `*-2.npy`. 


### Prepare the data splits

First you need add to the `./data/dataset_files` directory the following files:

| *Mandatory files* | *Optional files*  | 
|:-----------------------:|:---------------------:|
|  `classes.txt`, `train.txt` |  `val.txt`, `test.txt`, `info.txt`|

The `train.txt`, `val.txt` and `test.txt` files associate an audio name (or relative path) to a label number (that has
to *start at zero*).
The `classes.txt` file translates those label numbers to label names.
Finally the `info.txt` allows you to provide information (like number of audio files in the database) about each class.

If you are using the option `compute_embeddings=False` then the file path should point yo the `.npy` path of the
embedding instead than the original audio file and the `dataset_directory` parameter should point to the folder
containing the embeddings. 

You can find examples of these files at  `./data/demo-dataset_files`.

### Train the classifier

Go to http://0.0.0.0:5000/ui and look for the ``TRAIN`` POST method. Click on 'Try it out', change whatever training args
you want and click 'Execute'. The training will be launched and you will be able to follow its status by executing the 
``TRAIN`` GET method which will also give a history of all trainings previously executed.

If the module has some sort of training monitoring configured (like Tensorboard) you will be able to follow it at 
http://0.0.0.0:6006.


## Test an audio classifier

Go to http://0.0.0.0:5000/ui and look for the `PREDICT` POST method. Click on 'Try it out', change whatever test args
you want and click 'Execute'. You can **either** supply a:

* a `data` argument with a path pointing to an audio file or a compressed file (eg. zip, tar, ...) containing audio
  files.

OR
* an `url` argument with an URL pointing to an audio file or a compressed file (eg. zip, tar, ...) containing audio
  files. Here is an [example](https://file-examples.com/wp-content/uploads/2017/11/file_example_WAV_1MG.wav) of such
  an url that you can use for testing purposes.

## Acknowledgments

The code in this project is based on the [original repo](https://github.com/IBM/MAX-Audio-Classifier) by
[IBM](https://github.com/IBM), and implements the paper
['Multi-level Attention Model for Weakly Supervised Audio Classification'](https://arxiv.org/abs/1803.02353) by Yu et al.

The main changes with respect to the original repo are that:

* we have added a training method so that the user is able to create his own custom classifier
* the code has been packaged into an installable Python package
* FFMPEG has been added to support a wider range of audio formats
* it has been made compatible with the [DEEPaaS API](http://docs.deep-hybrid-datacloud.eu/en/latest/user/overview/api.html)

If you consider this project to be useful, please consider citing the DEEP Hybrid DataCloud project:

* García, Álvaro López, et al. ["A Cloud-Based Framework for Machine Learning Workloads and Applications."](https://ieeexplore.ieee.org/abstract/document/8950411/authors) IEEE Access 8 (2020): 18681-18692.
 
along with any of the references below:

* _Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, Marvin Ritter_,["Audio set: An ontology and human-labeled dataset for audio events"](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45857.pdf), IEEE ICASSP, 2017.

* _Qiuqiang Kong, Yong Xu, Wenwu Wang, Mark D. Plumbley_,["Audio Set classification with attention model: A probabilistic perspective."](https://arxiv.org/pdf/1711.00927.pdf) arXiv preprint arXiv:1711.00927 (2017).

* _Changsong Yu, Karim Said Barsim, Qiuqiang Kong, Bin Yang_ ,["Multi-level Attention Model for Weakly Supervised Audio Classification."](https://arxiv.org/pdf/1803.02353.pdf) arXiv preprint arXiv:1803.02353 (2018).

* _S. Hershey, S. Chaudhuri, D. P. W. Ellis, J. F. Gemmeke, A. Jansen, R. C. Moore, M. Plakal, D. Platt, R. A. Saurous, B. Seybold et  al._, ["CNN architectures for large-scale audio classification,"](https://arxiv.org/pdf/1609.09430.pdf) arXiv preprint arXiv:1609.09430, 2016.
