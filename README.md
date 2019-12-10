
# Auto Tuning Spectral Clustering

<img src="./pics/adj_mat.png" width="35%" height="35%">
<img src="./pics/gp_vs_nme.png" width="40%" height="40%">


Python3 code for the IEEE SPL paper ["Auto-Tuning Spectral Clustering for SpeakerDiarization Using Normalized Maximum Eigengap"](https://drive.google.com/file/d/1CdEJPrpW6pRCObrppcZnw0_hRwWIHxi8/view?usp=sharing)




## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* This repo is based on python 3.7.

* The mainly required python3 libraries:
```
scikit-learn==0.22
kaldi_io==0.9.1
```
* [Kaldi](https://kaldi-asr.org/doc/about.html) is required to reproduce the numbers in the paper. Go to [Kaldi install](http://jrmeyer.github.io/asr/2016/01/26/Installing-Kaldi.html) to install Kaldi software.

* [Kaldi](https://kaldi-asr.org/doc/about.html) should  be installed in your home folder `~/kaldi` to be successfully loaded.

* You can still run the clustering algorithm without [Kaldi](http://jrmeyer.github.io/asr/2016/01/26/Installing-Kaldi.html) by saving your affinity matrix into .npy.

### Installing

You have to first have [_virtualenv_ ](https://docs.python-guide.org/dev/virtualenvs/) installed on your machine. Install [_virtualenv_ ](https://docs.python-guide.org/dev/virtualenvs/) with the following command:
```
sudo pip3 install virtualenv 
```

If you installed virtualenv, run the "install_venv.sh" script to make a virtual-env.
```
./install_venv.sh
```
This command will create a folder named "env_nmesc".




## Authors


