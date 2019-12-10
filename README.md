
# Auto Tuning Spectral Clustering

<img src="./pics/adj_mat.png" width="40%" height="40%">
<img src="./pics/gp_vs_nme.png" width="40%" height="40%">  

Python3 code for the IEEE SPL paper ["Auto-Tuning Spectral Clustering for SpeakerDiarization Using Normalized Maximum Eigengap"](https://drive.google.com/file/d/1CdEJPrpW6pRCObrppcZnw0_hRwWIHxi8/view?usp=sharing)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### TLDR; One-click demo file.

* _run_demo_clustering.sh_ installs a virtualenv and runs spectral a clustering example.
* [**virtualenv**](https://docs.python-guide.org/dev/virtualenvs/) should be installed on your machine.

```bash
source run_demo_clustering.sh
```
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

You have to first have [**virtualenv**](https://docs.python-guide.org/dev/virtualenvs/) installed on your machine. Install [**virtualenv**](https://docs.python-guide.org/dev/virtualenvs/) with the following command:
```
sudo pip3 install virtualenv 
```
If you installed virtualenv, run the "install_venv.sh" script to make a virtual-env.
```
source install_venv.sh
```
This command will create a folder named "env_nmesc".


### Usage Example
#### Running the python code with arguments:
```bash
python spectral_opt.py --distance_score_file $DISTANCE_SCORE_FILE \
                       --threshold $threshold \
                       --score-metric $score_metric \
                       --xvector_window $xvector_window \
                       --max_speaker $max_speaker \
                       --embedding_scp $embedding_scp \
                       --spt_est_thres $spt_est_thres \
                       --segment_file_input_path $SEGMENT_FILE_INPUT_PATH \
                       --spk_labels_out_path $SPK_LABELS_OUT_PATH \
                       --reco2num_spk $reco2num_spk 
```
#### Arguments:

**distance_score_file**: A list of affinity matrix files.  
```
DISTANCE_SCORE_FILE=$PWD/sample_CH_xvector/cos_scores/scores.scp
DISTANCE_SCORE_FILE=$PWD/sample_CH_xvector/cos_scores/scores.txt
```
Two options are available:  

(1) scores.scp: Kaldi style scp file that contains the absolute path to .ark files and its binary address. Space separted <utt_id> and <path>.

ex) scores.scp
```
iaaa /path/sample_CH_xvector/cos_scores/scores.1.ark:5
iafq /path/sample_CH_xvector/cos_scores/scores.1.ark:23129
...
```

(2) scores.txt: List of <utt_id> and the absolute path to .npy files.  
ex) scores.txt
```
iaaa /path/sample_CH_xvector/cos_scores/iaaa.npy
iafq /path/sample_CH_xvector/cos_scores/iafq.npy
...
```

* **threshold**: Manually setup a threshold. We apply this threshold for all utterances.  
ex) 
```bash
threshold=0.05
```

* **score-metric**: Use 'cos' to apply for affinity matrix based on cosine similarity.  
ex) 
```bash
score_metric='cos'
```

* **max_speaker**: Default is 8. If you do not provide oracle number of speakers (reco2num_spk), the estimated number of speakers is capped by _max_speaker_.  
ex) 
```bash
max_speaker=8
```
* **embedding_scp**:
embedding_scp $embedding_scp \
```bash
threshold
```

* **spt_est_thres**:
spt_est_thres $spt_est_thres \
```bash
spt_est_thres='None'
spt_est_thres="thres_utts.txt"
```

* **segment_file_input_path**:
```bash
threshold
```

* **reco2num_spk**: A list of oracle number of speakers. Default is 'None'.
reco2num_spk $reco2num_spk
```bash
reco2num_spk='None'
reco2num_spk='oracle_num_of_spk.txt'
```

oracle_num_of_spk.txt
```
iaaa 2
iafq 2
iabe 4
iadf 6
...
```

## Authors


