
# Auto Tuning Spectral Clustering

<img src="./pics/adj_mat.png" width="35%" height="35%">
<img src="./pics/gp_vs_nme.png" width="40%" height="40%">  

Python3 code for the IEEE SPL paper ["Auto-Tuning Spectral Clustering for SpeakerDiarization Using Normalized Maximum Eigengap"](https://drive.google.com/file/d/1CdEJPrpW6pRCObrppcZnw0_hRwWIHxi8/view?usp=sharing)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### TLDR; One-click demo file.

* _run_demo_clustering.sh_ installs the virtualenv and runs spectral clustering example.
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
#### How to run the spectral clustering code:
```bash
python spectral_opt.py --distance_score_file $DISTANCE_SCORE_FILE \
                       --threshold $threshold \
                       --score-metric $score_metric \
                       --xvector_window $xvector_window \
                       --asr_spk_turn_est_scp 'None' \
                       --max_speaker $max_speaker \
                       --max_speaker_list 'None'\
                       --embedding_scp $embedding_scp \
                       --spt_est_thres $spt_est_thres \
                       --segment_file_input_path $SEGMENT_FILE_INPUT_PATH \
                       --spk_labels_out_path $SPK_LABELS_OUT_PATH \
                       --reco2num_spk $reco2num_spk || exit 1
```
#### Arguments:

**distance_score_file**: A list of affinity matrix files.  
Two options are available:
(1) scores.scp: Kaldi style scp file that contains the absolute path to .ark files and its binary address. Space separted <utt_id> and <path>.

Ex) 
```
iaaa /path/sample_CH_xvector/cos_scores/scores.1.ark:5
iafq /path/sample_CH_xvector/cos_scores/scores.1.ark:23129
<utt_id> <path>
```
(2) scores.txt: List of <utt_id> and the absolute path to .npy files.
Ex) 
```
iaaa /path/sample_CH_xvector/cos_scores/iaaa.npy
iafq /path/sample_CH_xvector/cos_scores/iafq.npy
<utt_id> <path>
```

**threshold**:
```bash
threshold
```

**score-metric**:
score-metric $score_metric \
```bash
threshold
```
**xvector_window**: 
xvector_window $xvector_window \
```bash
threshold
```
**max_speaker**:
max_speaker $max_speaker \
```bash
threshold
```
**embedding_scp**:
embedding_scp $embedding_scp \
```bash
threshold
```

**spt_est_thres**:
spt_est_thres $spt_est_thres \
```bash
threshold
```
**segment_file_input_path**:
segment_file_input_path $SEGMENT_FILE_INPUT_PATH \
```bash
threshold
```
**segment_file_input_path**:
spk_labels_out_path $SPK_LABELS_OUT_PATH \
```bash
threshold
```
**reco2num_spk**
reco2num_spk $reco2num_spk
```bash
threshold
```


## Authors


