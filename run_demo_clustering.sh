
. ./sc_utils/bash_module.sh

######################################################################
### Python Environment Setup

env_name=env_nmesc
python_env=$PWD/$env_name/bin/activate

if [ -d "$PWD/$env_name/bin" ]; then
    text_yellow_info "The python_envfolder exists: $PWD/$env_name"
else
   . install_venv.sh "$env_name"
fi
source $python_env || exit 1


######################################################################
### Cosine Distance Calculation (Kaldi should be installed at ~/kaldi)
### This function saves 
### 1. Kaldi .ark/.scp file 
### 2. Numpy matrix .npy file

nj=1
score_metric='cos'
data_dir=$PWD/sample_CH_xvector
xvec_dir=$data_dir/xvector_embeddings


if [ -f "$data_dir/cos_scores/scores.scp" ]; then
    text_yellow_info "Cosine similariy scores exist: $data_dir/cos_scores"
else
    mkdir -p $PWD/sample_CH_xvector/cos_scores
    
    pushd $PWD/sc_utils
    text_yellow_info "Starting Script: affinity_score.py"
    ./score_embedding.sh --cmd "run.pl --mem 5G" \
                         --nj $nj \
                         --score-metric $score_metric \
                         --out-dir $data_dir/cos_scores \
                          $data_dir/xvector_embeddings \
                          $data_dir/cos_scores || exit 1
    popd
    text_yellow_info "Cosine distance caluclation has been finished"
fi

######################################################################
### Spectral Clustering
### Two input formats are supported for the input affinity matrix:
### 1. Kaldi .ark/.scp file (e.g. PLDA score)
###    Ex) List file: ./sample_CH_xvector/cos_scores/scores.scp
###        Affinity matrix in binary: ./sample_CH_xvector/cos_scores/scores.1.ark
###
### 2. Numpy matrix: <utt_name>.npy, and scores.txt lists all the .npy 
###    Ex) List file: ./sample_CH_xvector/cos_scores/scores.txt
###        Affinity matrix file: ./sample_CH_xvector/cos_scores/iaaa.npy 

score_metric='cos'
max_speaker=8
xvector_window=1.5
spt_est_thres="NMESC"
threshold='None'
reco2num_spk="None"
embedding_scp=$data_dir/xvector_embeddings/xvector.scp

SEGMENT_FILE_INPUT_PATH=$data_dir/xvector_embeddings/segments
SPK_LABELS_OUT_PATH=$data_dir/evaluation_output/labels


# Using .npy numpy matrix format
DISTANCE_SCORE_FILE=$data_dir/cos_scores/"scores.txt"
text_yellow_info "Running Spectral Clustering with .npy input..."
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

# Using kaldi matrix(ark file) format
DISTANCE_SCORE_FILE=$data_dir/cos_scores/"scores.scp"
text_yellow_info "Running Spectral Clustering with .npy input..."
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


text_yellow_info "Computing RTTM"
python ./sc_utils/make_rttm.py $SEGMENT_FILE_INPUT_PATH $SPK_LABELS_OUT_PATH $data_dir/evaluation_output/output.rttm || exit 1
text_yellow_info "RTTM calculation was successful. "

######################################################################
### Evaluation and Display Code
### Colar of 0.25 sec is applied and overlap regions are ignored.

DER_fn="MicroCH_DER.txt"
cat $data_dir/evaluation_output/output.rttm \
  | $PWD/src/md-eval.pl -1 -c 0.25 -r $data_dir/true_rttm/rttm -s - \
  > $data_dir/DER_results/$DER_fn || exit 1

display_and_save_DER $data_dir/DER_results/$DER_fn 


