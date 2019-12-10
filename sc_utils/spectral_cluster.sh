#!/bin/bash

# Copyright       2016  David Snyder
#            2017-2018  Matthew Maciejewski
# Apache 2.0.

# This script performs agglomerative clustering using scored
# pairs of subsegments and produces a rttm file with speaker
# labels derived from the clusters.

# Begin configuration section.
cmd="run.pl"
python_env=~/python/path/env/is/not/specified!!
stage=0
nj=10
cleanup=true
threshold=0.0
score_metric="score metric NOT specified!"
xvector_window=1000
read_costs=false
reco2num_spk='None'
spt_est_thres='None'
max_speaker=6
max_speaker_list='None'
asr_spk_turn_est_scp='None'
embedding_scp='None'

echo "$0 $@"  # Print the command line for logging

#if [ -f path.sh ]; then . ./path.sh; fi
#. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <src-dir> <dir>"
  echo " e.g.: $0 exp/ivectors_callhome exp/ivectors_callhome/results"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --python_env                                     # Python path to run the spectral_opt.py script"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --threshold <threshold|0>                        # Cluster stopping criterion. Clusters with scores greater"
  echo "  --score-metric <score-metric|0>                  # Score metric"
  echo "                                                   # than this value will be merged until all clusters"
  echo "                                                   # exceed this value."
  echo "  --read-costs <read-costs|false>                  # If true, interpret input scores as costs, i.e. similarity"
  echo "                                                   # is indicated by smaller values. If enabled, clusters will"
  echo "                                                   # be merged until all cluster scores are less than the"
  echo "                                                   # threshold value."
  echo "  --reco2num-spk <reco2num-spk-file>               # File containing mapping of recording ID"
  echo "                                                   # to number of speakers. Used instead of threshold"
  echo "                                                   # as stopping criterion if supplied."
  echo "  --cleanup <bool|false>                           # If true, remove temporary files"
  exit 1;
fi

srcdir=$1
dir=$2

source $python_env

mkdir -p $dir/tmp

for f in $srcdir/scores.scp $srcdir/spk2utt $srcdir/utt2spk $srcdir/segments ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

cp $srcdir/spk2utt $dir/tmp/
cp $srcdir/utt2spk $dir/tmp/
cp $srcdir/segments $dir/tmp/
#utils/fix_data_dir.sh $dir/tmp > /dev/null


#sdata=$dir/tmp/split$nj;
#utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log


### Arguments for ASR diarization END

DISTANCE_SCORE_FILE=$srcdir/"scores.scp"
THRESHOLD=$threshold
SEGMENT_FILE_INPUT_PATH=$srcdir"/segments"
SPK_LABELS_OUT_PATH="$dir/labels"

if [ -e "$reco2num_spk" ]; then
    RECO2NUM_SPK=$reco2num_spk
else 
    RECO2NUM_SPK="None"
fi   

PROTO_TYPE=40
CLUS_ALG='DistScoreInput'
SCORE_METRIC=$score_metric

echo "++++++++++++++++ Using threshold THRESHOLD: " $THRESHOLD 
echo "++++++++++++++++ Using score metric: " $SCORE_METRIC
python spectral_opt.py --distance_score_file $DISTANCE_SCORE_FILE \
                   --threshold $THRESHOLD \
                   --score-metric $SCORE_METRIC \
                   --xvector_window $xvector_window \
                   --asr_spk_turn_est_scp $asr_spk_turn_est_scp \
                   --max_speaker $max_speaker \
                   --max_speaker_list $max_speaker_list \
                   --embedding_scp $embedding_scp \
                   --spt_est_thres $spt_est_thres \
                   --segment_file_input_path $SEGMENT_FILE_INPUT_PATH \
                   --spk_labels_out_path $SPK_LABELS_OUT_PATH \
                   --reco2num_spk $RECO2NUM_SPK || exit 1


if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  make_rttm.py $srcdir/segments $dir/labels $dir/rttm || exit 1;
  echo "RTTM calculation was successful. "
fi

if $cleanup ; then
  rm -r $dir/tmp || exit 1;
fi

