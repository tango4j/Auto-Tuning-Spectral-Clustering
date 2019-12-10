#!/bin/bash
# Copyright       2016  David Snyder
#            2017-2018  Matthew Maciejewski
# Apache 2.0.

# This script performs agglomerative clustering using scored
# pairs of subsegments and produces a rttm file with speaker
# labels derived from the clusters.

# Begin configuration section.


cmd="run.pl"
python_env=~/virtualenvs/py3_keras_venv/bin/activate
stage=0
nj=10
cleanup=true
threshold=0.0
score_metric='cos'
read_costs=false
reco2num_spk='None'
out_dir='None'
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

echo "num of args: " $#

if [ $# != 2 ]; then
  echo "Usage: $0 <src-dir> <dir>"
  echo " e.g.: $0 exp/ivectors_callhome exp/ivectors_callhome/results"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --python_env                                     # Python path to run the python script"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --threshold <threshold|0>                        # Cluster stopping criterion. Clusters with scores greater"
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

xvecdir=$1
dir=$2
echo "ARGS: ", $xvecdir, $dir

source $python_env

DATE=`date '+%Y_%m_%d_%H_%M_%S'`

mkdir -p $dir/tmp
echo "Checking necessary files... "
for f in $xvecdir/xvector.scp $xvecdir/spk2utt $xvecdir/utt2spk $xvecdir/segments; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done
cp $xvecdir/xvector.scp $dir/tmp/feats.scp
cp $xvecdir/spk2utt $dir/tmp/
cp $xvecdir/utt2spk $dir/tmp/
cp $xvecdir/segments $dir/tmp/
cp $xvecdir/spk2utt $dir/
cp $xvecdir/utt2spk $dir/
cp $xvecdir/segments $dir/

#utils/fix_data_dir.sh $dir/tmp > /dev/null
#utils/filter_scp.pl $xvecdir/spk2utt $dir/tmp/feats.scp > $dir/tmp/feats.scp
echo "Splitting files... "

#sdata=$dir/tmp/split$nj;
#utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

#for JOB in $(seq 1 1 $nj); do
    #utils/filter_scp.pl $sdata/$JOB/spk2utt $sdata/$JOB/feats.scp > $sdata/$JOB/feats.scp
#done

############################################################################
# Non parallel mode 
echo "Starting Script: affinity_score.py"
python affinity_score.py \
      --scp $dir/tmp/feats.scp \
      --score-metric $score_metric \
      --spk2utt $dir/tmp/spk2utt \
      --utt2spk $dir/tmp/utt2spk \
      --segments $dir/tmp/segments \
      --parallel_job 1 \
      --scores $dir/scores.scp || exit 1;
############################################################################

############################################################################
#if [ $stage -le 0 ]; then
  #echo "$0: calculating scores"
  #$cmd JOB=1:$nj $dir/log/score_embedding.JOB.log \
    #python affinity_score.py \
      #--scp $sdata/JOB/feats.scp \
      #--score-metric $score_metric \
      #--spk2utt $sdata/JOB/spk2utt \
      #--utt2spk $sdata/JOB/utt2spk \
      #--segments $sdata/JOB/segments \
      #--parallel_job JOB \
      #--scores $dir/scores.JOB.scp || exit 1;

#fi
#if [ $stage -le 1 ]; then
  ##echo "$0: combining PLDA scores across jobs"
  #for j in $(seq $nj); do cat $dir/scores.$j.scp; done >$dir/scores.scp || exit 1;
  #for j in $(seq $nj); do cat $dir/segments.$j; done >$dir/segments || exit 1;
#fi
############################################################################

#############################################################################

echo "$0 score - $score_metric Calculation Complete"

if $cleanup ; then
  rm -rf $dir/tmp || exit 1;
fi

