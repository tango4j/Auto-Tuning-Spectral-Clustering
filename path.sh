export KALDI_ROOT="~/kaldi"
export DIAR_PWD=$KALDI_ROOT/egs/callhome_diarization/v1 
export PATH=$DIAR_PWD/utils:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk/bin:$DIAR_PWD:$PATH
export PYTHONPATH=$KALDI_ROOT/egs/wsj/s5/steps/libs:$PYTHONPATH

# Set up the diarization folders
rm -rf ./diarization
rm -rf ./utils

ln -s $DIAR_PWD/utils ./utils
ln -s $DIAR_PWD/diarization ./diarization

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1

. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
echo "<<< PATH  activated for clustering >>>"
