virtualenv -p $(which python3) $1
source $PWD/"$1"/bin/activate
pip3 install -U scikit-learn==0.22
pip3 install kaldi_io==0.9.1
pip3 ipdb
