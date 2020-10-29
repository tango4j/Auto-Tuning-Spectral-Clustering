venv_name=env_nmesc
virtualenv -p $(which python3) $venv_name
source $PWD/"$venv_name"/bin/activate
pip3 install -U scikit-learn==0.22
pip3 install kaldi_io==0.9.1
pip3 install pyamg
pip3 install ipdb
