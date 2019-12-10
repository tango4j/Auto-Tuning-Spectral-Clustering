import numpy as np
import pickle
import os
from subprocess import Popen, PIPE
import time
import datetime
import ipdb
from sklearn.utils import shuffle

def noremDiv(nu, de):
    if nu % de == 0:
        return nu // de
    else:
        return nu // de + 1

def batchDiv(nu, de):
    if nu % de == 0:
        return (nu // de) - 1
    else:
        return nu // de + 1

def cprint(st, c='r'):
    if c=='r':
        CRED = '\033[91m'
    elif c=='g':
        CRED = '\033[92m'
    elif c=='b':
        CRED = '\033[94m'
    elif c=='y':
        CRED = '\033[93m'
    CEND = '\033[0m'
    print(CRED + st + CEND)

def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

def rsc(x):
    return x.split(':')[0]

def bashGet(bash_command):
    p = Popen(bash_command.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    txtout = output.decode('utf-8')
    return txtout 

def dictConvert(inDict):
    key_list = list(inDict.keys())
    out = {}
    for t in key_list:
        # print(inDict[t])
        D = inDict[t].split('_')# speaker, start, dur, word
        out.update({t: [D[0], int(100*float(D[1])), int(100*float(D[2])), D[3]]})
    return out

def dictClean_Pickle(b):
    trans_dict = {}
    raw_dict = b[0]
    for key, val in raw_dict.items():
        trans_dict[key] = modules.dictConvert(val)
    with open('/data-local/taejin/feat_dir/Fisher/fisher_trans_dict.pickle', 'wb') as handle:
        pickle.dump(trans_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(file_path):
    print('Loading Pickle File: ', file_path)
    st = time.time()
    try:
        with open(file_path, 'rb') as handle:
            b = pickle.load(handle)
        print('Loading complete. Elapsed time: %fs' %(time.time()-st)) 
    except:
        print('No such file as: ', file_path)
        raise ValueError
    return b

def savePickle(pickle_path, save_list):
    with open(pickle_path, 'wb') as handle:
        pickle.dump(save_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_txt(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
        f.close()
        assert content != [], "File is empty. Abort. Given path: " + list_path
        return content

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def unison_shuffled_copies_three(amat, bmat, slmat):
    ipdb.set_trace()
    assert len(amat) == len(bmat) and len(bmat) == len(slmat)
    pmat = np.random.permutation(len(amat))
    return amat[pmat], bmat[pmat], slmat[pmat]

def unison_numpy_shuffled(amat, bmat, slmat):
    # ipdb.set_trace()
    assert len(amat) == len(bmat) and len(bmat) == len(slmat)
    amat, bmat, slmat = shuffle(amat, bmat, slmat)
    return amat, bmat, slmat

def getGPUbatchSize(num_gpus, batch_size):
    nf = int( noremDiv(batch_size,num_gpus))
    nl = batch_size - nf*(num_gpus-1)
    return np.cumsum([0] + [nf]*(num_gpus-1) + [nl])

def write_txt(w_path, list_to_wr):
    with open(w_path, "w") as output:
        for k, val in enumerate(list_to_wr):
            output.write(val + '\n')
    return None

def nanCheck(np_mat):
    if np.isnan(np_mat).any():
        print('Number of nan: ', np.count_nonzero(np.isnan(np_mat)))
        raise ValueError('mean_act_out matrix contains NAN value.')

def segRead(fn, start, end):
    fo = open(fn, "r")
    line = fo.readlines()[start:end]
    fo.close()
    return line

def makeDict(content):
    feat_dict = {}
    for line in content:
        line = line.split(' ')
        feat_dict[line[0]] = [line[1], line[2], line[3]]
    return feat_dict

def readFeat(dkey, feat_dict, kaldi_feat_path):
    fn = kaldi_feat_path + '/' + feat_dict[dkey][0]
    start = int(feat_dict[dkey][1])
    end = int(feat_dict[dkey][2])
    segRead(fn, start, end)

def loadFisherFeatList(kaldi_feat_path):
    fisher_mfcc_abs_path = kaldi_feat_path + '/' + '*.txt'
    kaldi_mfcc_file_index = []
    # Read all the kaldi-generated feature files
    for file in glob.glob(fisher_mfcc_abs_path):
        print('###### ARK file open: ', file)
        kaldi_mfcc_file_index.append(file)
    return kaldi_mfcc_file_index

def ftm(name):
    '''
    Fisher speaker tag remover:
    (Trans dictionary is indexed per session, not speaker.)
    '''
    return name.replace('-A', '').replace('-B', '')



def kaldiFeatLoader(kaldi_mfcc_file_index_list, trans_dict):
    '''
    Using trans_dict, this generator function loads
    kaldi feature file per session sequentially.

    Args:
        kaldi_mfcc_file_index_list: Please include all the .txt path for features
        trans_dict: Please include all the dictionary for the training/test data.

    Returns:
        fileid
        mfcc numpy array (length x #ch)
        trans_dict: transcription in dictionary format ex( key format: "fe_03_01234"
    '''
    for list_path in kaldi_mfcc_file_index_list:
        print(list_path)
        with open(list_path) as f:
            mfcc_lines = []
            raw_mfcc_id = list_path.split('/')[-1]
            for i, line in enumerate(f):
                line = line.strip()
                if 'fe' in line:  # The first line
                    fileid = line.replace('[', '').strip()
                    print('Captured fileID: %s' %(fileid))
                    start_line_num = str(i+1)
                
                elif ']' in list(line):  # The last line -> output a set of training samples
                    line = line.replace(']', '')
                    print(line)
                    mfcc_lines.append([float(x) for x in line.strip().split(' ')])
                    end_line_num = str(i)
                    index_list = [fileid, raw_mfcc_id, start_line_num, end_line_num]
                    
                    yield [fileid, 
                           np.asarray(mfcc_lines), 
                           trans_dict[ftm(fileid)]]
                    mfcc_lines = []  # Empty the buffer list
                
                else:
                    mfcc_lines.append([float(x) for x in line.strip().split(' ')])


def seqLen(tD):
    '''Calculate the number of non-zero elements for variable length RNN
    '''
    # return np.expand_dims(np.count_nonzero(tD, axis=1), axis=1)
    # return list(np.count_nonzero(tD, axis=1))
    # SLout = np.count_nonzero(tD, axis=1) 
    SLout = np.count_nonzero(tD, axis=1) 
    return SLout


def fisherSpkCH(fileid):
    ''' Fisher Corpora Channel Mapper '''
    spk = fileid.strip()[-1]
    if spk == 'A':
        return 0
    elif spk == 'B':
        return 1

def makeTxtBash(file_name):
    if not os.path.isfile(file_name):
        bashGet('touch ' + file_name)
    else:
        bashGet('rm ' + file_name)
        bashGet('touch ' + file_name)

