
from __future__ import division
import sys
sys.path.append("../")
sys.path.append("./sc_utils")

import argparse
import copy

import numpy as np
import modules
import kaldi_io


from sklearn.utils import check_random_state
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.validation import check_array
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering as sklearn_SpectralClustering
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import MinMaxScaler

import scipy 
import scipy.sparse as sparse
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

import warnings

scaler = MinMaxScaler(feature_range=(0, 1))

class SparseSpectralClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, eigen_solver=None, random_state=None,
                 n_init=10, gamma=1., affinity='rbf', p_neighbors=10,
                 eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, n_jobs=None):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.p_neighbors = p_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            raise ValueError('affinity_matrix is not specified.')
        
        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels)
        return self

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"


def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans'):
    if assign_labels not in ('kmeans', 'discretize'):
        raise ValueError("The 'assign_labels' parameter should be "
                         "'kmeans' or 'discretize', but '%s' was given"
                         % assign_labels)

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    maps = spectral_embedding(affinity, n_components=n_components,
                              eigen_solver=eigen_solver,
                              random_state=random_state,
                              eigen_tol=eigen_tol, drop_first=False)

    if assign_labels == 'kmeans':
        kmeans = KMeans(n_clusters, random_state=random_state,n_init=n_init).fit(maps)
        labels = kmeans.labels_
    else:
        labels = discretize(maps, random_state=random_state)

    return labels


def spectral_embedding(adjacency, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0,
                       norm_laplacian=True, drop_first=True):
    adjacency = check_symmetric(adjacency)

    eigen_solver = 'arpack'
    norm_laplacian=False
    random_state = check_random_state(random_state)
    n_nodes = adjacency.shape[0]
    if not _graph_is_connected(adjacency):
        warnings.warn("Graph is not fully connected, spectral embedding"
                      " may not work as expected.")
    laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
                                      return_diag=True)
    if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and
       (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)):
        # print("[INFILE] eigen_solver : ", eigen_solver, "norm_laplacian:", norm_laplacian)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        try:
            laplacian *= -1
            v0 = random_state.uniform(-1, 1, laplacian.shape[0])
            lambdas, diffusion_map = eigsh(laplacian, k=n_components,
                                           sigma=1.0, which='LM',
                                           tol=eigen_tol, v0=v0)
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            eigen_solver = "lobpcg"
            laplacian *= -1

    embedding = _deterministic_vector_sign_flip(embedding)
    return embedding[:n_components].T


def _set_diag(laplacian, value, norm_laplacian):
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = value
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            laplacian = laplacian.todia()
        else:
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian

def get_kneighbors_conn(X_dist, p_neighbors):
    X_dist_out = np.zeros_like(X_dist)
    for i, line in enumerate(X_dist):
        sorted_idx = np.argsort(line)
        sorted_idx = sorted_idx[::-1]
        indices = sorted_idx[:p_neighbors]
        X_dist_out[indices, i] = 1
    return X_dist_out

def getLaplacian(X):
    X[np.diag_indices(X.shape[0])]=0
    A = X
    D = np.sum(np.abs(A), axis=1)
    D = np.diag(D)
    L = D - A
    return L
   
def eig_decompose(L, k):
    lambdas, eig_vecs = scipy.linalg.eigh(L)
    # lambdas, eig_vecs = scipy.sparse.linalg.eigsh(L)
    return lambdas, eig_vecs

def getLamdaGaplist(lambdas):
    lambda_gap_list = []
    for i in range(len(lambdas)-1):
        lambda_gap_list.append(float(lambdas[i+1])-float(lambdas[i]))
    return lambda_gap_list

def estimate_num_of_spkrs(X_conn, SPK_MAX):
    L  = getLaplacian(X_conn)
    lambdas, eig_vals = eig_decompose(L, k=X_conn.shape[0])
    lambdas = np.sort(lambdas)
    lambda_gap_list = getLamdaGaplist(lambdas)
    num_of_spk = np.argmax(lambda_gap_list[:min(SPK_MAX,len(lambda_gap_list))]) + 1
    return num_of_spk, lambdas, lambda_gap_list


def kaldi_style_lable_writer(seg_lable_list, write_path):
    with open(write_path, 'w') as the_file:                                       
        for tup in seg_lable_list:
            line = tup[0] + ' ' + str(tup[1]) + ' \n'
            the_file.write(line)   

def nps(str_num):
    int_num = int(str_num)
    float_num = float(int_num/100.00)
    return round(float_num, 2)

def read_embd_seg_info(param):

    open(param.embedding_scp)
    embd_seg_dict = {}
     
    # for embd_sess_line in spk_embed_sess_list:
    for embd_sess_line, val in kaldi_io.read_vec_flt_scp(param.embedding_scp):

        seg_id = embd_sess_line
        split_seg_info = seg_id.split('-')
        sess_id = split_seg_info[0]
        try:
            if len(split_seg_info) == 5:
                offset = nps(split_seg_info[1])
                start, end = round(offset + nps(split_seg_info[3]), 2),  round(offset + nps(split_seg_info[4]), 2)
            elif len(split_seg_info) == 3: 
                # start, end = round(offset + nps(split_seg_info[1]), 2),  round(offset + nps(split_seg_info[2]), 2)
                pass
            else:
                raise ValueError("Incorrect segments file format (segment id is wrong) ")
        except:
            raise ValueError("Incorrect segments file format.")
        
        if sess_id not in embd_seg_dict:
            embd_seg_dict[sess_id] = [(start, end)]
        else:
            embd_seg_dict[sess_id].append((start, end))

    return embd_seg_dict

def _graph_is_connected(graph):
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _graph_connected_component(graph, node_id):
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore = np.zeros(n_node, dtype=np.bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def get_X_conn_from_dist(X_dist_raw, p_neighbors):
    # p_neighbors = int(X_dist_raw.shape[0] * threshold)
    X_r = get_kneighbors_conn(X_dist_raw, p_neighbors) 
    X_conn_from_dist= 0.5 * (X_r + X_r.T)
    return X_conn_from_dist
    

def isFullyConnected(X_conn_from_dist):
    gC = _graph_connected_component(X_conn_from_dist, 0).sum() == X_conn_from_dist.shape[0]
    return gC

def gc_thres_min_gc(mat, max_n, n_list):
    p_neighbors, index = 1, 0
    X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors) 
    fully_connected = isFullyConnected(X_conn_from_dist)
    for i, p_neighbors in enumerate(n_list):
        fully_connected = isFullyConnected(X_conn_from_dist)
        X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors) 
        if fully_connected or p_neighbors > max_n:
            if p_neighbors > max_n and not fully_connected:
                print("Still not fully conneceted but exceeded max_N")
            print("---- Increased thres gc p_neighbors:",p_neighbors,
                  "/",X_conn_from_dist.shape[0],
                  "fully_connected:", fully_connected, 
                  "ratio:", round(float(p_neighbors/X_conn_from_dist.shape[0]), 5))
            break

    return X_conn_from_dist, p_neighbors

def scp2dict(path):
    t_list = open(path)
    out_dict = {}
    for line in t_list:
        key = line.strip().split()[0]
        val = line.strip().split()[1]
        if key not in out_dict:
            out_dict[key] = val
    return out_dict

def checkOutput(key, seg_list, Yk):
    if len(seg_list) != Yk.shape[0]:
        print(idx+1, "Segments file length mismatch -key:", key, len(seg_list), Yk.shape[0])
        raise ValueError("Mismatch of lengths")
        return None 

def getSegmentDict(param):
    seg_dict, segments_total_dict = {}, {}

    line_generator_segment = open(param.segment_file_input_path)

    for line in line_generator_segment:
        seg_id = line.strip().split()[0]
        sess_id = line.strip().split()[1]
        if '-rec' in sess_id:
            sess_id = sess_id.replace('-rec', '')

        if sess_id in seg_dict:
            seg_dict[sess_id].append(seg_id)
            segments_total_dict[sess_id].append(line.strip())
        elif sess_id not in seg_dict:
            seg_dict[sess_id] = [seg_id]
            segments_total_dict[sess_id] = [line.strip()]
    return seg_dict, segments_total_dict


class GraphSpectralClusteringClass(object):
    def __init__(self, param):
        self.param = param
        if "." in self.param.threshold:
            self.param.threshold = float(self.param.threshold)

        if self.param.max_speaker_list != 'None':
            print("Loading max_speaker_list file: ", self.param.max_speaker_list)
            self.maxspk_dict = scp2dict(self.param.max_speaker_list)

        if self.param.reco2num_spk != 'None':
            print("Loading reco2num_spk file: ", self.param.reco2num_spk)
            self.reco2num_dict = scp2dict(self.param.reco2num_spk)

        if self.param.embedding_scp != 'None':
            print("Loading Embedding files for time stamps...")
            self.embd_range_dict = read_embd_seg_info(self.param)

        if self.param.asr_spk_turn_est_scp != 'None':
            self.lex_range_dict = read_turn_est_v0(self.param)

        self.labels_out_list = []
        self.seg_dict, self.segments_total_dict = getSegmentDict(self.param)

        self.est_num_spks_out_list = []
        self.lambdas_list = []

        self.use_gc_thres=False
   
    def npy_to_generator(self):
        base_path = self.param.distance_score_file.replace('scores.txt', '')
        cont = modules.read_txt(param.distance_score_file)
        for key in cont:
            mat = np.load(base_path+'/'+ key +'.npy')
            yield key, mat
        

    def prepData(self):
        if self.param.distance_score_file.split('.')[-1] == "scp":
            print("=== [INFO] .scp file and .ark files were provided")
            self.key_mat_generator_dist_score = list(kaldi_io.read_mat_scp(self.param.distance_score_file))
        elif self.param.distance_score_file.split('.')[-1] == "txt":
            print("=== [INFO] .txt file and .npy files were provided")
            self.key_mat_generator_dist_score = self.npy_to_generator()

        if self.param.spt_est_thres in ["EigRatio", "NMESC"]:
            pass
        elif self.param.spt_est_thres != "None":
            cont_spt_est_thres = modules.read_txt(self.param.spt_est_thres)
            self.spt_est_thres_dict = { x.split()[0]:float(x.split()[1]) for x in cont_spt_est_thres }

    def performClustering(self):
        for idx, (key, mat) in enumerate(self.key_mat_generator_dist_score):
            
            if self.param.max_speaker_list != "None":
                self.param.max_speaker = int(maxspk_dict[key])

            if 'plda' in self.param.score_metric:
                # modules.cprint("Using PLDA score thresholding mode.",'y')
                Y = self.PLDAclustering(idx, key, mat, self.param)
            elif 'cos' in self.param.score_metric:
                Y = self.COSclustering(idx, key, mat, self.param)
            else:
                raise ValueError('self.param.score_metric contains invalid score metric:', self.param.score_metric)
           
            # print("score metric: ", self.param.score_metric)
            Yk = Y + 1 # Index shift for kaldi index
            self.seg_list = self.seg_dict[key]
            checkOutput(key, self.seg_list, Yk)
            self.labels_out_list.extend(zip(self.seg_list, Yk))
            self.getOutputPaths(self.param, self.labels_out_list)

        modules.write_txt(self.lambdas_out_path, self.lambdas_list)
        modules.cprint('Method: Spectral Clustering has been finished ', 'y')

    def getOutputPaths(self, param, labels_out_list):
        self.est_num_of_spk_out_path = '/'.join(self.param.spk_labels_out_path.split('/')[:-1]) + '/spt_reco2num_spks'
        self.lambdas_out_path = '/'.join(self.param.spk_labels_out_path.split('/')[:-1]) + '/lambdas'
        
        kaldi_style_lable_writer(self.labels_out_list, self.param.spk_labels_out_path)
        kaldi_style_lable_writer(self.est_num_spks_out_list, self.est_num_of_spk_out_path)
    
    def NMEanalysis(self, mat, SPK_MAX, max_rp_threshold, sparse_search=True, search_p_volume=20):
        eps = 1e-10
        eig_ratio_list = []
        
        max_N = int(mat.shape[0] * max_rp_threshold)
        if sparse_search:
            N = min(max_N, search_p_volume)
            p_neighbors_list = list(np.linspace(1, max_N, N, endpoint=True).astype(int))
        else:
            p_neighbors_list = list(range(1, max_N))
        print("Scanning eig_ratio of length [{}] mat size [{}] ...".format(len(p_neighbors_list), mat.shape[0]))
        
        est_spk_n_dict = {}
        for p_neighbors in p_neighbors_list:
            X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors)
            est_num_of_spk, lambdas, lambda_gap_list = estimate_num_of_spkrs(X_conn_from_dist, SPK_MAX)
            est_spk_n_dict[p_neighbors] = (est_num_of_spk, lambdas)
            arg_sorted_idx = np.argsort(lambda_gap_list[:SPK_MAX])[::-1] 
            max_key = arg_sorted_idx[0]  
            max_eig_gap = lambda_gap_list[max_key]/(max(lambdas) + eps) 
            eig_ratio_value = (p_neighbors/mat.shape[0])/(max_eig_gap+eps)
            eig_ratio_list.append(eig_ratio_value)
         
        index_nn = np.argmin(eig_ratio_list)
        rp_p_neighbors = p_neighbors_list[index_nn]
        X_conn_from_dist = get_X_conn_from_dist(mat, rp_p_neighbors)
        if not isFullyConnected(X_conn_from_dist):
            X_conn_from_dist, rp_p_neighbors = gc_thres_min_gc(mat, max_N, p_neighbors_list)
        return X_conn_from_dist, float(rp_p_neighbors/mat.shape[0]), est_spk_n_dict[rp_p_neighbors][0], est_spk_n_dict[rp_p_neighbors][1]
    
    @staticmethod 
    def print_status_estNspk(idx, key, mat, threshold, est_num_of_spk, param):
        print(idx+1, " score_metric:", param.score_metric, 
                     " affinity matrix pruning - threshold: {:3.3f}".format(threshold),
                     " key:", key,"Est # spk: " + str(est_num_of_spk), 
                     " Max # spk:", param.max_speaker, 
                     " MAT size : ", mat.shape)
    @staticmethod 
    def print_status_givenNspk(idx, key, mat, rp_threshold, est_num_of_spk, param):
        print(idx+1, " score_metric:", param.score_metric,
                     " Rank based pruning - RP threshold: {:4.4f}".format(rp_threshold), 
                     " key:", key,
                     " Given Number of Speakers (reco2num_spk): " + str(est_num_of_spk), 
                     " MAT size : ", mat.shape)

    def COSclustering(self, idx, key, mat, param):
        X_dist_raw = mat
        rp_threshold = param.threshold
        if param.spt_est_thres in ["EigRatio", "NMESC"]:
            # param.sparse_search = False
            X_conn_from_dist, rp_threshold, est_num_of_spk, lambdas = self.NMEanalysis(mat, param.max_speaker, max_rp_threshold=0.250, sparse_search=param.sparse_search)

        
        elif param.spt_est_thres != 'None':
            if key == "iaeu":
                rp_threshold = 0.081
            else:
                rp_threshold = self.spt_est_thres_dict[key]
    
            p_neighbors = int(mat.shape[0] * rp_threshold)
            X_conn_from_dist = get_X_conn_from_dist(X_dist_raw, p_neighbors)
        
        elif self.use_gc_thres:
            p_neighbors = int(mat.shape[0] * param.threshold)
            X_conn_from_dist = get_X_conn_from_dist(mat, p_neighbors)
        
        else:
            ### If score metric is not PLDA, threshold is used for similarity ranking pruning.
            p_neighbors = int(mat.shape[0] * param.threshold)
            X_r = get_kneighbors_conn(X_dist_raw, p_neighbors) 
            X_conn_from_dist= 0.5 * (X_r + X_r.T)
       

        #################################################
        ### Use ASR result from turn probability file ###
        #################################################
        if param.asr_spk_turn_est_scp != 'None':
            '''
            Use ASR transcript to estimate the turn probabilites.
            '''
            assign_thr = float(param.xvector_window/2.0)
            X_conn_from_dist = add_turn_est_prob(X_conn_from_dist, 
                                                 embd_range_dict[key], 
                                                 lex_range_dict[key], 
                                                 assign_thr)


        '''
        Determine the number of speakers.
        if param.reco2num_spk contains speaker number info, we use that.
        Otherwise we estimate the number of speakers using estimate_num_of_spkrs()

        '''
        if param.reco2num_spk != 'None': 
            est_num_of_spk = int(self.reco2num_dict[key])
            ### Use the given number of speakers
            est_num_of_spk = min(est_num_of_spk, param.max_speaker) 
            _, lambdas, lambda_gap_list = estimate_num_of_spkrs(X_conn_from_dist, param.max_speaker)
            self.print_status_givenNspk(idx, key, mat, rp_threshold, est_num_of_spk, param)

        else: 
            ### Estimate the number of speakers in the given session
            self.print_status_estNspk(idx, key, mat, rp_threshold, est_num_of_spk, param)

        
        lambdas_str = ' '.join([ str(x) for x in lambdas ] )
        self.lambdas_list.append(key + " " + lambdas_str)
        self.est_num_spks_out_list.append( [key, str(est_num_of_spk)] ) 
        
        ### Handle the sklearn/numpy bug of eigenvalue parameter.
        spectral_model = SparseSpectralClustering(affinity='precomputed', 
                                                n_jobs=-2, 
                                                n_clusters=est_num_of_spk,
                                                eigen_tol=1e-10)

        Y = spectral_model.fit_predict(X_conn_from_dist)
        return Y
    

    def PLDAclustering(self, idx, mat, param):
        scaler.fit(mat) 
        X_dist_raw = mat

        X_r = get_kneighbors_conn_thres(X_dist_raw, param.threshold)
        X_conn_from_dist= 0.5 * (X_r + X_r.T)

        if param.reco2num_spk != 'None': 
            ### Use the given number of speakers
            est_num_of_spk = int(self.reco2num_dict[key])
            self.print_status_givenNspk(self, idx, key, mat, est_num_of_spk, param)

        else: 
            ### Estimate the number of speakers in the given session
            est_num_of_spk, lambdas, lambda_gap_list = estimate_num_of_spkrs(X_conn_from_dist, param.max_speaker)
            self.print_status_estNspk(self, idx, key, mat, est_num_of_spk, param)
        
        ### Handle the sklearn/numpy bug of eigenvalue parameter.
        spectral_model = SparseSpectralClustering(affinity='precomputed', 
                                                n_jobs=-2, 
                                                n_clusters=est_num_of_spk,
                                                eigen_tol=1e-10)

        Y = spectral_model.fit_predict(X_conn_from_dist)
        return Y
    


parser = argparse.ArgumentParser()
parser.add_argument('--distance_score_file', action='store', type=str, help='Path for distance score scp')
parser.add_argument('--asr_spk_turn_est_scp', action='store', type=str, help='Path for scp file with ctm list', default='None')
parser.add_argument('--embedding_scp', action='store', type=str, help='Path for scp file embedding segment info', default='None')
parser.add_argument('--threshold', action='store', type=str, help='Threshold ratio of distance pruning')
parser.add_argument('--segment_file_input_path', action='store', type=str, help='Path for segment file')
parser.add_argument('--spk_labels_out_path', action='store', type=str, help='Path for output speaker labels')
parser.add_argument('--reco2num_spk', action='store', type=str, default='None')
parser.add_argument('--score-metric', action='store', type=str, default='cos')
parser.add_argument('--max_speaker', action='store', type=int, default=8)
parser.add_argument('--xvector_window', action='store', type=float, default=1.5)
parser.add_argument('--spt_est_thres', action='store', type=str)
parser.add_argument('--max_speaker_list', action='store', type=str)
parser.add_argument('--sparse_search', action='store', type=str, default=True)

param = parser.parse_args()

SC = GraphSpectralClusteringClass(param)
SC.prepData()
SC.performClustering()

