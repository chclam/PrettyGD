from math import sqrt
from itertools import product, combinations
import numpy as np
import torch as tt
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange 
from sklearn.preprocessing import MinMaxScaler

import graph as G


class PrettyGD:
  def __init__(self, lr=0.001, weights=None):
    self.lr = lr
    self.losses = []
    self.weights = {
      "displacement": 1,
      "crossing_ang_res": 1,
      "angular_res": 1,
      "gabriel": 1,
      "vertex_res": 1
    }
    if weights is not None:
      self.weights.update(weights)

  def fit(self, V, adj_V):
    self.scaler = MinMaxScaler()
    V = self.scaler.fit_transform(V)
    self.W = tt.tensor(V.copy(), requires_grad=True) # new vertex coordinates 
    self.V = tt.tensor(V) # original vertex coordinates
    self.adj_V = adj_V
    self.edge_li = G.get_edge_li(V, adj_V) # precompute the edge list 

  def train(self, N=10):
    opt = Adam([self.W], lr=self.lr)
    sch = ExponentialLR(opt, gamma=0.9)
    self.weights = dict([(k, tt.tensor(self.weights[k])) for k in self.weights.keys()])
    for i in (t:= trange(N)):
      opt.zero_grad()
      loss = self.graph_loss()
      loss.backward()
      opt.step()
      sch.step()
      self.losses.append(loss.item())
      t.set_description(f"Loss: {loss:.6f}; Progress")

  def get_graph_coords(self):
    ret = self.W.detach().numpy()
    ret = self.scaler.inverse_transform(ret)
    return ret

  def graph_loss(self):
    loss_disp = self.loss_displacement()
    loss_disp = tt.multiply(loss_disp, self.weights['displacement'])

    loss_cross = self.loss_crossings()
    loss_cross = tt.multiply(loss_cross, self.weights['crossing_ang_res'])

    loss_ang_res = self.loss_angular_res()
    loss_ang_res = tt.multiply(loss_ang_res, self.weights['angular_res'])

    loss_gabriel = self.loss_gabriel()
    loss_gabriel = tt.multiply(loss_gabriel, self.weights['gabriel'])

    loss_vert_res = self.loss_vert_res()
    loss_vert_res = tt.multiply(loss_vert_res, self.weights['vertex_res'])

    ret = (loss_disp, loss_cross, loss_ang_res, loss_gabriel, loss_vert_res)
    ret = tt.stack(ret)
    ret = tt.sum(ret)
    return ret
  
  def loss_displacement(self):
    # squared euclidean distance
    ret = tt.subtract(self.W, self.V)
    ret = tt.pow(ret, 2)
    ret = tt.sum(ret, axis=1)
    ret = tt.sum(ret)
    return ret

  def loss_angular_res(self):
    SENS = 1 # sensitivity of angular energy 
    ang = G.get_angles(self.W, self.adj_V)
    # calculate loss of the calculated angles
    SENS = tt.tensor(SENS)
    SENS = tt.multiply(tt.tensor(-1), SENS)
    ret = tt.multiply(SENS, ang)
    ret = tt.exp(ret)
    ret = tt.sum(ret)
    return ret

  def loss_vert_res(self):
    r = 1 / sqrt(len(self.W))
    W_idx = np.arange(0, len(self.W))
    WW = list(combinations(W_idx, 2)) 
    WW = np.asarray(WW)
    W_i = G.to_vert_vals(WW[:,0], self.W)
    W_j = G.to_vert_vals(WW[:,1], self.W)
    # euclidean distance
    ret = tt.subtract(W_i, W_j)
    ret = tt.square(ret)
    ret = tt.sum(ret, axis=1)
    ret = tt.sqrt(ret)
    d_max = tt.max(ret)
    ret = tt.divide(ret, r * d_max)
    # the rest with relu etc.
    ret = tt.subtract(tt.ones(len(ret)), ret)
    ret = F.relu(ret)
    ret = tt.square(ret)
    ret = tt.sum(ret)
    return ret 

  def loss_crossings(self):
    ints = G.get_intersects(self.W, self.edge_li)
    p_idx = []
    q_idx = []
    r_idx = []
    s_idx = []
    for it in ints:
      p_idx.append(it['e1'][0])
      q_idx.append(it['e1'][1])
      r_idx.append(it['e2'][0])
      s_idx.append(it['e2'][1])
    # gather the vectorized vertex positions from indices
    p = G.to_vert_vals(p_idx, self.W)
    q = G.to_vert_vals(q_idx, self.W)
    r = G.to_vert_vals(r_idx, self.W)
    s = G.to_vert_vals(s_idx, self.W)
    # calculate vectorized crossing angle
    e1 = tt.subtract(p, q)
    e2 = tt.subtract(r, s)
    ret = F.cosine_similarity(e1, e2) 
    ret = tt.pow(ret, 2)
    ret = tt.sum(ret)
    return ret

  def loss_gabriel(self):
    edges = np.array(self.edge_li)

    edge_inds = np.arange(0, len(edges))
    W_inds = np.arange(0, len(self.W))
    EW_inds = np.asarray(list(product(edge_inds, W_inds)))
    IJ = np.take(edges, EW_inds[:,0], axis=0)
    K = EW_inds[:,1]
    # remove instances where i = k or j = k, i.e. where vert k == endpoints i or j 
    same_ijk = [] 
    same_ijk.extend(np.argwhere(np.subtract(IJ[:,0], K) == 0))
    same_ijk.extend(np.argwhere(np.subtract(IJ[:,1], K) == 0))
    IJ = np.delete(IJ, same_ijk, axis=0)
    K = np.delete(K, same_ijk, axis=0)
   
    X_i = G.to_vert_vals(IJ[:,0], self.W)
    X_j = G.to_vert_vals(IJ[:,1], self.W)
    X_k = G.to_vert_vals(K, self.W)
    
    r_ij = tt.subtract(X_i, X_j)
    r_ij = tt.abs(r_ij)
    r_ij = tt.divide(r_ij, 2)

    c_ij = tt.add(X_i, X_j)
    c_ij = tt.divide(c_ij, 2)

    ret = tt.subtract(X_k, c_ij)
    ret = tt.abs(ret)
    ret = tt.subtract(r_ij, ret)
    ret = F.relu(ret)
    ret = tt.pow(ret, 2)
    ret = tt.sum(ret, axis=1)
    # add x and y forces
    ret = tt.sum(ret, axis=0)
    return ret
  
