import torch
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange 
from math import pi
import numpy as np
from sortedcontainers import SortedList, SortedKeyList
from shapely.geometry import LineString
from itertools import product, combinations

import geometry as geo


'''
TO DO: 
  * Vectorize get_intersects() as long as line sweep is not implemented.

'''

def train(V, adj_V, N=5, lr=0.001, w_disp=20, w_cross=0.1, w_ang_res=0.2, w_gabriel=0.1):
  W = V.copy()
  W = torch.tensor(W, requires_grad=True)
  V = torch.tensor(V)
  opt = Adam([W], lr=lr)
  sch = ExponentialLR(opt, gamma=0.9)
  losses = []
  for i in (t:= trange(N)):
    opt.zero_grad()
    loss = loss_function(W, V, adj_V, w_disp, w_cross, w_ang_res, w_gabriel)
    loss.backward()
    opt.step()
    sch.step()
    loss = loss.item()
    losses.append(loss)
    t.set_description(f"Loss: {loss:.6f}; Progress")
  return W, losses

def loss_function(W, V, adj_V, w_disp, w_cross, w_ang_res, w_gabriel):
  # weights to tensor
  w_disp = torch.tensor(w_disp)
  w_cross = torch.tensor(w_cross)
  w_ang_res = torch.tensor(w_ang_res)

  if w_disp == 0:
    loss_disp = torch.tensor(0)
  else:
    loss_disp = loss_displacement(W, V, adj_V)
    loss_disp = torch.multiply(loss_disp, w_disp)

  if w_cross == 0:
    loss_cross = torch.tensor(0)
  else:
    loss_cross = loss_crossings(W, adj_V)
    loss_cross = torch.multiply(loss_cross, w_cross)

  if w_ang_res == 0:
    loss_ang_res = torch.tensor(0)
  else:
    loss_ang_res = loss_angular_res(W, adj_V)
    loss_ang_res = torch.multiply(loss_ang_res, w_ang_res)

  if w_gabriel == 0:
    loss_gabriel = torch.tensor(0)
  else:
    loss_gabriel = loss_angular_res(W, adj_V)
    loss_gabriel = torch.multiply(loss_gabriel, w_gabriel)

  ret = (loss_disp, loss_cross, loss_ang_res, loss_gabriel)
  ret = torch.stack(ret)
  ret = torch.sum(ret)
  return ret
  
def loss_gabriel(W, adj_V):
  edges = get_edgeset(W, adj_V)
  edges = np.array(edges)

  edge_inds = np.arange(0, len(edges))
  W_inds = np.arange(0, len(W))
  EW_inds = np.asarray(list(product(edge_inds, W_inds)))
  IJ = np.take(edges, EW_inds[:,0], axis=0)
  K = EW_inds[:,1]
  # remove instances where i = k or j = k, i.e. where vert k == endpoints i or j 
  same_ijk = [] 
  same_ijk.extend(np.argwhere(np.subtract(IJ[:,0], K) == 0))
  same_ijk.extend(np.argwhere(np.subtract(IJ[:,1], K) == 0))
  IJ = np.delete(IJ, same_ijk, axis=0)
  K = np.delete(K, same_ijk, axis=0)
 
  X_i = to_vert_vals(IJ[:,0], W)
  X_j = to_vert_vals(IJ[:,1], W)
  X_k = to_vert_vals(K, W)
  
  r_ij = torch.subtract(X_i, X_j)
  r_ij = torch.abs(r_ij)
  r_ij = torch.divide(r_ij, 2)

  c_ij = torch.add(X_i, X_j)
  c_ij = torch.divide(c_ij, 2)

  ret = torch.subtract(X_k, c_ij)
  ret = torch.abs(ret)
  ret = torch.subtract(r_ij, ret)
  ret = F.relu(ret)
  ret = torch.pow(ret, 2)
  ret = torch.sum(ret, axis=1)
  # add x and y forces
  ret = torch.sum(ret, axis=0)
  return ret
  
def get_edgeset(V, adj_V):
  if type(V) is torch.Tensor:
    V = V.detach().numpy()
  ret = []
  for i in range(len(V)):
    for j in adj_V[i]:
      if (j, i) not in ret:
        ret.append((i, j))
  return ret

def loss_displacement(W, V, adj_V):
  norm = get_avg_edge_length(W, adj_V)
  # squared euclidean distance
  ret = torch.subtract(W, V)
  ret = torch.pow(ret, 2)
  ret = torch.sum(ret, axis=1)
  ret = torch.sum(ret)
  # use average edge length as distance normalizer on the map
  ret = torch.divide(ret, norm)
  return ret

def get_avg_edge_length(W, adj_V, stoch=True):
  # calculate the average edge length stochastically
  samp_inds = np.random.randint(0, len(adj_V), size=max(1, len(adj_V) // 10))
  edge_samp_i = []
  edge_samp_j = []
  for i in samp_inds:
    if len(adj_V[i]) == 0:
      continue
    nbr_samp = np.random.randint(0, len(adj_V[i]), max(1, len(adj_V[i]) // 5))
    for j in nbr_samp:
      edge_samp_i.append(i)
      edge_samp_j.append(adj_V[i][j])

  p = to_vert_vals(edge_samp_i, W)
  q = to_vert_vals(edge_samp_j, W)
  
  # average edge length as normalizer
  norm = torch.subtract(p, q)
  norm = torch.pow(norm, 2)
  norm = torch.sum(norm, axis=1)
  norm = torch.sqrt(norm)
  norm = norm.mean()
  return norm

def loss_crossings(W, adj_V):
  ints = get_intersects(W, adj_V)
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
  p = to_vert_vals(p_idx, W)
  q = to_vert_vals(q_idx, W)
  r = to_vert_vals(r_idx, W)
  s = to_vert_vals(s_idx, W)
  # calculate vectorized crossing angle
  e1 = torch.subtract(p, q)
  e2 = torch.subtract(r, s)
  ret = F.cosine_similarity(e1, e2) 
  ret = torch.pow(ret, 2)
  ret = torch.sum(ret)
  return ret

def is_lower_endpoint(p_idx, adj_V, status):
  for p_nbr in adj_V[p_idx]:
    if p_nbr not in status[:,0]:
      return False
  return True 

def loss_angular_res(W, adj_V):
  # get indices for incident edges for vectorization
  sens = 1 # sensitivity of angular energy 
  i_idx = []
  j_idx = []
  k_idx = []
  for i, v in enumerate(W):
    if len(adj_V[i]) < 2:
      continue
    nbr_order = get_nbr_order(v, W, adj_V[i])
    for a in range(1, len(nbr_order)):
      b = (a + 1) % len(nbr_order) # make sure to also get the angle formed by last and first nbr in "nbr_order"
      j = nbr_order[a]
      k = nbr_order[b]
      i_idx.append(i)
      j_idx.append(j)
      k_idx.append(k)
  # gather the vertex values for vectorized angle calculations
  i = to_vert_vals(i_idx, W)
  j = to_vert_vals(j_idx, W)
  k = to_vert_vals(k_idx, W)
  # calculate angles between vectorized edges ji and ki
  ji = torch.subtract(j, i)
  ki = torch.subtract(k, i)
  # dot product manually for t since torch.dot() doesn't support on 2d tensors
  t = torch.multiply(ji, ki)
  t = torch.sum(t, axis=1)
  b = torch.multiply(torch.norm(ji, dim=1), torch.norm(ki, dim=1))
  ang = torch.divide(t, b)
  ang = torch.arccos(ang)
  # calculate loss of the calculated angles
  sens = torch.tensor(sens)
  sens = torch.multiply(torch.tensor(-1), sens)
  ret = torch.multiply(sens, ang)
  ret = torch.exp(ret)
  ret = torch.sum(ret)
  return ret

def to_vert_vals(vert_idx, W):
  '''
  Given a list of vertex indices for W, gather the vertex values i.e. x,y-values.
  '''
  vert_idx = torch.tensor(vert_idx)
  vert_idx = torch.unsqueeze(vert_idx, dim=1)
  vert_idx = vert_idx.expand(-1, 2)
  vert_vals = torch.gather(W, 0, vert_idx)
  return vert_vals

def get_nbr_order(v, V, adj_v):
  '''
  return the circular order of the neighbors of v starting
  from edge vu, where u is the neighbor indexed first in adj_v.
  '''
  u = V[adj_v[0]]
  vu = torch.subtract(u, v)
  angs = [0] # initialize as first in the circular order
  for i in range(1, len(adj_v)):
    w = V[adj_v[i]]
    vw = torch.subtract(w, v)
    # get angle between edges vu, vw
    ang = torch.arccos(torch.dot(vu, vw) / (torch.norm(vu) * torch.norm(vw)))
    if (vw[0] * vu[1] - vu[0] * vw[1]) <= 0:
      # if vu is the right vector of the angle,
      # calculate the outer angle
      ang = (2 * pi) - ang
    angs.append(ang)
  angs = torch.tensor(angs)
  ret = torch.argsort(angs)
  ret = [adj_v[idx] for idx in ret]
  return ret

def get_intersects(V, adj_V):
  '''
  Returns vertex indices of intersecting edges and
  their intersection point.
  Naive (but vectorized!!!!) O(n^2) check on intersection,
  TODO: change to line sweep when necessary.
  '''
  if type(V) is torch.Tensor:
    V = V.detach().numpy()
  # get list with vertex pairs forming edges
  E = []
  for v in range(len(V)):
    for w in adj_V[v]:
      if [w, v] in E:
        continue
      E.append([v, w])
  # calculate intersections
  E_inds = np.arange(0, len(E))
  EE_inds = np.asarray(list(combinations(E_inds, 2)))
  # get rid of edges that share endpoints
  PQ_inds = np.take(E, EE_inds[:,0], axis=0)
  RS_inds = np.take(E, EE_inds[:,1], axis=0)
  # endpoints p == q <==> p_idx - q_idx == 0
  same_endpoints = np.argwhere(
    (PQ_inds[:,0] - RS_inds[:,0] == 0) |
    (PQ_inds[:,1] - RS_inds[:,1] == 0) |
    (PQ_inds[:,0] - RS_inds[:,1] == 0) |
    (PQ_inds[:,1] - RS_inds[:,0] == 0)
  )
  PQ_inds = np.delete(PQ_inds, same_endpoints, axis=0)
  RS_inds = np.delete(RS_inds, same_endpoints, axis=0)
  # Get x,y values of P,Q,R,S
  P = np.take(V, PQ_inds[:,0], axis=0)
  Q = np.take(V, PQ_inds[:,1], axis=0)
  R = np.take(V, RS_inds[:,0], axis=0)
  S = np.take(V, RS_inds[:,1], axis=0)

  s1_x = np.subtract(Q[:,0], P[:,0])
  s1_y = np.subtract(Q[:,1], P[:,1])
  s2_x = np.subtract(S[:,0], R[:,0])
  s2_y = np.subtract(S[:,1], R[:,1])

  b =  -s2_x * s1_y + s1_x * s2_y
  s = (-s1_y * (P[:,0] - R[:,0]) + s1_x * (P[:,1] - R[:,1])) / b
  t = (s2_x * (P[:,1] - R[:,1]) - s2_y * (P[:,0] - R[:,0])) / b

  int_inds = np.argwhere((s >= 0) & (s <= 1) & (t >= 0) & (t <= 1))
  
  ints_x = np.add(P[:,0], np.multiply(t, s1_x))
  ints_y = np.add(P[:,1], np.multiply(t, s1_y))
  ints_x = np.squeeze(np.take(ints_x, int_inds, axis=0))
  ints_y = np.squeeze(np.take(ints_y, int_inds, axis=0))
  PQ_ints = np.squeeze(np.take(PQ_inds, int_inds, axis=0))
  RS_ints = np.squeeze(np.take(RS_inds, int_inds, axis=0))

  ret = []
  for i in range(len(ints_x)):
    it = dict(
      e1=PQ_ints[i], 
      e2=RS_ints[i], 
      intersection=[ints_x[i], ints_y[i]]
    )
    ret.append(it)
  return ret

