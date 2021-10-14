import torch
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange 
from math import pi
import numpy as np
from sortedcontainers import SortedList, SortedKeyList
from shapely.geometry import LineString
from sympy.geometry import Segment
from itertools import combinations


def train(V, adj_V, N=5, lr=0.001, w_disp=20, w_cross=0.1, w_ang_res=0.2):
  W = V.copy()
  W = torch.tensor(W, requires_grad=True)
  V = torch.tensor(V)
  opt = Adam([W], lr=lr)
  sch = ExponentialLR(opt, gamma=0.9)
  losses = []
  for i in (t:= trange(N)):
    opt.zero_grad()
    loss = loss_function(W, V, adj_V, w_disp, w_cross, w_ang_res)
    loss.backward()
    opt.step()
    sch.step()
    loss = loss.item()
    losses.append(loss)
    t.set_description(f"Loss: {loss:.6f}; Progress")
  return W, losses

def loss_function(W, V, adj_V, w_disp, w_cross, w_ang_res):
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

  ret = torch.add(loss_disp, loss_cross)
  ret = torch.add(ret, loss_ang_res)
  return ret
  
def loss_displacement(W, V, adj_V):
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
  # squared euclidean distance
  ret = torch.subtract(W, V)
  ret = torch.pow(ret, 2)
  ret = torch.sum(ret, axis=1)
  ret = torch.sum(ret)
  ret = torch.divide(ret, norm)
  return ret

def loss_crossings(W, adj_V):
  ints = get_intersects(W.detach().numpy(), adj_V)
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
  
def get_edgeset(W, adj_V):
  p_inds = []
  q_inds = []
  for i, p in enumerate(W):
    for j in adj_V[i]:
      q = W[j]
      if not (i in q_inds or j in p_inds):
        p_inds.append(i)
        q_inds.append(j)
  ps = np.take(W, p_inds, axis=0)
  ps = np.expand_dims(ps, axis=1)

  qs = np.take(W, q_inds, axis=0)
  qs = np.expand_dims(qs, axis=1)

  ret = np.append(ps, qs, axis=1)
  # add extra column to keep track of upper endpoint
  # 0 = upper endpoint first; 1 = lower endpoint first
  ret = np.append(ret, np.zeros(shape=ps.shape), axis=1) 
  return ret
    
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
  Naive O(n^2) check on intersection,
  TODO: change to line sweep when necessary.
  '''
  # get list with vertex pairs forming edges
  E = []
  for v in range(len(V)):
    for w in adj_V[v]:
      if [w, v] in E:
        continue
      E.append([v, w])
  # calculate intersections
  EE = combinations(E, 2)
  ret = []
  for [p, q], [r, s] in EE:
    if len(set([p, q, r, s])) < 4:
      continue # get rid of incident edges
    e1 = LineString([V[p], V[q]])
    e2 = LineString([V[r], V[s]])
    if not e1.intersects(e2):
      continue 
    int_pnt = e1.intersection(e2).xy
    int_pnt = np.array(int_pnt)
    int_pnt = np.squeeze(int_pnt)
    intersect = {
      'e1': [p, q],
      'e2': [r, s],
      'intersection': int_pnt
    }
    ret.append(intersect)
  return ret
