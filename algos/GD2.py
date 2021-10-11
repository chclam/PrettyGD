import torch
from torch.optim import SGD
import torch.nn.functional as F
from tqdm import trange 
from math import pi
import numpy as np

from . import geometry as geo

def train(V, adj_V, N=5, w_disp=20, w_cross=0.1, w_ang_res=0.2):
  W = V.copy()
  W = torch.tensor(W, requires_grad=True)
  V = torch.tensor(V)
  opt = SGD([W], lr=0.0001, momentum=0.9)
  losses = []
  for i in (t:= trange(N)):
    opt.zero_grad()
    loss = loss_function(W, V, adj_V, w_disp, w_cross, w_ang_res)
    loss.backward()
    opt.step()
    loss = loss.item()
    losses.append(loss)
    t.set_description(f"Loss: {loss:.2f}; Progress")
  return W, losses

def loss_function(W, V, adj_V, w_disp, w_cross, w_ang_res):
  # weights to tensor
  w_disp = torch.tensor(w_disp)
  w_cross = torch.tensor(w_cross)
  w_ang_res = torch.tensor(w_ang_res)

  loss_disp = loss_displacement(W, V)
  loss_disp = torch.multiply(loss_disp, w_disp)

  loss_cross = loss_crossings(W, adj_V)
  loss_cross = torch.multiply(loss_cross, w_cross)

  loss_ang_res = loss_angular_res(W, adj_V)
  loss_ang_res = torch.multiply(loss_ang_res, w_ang_res)

  ret = torch.add(loss_disp, loss_cross)
  ret = torch.add(ret, loss_ang_res)
  return ret
  
def loss_displacement(W, V):
  # squared euclidean distance
  ret = torch.subtract(W, V)
  ret = torch.pow(ret, 2)
  ret = torch.sum(ret, axis=1)
  ret = torch.sum(ret)
  return ret

def loss_crossings(W, adj_V):
  ints = geo.get_intersects(W, adj_V)
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
  ret = np.argsort(angs)
  ret = [adj_v[idx] for idx in ret]
  return ret

