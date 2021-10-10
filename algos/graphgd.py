import torch
from torch.optim import SGD
import torch.nn.functional as F
from tqdm import trange 

from . import geometry as geo

def train(V, adj_V, N=5):
  W = V.copy()
  V = torch.tensor(V)
  W = torch.tensor(W, requires_grad=True)
  opt = SGD([W], lr=0.00001, momentum=0.9)
  losses = []
  for i in trange(N):
    opt.zero_grad()
    loss = loss_function(W, V, adj_V)
    losses.append(loss)
    loss.backward()
    opt.step()
  return W, losses

def loss_function(W, V, adj_V):
  loss_disp = loss_displacement(W, V)
  loss_cross = loss_crossings(W, adj_V)
  ret = torch.add(loss_disp, loss_cross)
  return ret
  
def loss_displacement(W, V):
  # squared euclidean distance
  ret = torch.subtract(W, V)
  ret = torch.pow(ret, 2)
  ret = torch.sum(ret, axis=1)
  ret = torch.sum(ret)
  return ret

def to_vert_vals(vert_idx, W):
  '''
  Given a list of vertex indices for W, gather the vertex values i.e. x,y-values.
  '''
  vert_idx = torch.tensor(vert_idx)
  vert_idx = torch.unsqueeze(vert_idx, dim=1)
  vert_idx = vert_idx.expand(-1, 2)
  ret = torch.gather(W, 0, vert_idx)
  return ret

def loss_crossings(W, adj_V):
  ints = geo.get_intersects(W, adj_V)
  
  p_idx = [it['e1'][0] for it in ints]
  q_idx = [it['e1'][1] for it in ints]
  r_idx = [it['e2'][0] for it in ints]
  s_idx = [it['e2'][1] for it in ints]

  p = to_vert_vals(p_idx, W)
  q = to_vert_vals(q_idx, W)
  r = to_vert_vals(r_idx, W)
  s = to_vert_vals(s_idx, W)
  
  e1 = torch.subtract(p, q)
  e2 = torch.subtract(r, s)
  ret = F.cosine_similarity(e1, e2)
  ret = torch.pow(ret, 2)
  ret = torch.sum(ret)
  return ret
  
