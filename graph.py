import torch as tt
import numpy as np
from math import pi
from itertools import combinations

'''
Graph Operations for PrettyGD.
'''

def get_angles(W, adj_V):
  if (to_tensor := (type(W) is not tt.Tensor)):
    W = tt.Tensor(W)
  # get indices for incident edges for vectorization
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
  ji = tt.subtract(j, i)
  ki = tt.subtract(k, i)
  # dot product manually for t since torch.dot() doesn't support 2d tensors
  t = tt.multiply(ji, ki)
  t = tt.sum(t, axis=1)
  b = tt.multiply(tt.norm(ji, dim=1), tt.norm(ki, dim=1))
  ang = tt.divide(t, b)
  # get rid of rounding errors to prevent undefined inputs in arccos
  ang = tt.max(ang, -tt.ones(len(ang)))
  ang = tt.min(ang, tt.ones(len(ang)))
  ang = tt.acos(ang)
  if to_tensor:
    ang = ang.detach().numpy()
  return ang

def get_edge_li(V, adj_V):
  if type(V) is tt.Tensor:
    V = V.detach().numpy()
  # get list with vertex pairs forming edges
  ret = []
  for v in range(len(V)):
    for w in adj_V[v]:
      if [w, v] in ret:
        continue
      ret.append([v, w])
  return ret

def get_nbr_order(v, V, adj_v):
  '''
  return the circular order of the neighbors of v starting
  from edge vu, where u is the neighbor indexed first in adj_v.
  '''
  u = V[adj_v[0]]
  vu = tt.subtract(u, v)
  angs = [0] # initialize as first in the circular order
  for i in range(1, len(adj_v)):
    w = V[adj_v[i]]
    vw = tt.subtract(w, v)
    # get angle between edges vu, vw
    ang = tt.arccos(tt.dot(vu, vw) / (tt.norm(vu) * tt.norm(vw)))
    if (vw[0] * vu[1] - vu[0] * vw[1]) <= 0:
      # if vu is the right vector of the angle,
      # calculate the outer angle
      ang = (2 * pi) - ang
    angs.append(ang)
  angs = tt.tensor(angs)
  ret = tt.argsort(angs)
  ret = [adj_v[idx] for idx in ret]
  return ret

def get_intersects(V, edge_li):
  '''
  Returns vertex indices of intersecting edges and
  their intersection point.
  Naive (but vectorized!!!!) O(n^2) check on intersection,
  TODO: change to line sweep when necessary.
  '''
  if type(V) is tt.Tensor:
    V = V.detach().numpy()
  # calculate intersections
  E_inds = np.arange(0, len(edge_li))
  EE_inds = np.asarray(list(combinations(E_inds, 2)))
  # get rid of edges that share endpoints
  PQ_inds = np.take(edge_li, EE_inds[:,0], axis=0)
  RS_inds = np.take(edge_li, EE_inds[:,1], axis=0)
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

def to_vert_vals(vert_idx, V):
  '''
  Given a list of vertex indices for W, gather the vertex values i.e. x,y-values.
  '''
  vert_idx = tt.tensor(vert_idx)
  vert_idx = tt.unsqueeze(vert_idx, dim=1)
  vert_idx = vert_idx.expand(-1, 2)
  vert_vals = tt.gather(V, 0, vert_idx)
  return vert_vals

def get_displacements(V, W):
  # returns the euclidean distance
  ret = np.subtract(V, W)
  ret = np.square(ret)
  ret = np.sum(ret, axis=1)
  ret = np.sqrt(ret)
  return ret
  
