import numpy as np
from math import sqrt, pi
from itertools import combinations
from shapely.geometry import LineString
from numpy.linalg import norm

def d1(p, q):
  ret = abs(p[0] - q[0])
  ret += abs(p[1] - q[1])
  return ret

def d2(p, q):
  ret = (p[0] - q[0]) ** 2
  ret += (p[1] - q[1]) ** 2
  ret = sqrt(ret)
  return ret

def get_angle(v, w, u):
  '''
  Gets angle between vectors vu and vw.
  '''
  vu = np.subtract(u, v)
  vw = np.subtract(w, v)
  ret = np.arccos(np.dot(vu, vw) / (norm(vu) * norm(vw)))
  return ret

def get_nbr_order(v, V, adj_v):
  '''
  return the circular order of the neighbors of v starting
  from edge vu, where u is the neighbor indexed first in adj_v.
  '''
  u = V[adj_v[0]]
  vu = np.subtract(u, v)
  angs = [0] # initialize as first in the circular order
  for i in range(1, len(adj_v)):
    w = V[adj_v[i]]
    vw = np.subtract(w, v)
    # get angle between edges vu, vw
    ang = np.arccos(np.dot(vu, vw) / (norm(vu) * norm(vw)))
    if (vw[0] * vu[1] - vu[0] * vw[1]) <= 0:
      # if vu is the right vector of the angle,
      # calculate the outer angle
      ang = (2 * pi) - ang
    angs.append(ang)
  ret = np.argsort(angs)
  ret = [adj_v[idx] for idx in ret]
  return ret

def bfs(v, adj_V):
  vis = [v]
  vis.extend(adj_V[v])
  q = list(adj_V[v])
  while len(q) > 0:
    w = q.pop(0)
    for u in adj_V[w]:
      if u not in vis:
        vis.append(u)
        q.append(u)
  return vis

def get_angular_res(V, adj_V):
  # returns average angle between incident edges
  ret = []
  for i in range(len(V)):
    if len(adj_V[i]) < 2:
      continue
    v = V[i]
    nbr_order = get_nbr_order(v, V, adj_V[i])
    for j in range(0, len(nbr_order)):
      k = (j + 1) % len(nbr_order)
      u = V[nbr_order[j]]
      w = V[nbr_order[k]]
      a = get_angle(v, w, u)
      ret.append(a)
  ret = sum(ret) / len(ret)
  return ret

def get_displacement(V, W):
  # returns the **squared** euclidean distance
  ret = np.subtract(V, W)
  ret = np.power(ret, 2)
  ret = np.sum(ret, axis=1)
  ret = np.sqrt(ret)
  return ret
  
