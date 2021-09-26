import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi
from itertools import combinations

def draw(V, adj_V, color='b'):  
  # draw vertices
  plt.plot(V[:,0], V[:,1], '{}o'.format(color))
  for i in range(len(V)):
    # add vertex labels
    plt.text(V[i,0] + 0.01, V[i,1] + 0.01, i)
  
  # draw edges
  for i in range(len(V)):
    v = V[i]
    for j in adj_V[i]:
      w = V[j]
      xs = [v[0], w[0]]
      ys = [v[1], w[1]]
      plt.plot(xs, ys, '{}-'.format(color))
  plt.show()
  
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

def gen_graph():
  V = np.random.rand(10, 2)
  adj_V = {}
  # create random adjacency list
  for i in range(len(V)):
    adj = np.random.randint(0, len(V) - 1, np.random.randint(0, 5))
    adj = adj[adj != i]
    adj = list(set(adj))
    adj_V[i] = adj

  # make adjacency list symmetric
  for i in range(len(V)):
    for nbr in adj_V[i]:
      if i not in adj_V[nbr]:
        adj_V[nbr].extend([i])
  return V, adj_V

def d2(p, q):
  ret = (p[0] - q[0]) ** 2
  ret += (p[1] - q[1]) ** 2
  ret = sqrt(ret)
  return ret

def d1(p, q):
  ret = abs(p[0] - q[0])
  ret += abs(p[1] - q[1])
  return ret

def card(v):
  ret = np.power(v, 2)
  ret = np.sum(ret)
  ret = sqrt(ret)
  return ret

def get_angle(v, w, u):
  '''
  Gets angle between vectors vu and vw.
  '''
  vu = np.subtract(u, v)
  vw = np.subtract(w, v)
  ret = np.arccos(np.dot(vu, vw) / (card(vu) * card(vw)))
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
    ang = np.arccos(np.dot(vu, vw) / (card(vu) * card(vw)))
    if (vw[0] * vu[1] - vu[0] * vw[1]) <= 0:
      # if vu is the right vector of the angle,
      # calculate the outer angle
      ang = (2 * pi) - ang
    angs.append(ang)
  ret = np.argsort(angs)
  ret = [adj_v[idx] for idx in ret]
  return ret

