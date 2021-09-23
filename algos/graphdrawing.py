import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def draw(V, adj_V, color='b'):  
  # draw vertices
  plt.plot(V[:,0], V[:,1], '{}o'.format(color))
  for i in range(len(V)):
    # add vertex labels
    plt.text(V[i,0] + 0.01, V[i,1] + 0.01, i)
  
  # draw edges]
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

