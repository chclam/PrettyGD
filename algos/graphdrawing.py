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
  
def gen_graph(num_verts=10, max_nbrs=5):
  # returns vertices with corresponding adjacency list 
  V = np.random.rand(num_verts, 2)
  adj_V = {}
  # create random adjacency list
  for i in range(len(V)):
    adj = np.random.randint(0, len(V) - 1, np.random.randint(0, min(len(V) - 1, max_nbrs)))
    adj = adj[adj != i]
    adj = list(set(adj))
    adj_V[i] = adj
  # make adjacency list symmetric
  for i in range(len(V)):
    for nbr in adj_V[i]:
      if i not in adj_V[nbr]:
        adj_V[nbr].extend([i])
  return V, adj_V

