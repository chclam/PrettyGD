from . import graphdrawing as gd 
import numpy as np
import matplotlib.pyplot as plt

def force_directed(V, adj_V, C1=1, C2=1, C3=1):
  # force to original position
  W = V.copy()
  F = np.subtract(W, V)
  F = np.multiply(C1, F)
  # magnitude
  for i in range(len(V)):
    for j in range(len(V)):
      if i == j:
        continue
      F_mag = C2 / (gd.d2(V[i], V[j]))
      F_ang = np.arctan2(W[i][1] - W[j][1], W[i][0] - W[j][0])
      F[i][0] += F_mag * np.cos(F_ang)
      F[i][1] += F_mag * np.sin(F_ang)
  # apply force to new positions
  for i in range(len(V)):
    W[i][0] += C3 * F[i][0]
    W[i][1] += C3 * F[i][1]
  return W
  
if __name__ == "__main__":
  V, adj_V = gd.gen_graph()
  gd.draw(V, adj_V)
  W = birch(V, adj_V)
  gd.draw(W, adj_V)
  plt.show()

