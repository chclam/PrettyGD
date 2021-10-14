import odf_vis as odf
import GD2
import graphdrawing as gd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import geometry as geo

def format_json_loads(X):
  coords = X['gemeentes']
  coords = np.array(coords)
  adj_gem_raw = X['adj_list']
  adj_gem = {}
  for k in adj_gem_raw.keys():
    adj_gem[int(k)] = adj_gem_raw[k]
  return coords, adj_gem

def get_edgeset(W, adj_V):
  ret = []
  for i in range(len(W)):
    for j in adj_V[i]:
      if (j, i) not in ret:
        ret.append((i, j))
  return ret

if __name__ == "__main__":
  GEM_FILE = "gemeente_data.json"

  if GEM_FILE not in os.listdir():
    coords, adj_gem = odf.get_ww()
    with open(GEM_FILE, 'w') as out:
      X = {
        'gemeentes': coords.tolist(),
        'adj_list': adj_gem
      }
      json.dump(X, out)

  with open(GEM_FILE) as gem_data:
    gem_data = json.load(gem_data)
    coords, adj_gem = format_json_loads(gem_data)
    #num_iters = int(input("Number of iterations: "))
    #W, losses = GD2.train(coords, adj_gem, N=num_iters, lr=1, w_disp=0.01, w_cross=0.5, w_ang_res=1)
    #W = W.detach().numpy()
    #disps = geo.get_displacement(W, coords)
    gd.draw(coords, adj_gem, 'r')
    #plt.scatter(x=W[:,0], y=W[:,1], c=disps, s=200, cmap='YlOrRd')
    plt.show()
