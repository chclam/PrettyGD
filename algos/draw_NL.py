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

def get_edgeset(V, adj_V):
  ret = []
  for i in range(len(V)):
    for j in adj_V[i]:
      if (j, i) not in ret:
        ret.append((i, j))
  return ret

def draw_map(V, adj_V):
  import plotly.express as px
  import plotly.graph_objects as go
  fig = px.scatter_mapbox(lat=V[:,0], lon=V[:,1], size=disps)
  edge_inds = get_edgeset(V, adj_V)
  edges = []
  for i, j in edge_inds:
    edges.extend([V[i], V[j], [None, None]])
  edges = np.array(edges)
  fig.add_trace(go.Scattermapbox(lat=edges[:,0], lon=edges[:,1], mode='lines', showlegend=True))
  fig.update_layout(mapbox_style='open-street-map', mapbox_zoom=5)
  fig.show()

def to_lat_lon(X):
  '''
  Convert from rijkdriehoekscoordinaten (espg 28992) to lat lon (espg 4326).
  '''
  from pyproj import Transformer
  trans = Transformer.from_crs(28992, 4326)
  ret = trans.itransform(X)
  ret = np.array(list(ret))
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
    #num_iters = 1
    num_iters = int(input("Number of iterations: "))
    W, losses = GD2.train(coords, adj_gem, N=num_iters, lr=400, w_disp=0.00001, w_cross=1, w_ang_res=1, w_gabriel=1)
    W = W.detach().numpy()
    disps = geo.get_displacement(W, coords)

   # gd.draw(W, adj_gem)
    W = to_lat_lon(W)

    draw_map(W, adj_gem)

    plt.plot(losses, 'bo-')
    plt.title('Total Loss')
    plt.show()

    disps = np.array(disps)
    plt.hist(disps)
    #plt.plot(disps, 'ro')
    plt.title('Histogram of Displacements')
    plt.show()
