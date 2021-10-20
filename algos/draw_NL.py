import odf_vis as odf
import GD2
import graphdrawing as gd
import matplotlib.pyplot as plt
import os
import json
import plotly.express as px
import plotly.graph_objects as go
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

def draw_map(V, adj_V, title=None, disps=None):
  fig = go.Figure()
  # draw edges
  edge_inds = get_edgeset(V, adj_V)
  edges = []
  for i, j in edge_inds:
    edges.extend([V[i], V[j], [None, None]])
  edges = np.array(edges)
  fig.add_trace(go.Scattermapbox(lat=edges[:,0], lon=edges[:,1], mode='lines', name='Woon-werk-relatie', subplot='mapbox'))
  # draw vertices
  fig.add_trace(go.Scattermapbox(lat=V[:,0], lon=V[:,1], mode='markers', marker=go.scattermapbox.Marker(size=12), showlegend=True, name='Gemeenten', subplot='mapbox'))
  #fig.update_layout(mapbox_style='open-street-map', title=title, mapbox=dict(zoom=5, center=dict(lat=V[0,0], lon=V[0,1])))
  map_center = dict(lat=np.median(V[:,0]), lon=np.median(V[:,1]))

  crossings = GD2.get_intersects(V, adj_V)
  crossings = np.array([c['intersection'] for c in crossings])
  fig.add_trace(go.Scattermapbox(lat=crossings[:,0], lon=crossings[:,1], mode='markers', marker=go.scattermapbox.Marker(size=10),  name='Crossings', subplot='mapbox'))

  fig.update_layout(mapbox_style='open-street-map', title=title, mapbox_zoom=6, mapbox_center=map_center)
  fig.show()

def draw_disp_map(W, V):
  NODE_SIZE = 10
  fig = px.scatter_mapbox(lat=W[:,0], lon=W[:,1])
  fig.add_trace(go.Scattermapbox(lat=V[:,0], lon=V[:,1], mode='markers', showlegend=True, name="Original Positions"))
  fig.update_layout(mapbox_style='open-street-map')
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
    num_iters = int(input("Number of iterations: "))
    W, losses = GD2.train(coords, adj_gem, N=num_iters, lr=400, w_disp=0.0001, w_cross=1, w_ang_res=1, w_gabriel=1)
    W = W.detach().numpy()
    disps = geo.get_displacement(W, coords)

    W = to_lat_lon(W)
    coords = to_lat_lon(coords)

    draw_map(coords, adj_gem, title='Original Positions')
    draw_map(W, adj_gem, title='New Positions', disps=disps)

    # loss graph
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=5.0)
    ax1.plot(losses, 'ro-')
    ax1.set_title('Total Loss')
    ax1.set(xlabel='Iteration', ylabel='Total Loss')

    # histogram
    disps = np.array(disps)
    ax2.hist(disps)
    ax2.set_title('Displacements')
    ax2.set(xlabel='Displacement in Meters', ylabel='Number of Gemeenten')
    plt.show()

