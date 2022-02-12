#!/usr/bin/env python3

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyproj import Transformer

import graph as G
from prettygd import PrettyGD 

def to_lat_lon(X):
  '''
  Convert to lat-lon (espg 4326) from rijkdriehoekscoordinaten (espg 28992).
  '''
  trans = Transformer.from_crs(28992, 4326)
  ret = trans.itransform(X)
  ret = np.array(list(ret))
  return ret

def draw_map(V, adj_V, title=None):
  fig = go.Figure()
  # draw edges
  edge_inds = G.get_edge_li(V, adj_V)
  edges = []
  for i, j in edge_inds:
    edges.extend([V[i], V[j], [None, None]])
  edges = np.array(edges)
  fig.add_trace(go.Scattermapbox(lat=edges[:,0], lon=edges[:,1], mode='lines', name='Woon-werk-relatie', subplot='mapbox'))
  # draw vertices
  fig.add_trace(go.Scattermapbox(lat=V[:,0], lon=V[:,1], mode='markers', marker=go.scattermapbox.Marker(size=12), showlegend=True, name='Gemeenten', subplot='mapbox'))
  map_center = dict(lat=np.median(V[:,0]), lon=np.median(V[:,1]))
  fig.update_layout(mapbox_style='open-street-map', title=title, mapbox_zoom=6, mapbox_center=map_center)
  fig.show()

def draw_disp_map(W, V):
  '''
  Draws the original and new positions of the vertices on the map.
  '''
  disp_edges = []
  for i, v in enumerate(V):
    disp_edges.extend([W[i], V[i], [None, None]])
  disp_edges = np.array(disp_edges)
  fig = go.Figure(go.Scattermapbox(lat=disp_edges[:,0], lon=disp_edges[:,1], mode='lines', name='displacements', subplot='mapbox'))

  fig.update_traces(line_color='rgb(0, 0, 0)', selector=dict(type='scattermapbox'))
  fig.add_trace(go.Scattermapbox(lat=W[:,0], lon=W[:,1], mode='markers', showlegend=True, name="New Positions"))
  fig.add_trace(go.Scattermapbox(lat=V[:,0], lon=V[:,1], mode='markers', showlegend=True, name="Original Positions"))
  fig.update_layout(mapbox_style='open-street-map')
  fig.show()

def plot_stats(losses, disps, fname):
  # loss graph
  fig, (ax1, ax2) = plt.subplots(2, 1)
  fig.tight_layout(pad=5.0)
  ax1.plot(losses, 'ro-')
  ax1.set_title('Total Loss')
  ax1.set(xlabel='Iteration', ylabel='Total Loss')
 
  # histogram of displacements
  ax2.hist(disps)
  ax2.set_title('Displacements')
  ax2.set(xlabel='Displacement in Meters', ylabel='Number of Gemeenten')
  try:
    plt.savefig(fname)
    print(f"{fname} has been saved.")
  except Exception as e:
    print("Failed to plot the statistics: {e}")
    
if __name__ == "__main__":
  DATA_FILE = "datasets/gemeente_data.json"
  N = 30

  with open(DATA_FILE) as gem_data:
    gem_data = json.load(gem_data)

  # format dataset
  gem_coords = np.array(gem_data['gemeentes']) # coordinate of every gemeente (municipality)
  adj_gem = gem_data['adj_list'] # adjacency list of the gemeenten (connectivity of each gemeente)
  adj_gem = dict([(int(k), adj_gem[k]) for k in adj_gem.keys()]) # format to proper dictionary with integer keys
    
  # load data to PrettyGD and get new coordinates
  pgd = PrettyGD(lr=0.001)
  pgd.fit(gem_coords, adj_gem)
  pgd.train(N)
  new_coords = pgd.get_graph_coords()

  # convert and draw results
  disps = G.get_displacements(gem_coords, new_coords)
  gem_coords = to_lat_lon(gem_coords) 
  new_coords = to_lat_lon(new_coords) 

  plot_stats(pgd.losses, disps, fname="stats_displacement.jpg")
  draw_map(new_coords, adj_gem, title='New Positions')
  # uncomment the following line to visualize the displacements on the map
  # draw_disp_map(new_coords, gem_coords) 

  
