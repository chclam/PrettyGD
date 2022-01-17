import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_FILE = "datasets/gemeente_data.json"

if os.path.isfile(OUT_FILE):
  print(f"File '{OUT_FILE}' already exists.")
  exit(0)

with open('datasets/odf_data/NL_muni_point.geojson') as geod:
  geod = json.load(geod)
  geod = geod['features']
  gem_to_ind = {}
  for i, gem in enumerate(geod):
    gem_to_ind[gem['properties']['id']] = i
  
  coords = [gem['geometry']['coordinates'] for gem in geod]
  coords = np.array(coords)
  
  adj_gem = {}
  for i in range(len(coords)):
    adj_gem[i] = []
  comd = pd.read_csv('datasets/odf_data/NL_commuting.csv')
  for i, gem in comd.iterrows():
    if gem['value'] < 500: 
      continue
    p_ind = gem_to_ind[gem['muni_from']]
    q_ind = gem_to_ind[gem['muni_to']]
    
    if q_ind not in adj_gem[p_ind] and p_ind != q_ind:
      adj_gem[p_ind].append(q_ind)
  
  # make symmetric
  for gem1 in adj_gem.keys():
    for gem2 in adj_gem[gem1]:
      if gem1 not in adj_gem[gem2]:
        adj_gem[gem2].append(gem1)
    
  
with open(OUT_FILE, 'w') as out:
  X = {
    'gemeentes': coords.tolist(),
    'adj_list': adj_gem
  }
  json.dump(X, out)

print(f"File '{OUT_FILE}' successfully written.")
