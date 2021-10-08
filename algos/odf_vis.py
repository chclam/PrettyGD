import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_ww():
  # really ugly code
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
    
  return coords, adj_gem


if __name__ == "__main__":
  import graphdrawing as gd
  import lin_yen as ly
  import matplotlib.pyplot as plt
  import geometry as geo
  import json
  import os
  
  GEM_FILE = "./gemeente_data.json"

  if GEM_FILE not in os.listdir():
    coords, adj_gem = get_ww()
    with open(GEM_FILE, 'w') as out:
      print('h')
      X = {
        'gemeentes': coords,
        'adj_list': adj_gem
      }
      json.dump(X, out)

  with open(GEM_FILE) as gem_data:
    gem_data = json.load(gem_data)
    coords = gem_data['gemeentes']
    adj_gem = gem_data['adj_list']
      
    W = ly.bigangle(coords, adj_gem, N=5, C0=1, C1=1, C2=1, C3=1, F2V=True)
    gd.draw(coords, adj_gem)
    plt.show()
    disps = geo.get_displacement(coords, W)
    plt.scatter(x=W[:,0], y=W[:,1], c=disps, s=2000, cmap='YlOrRd')
    gd.draw(W, adj_gem)
    plt.show()
