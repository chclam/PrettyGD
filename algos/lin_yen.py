from . import geometry as geo
from . import graphdrawing as gd
from math import cos, pi
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def get_rep_force(v, u, w, C3, C4, C5):
  f_e = C3 * ((np.arctan(d2(v, u) / C4)) + (np.arctan(d2(v, w) / C4)))
  ang = geo.get_angle(v, w, u)
  f_ang = C5 * (np.cos(ang / 2)) / (np.sin(ang / 2))
  ret = f_e + f_ang
  return ret

def get_spring_force(v, u, C1, C2):
  uv = np.subtract(u, v)
  ret = geo.card(uv)
  ret /= C2
  ret = np.log(ret)
  ret *= C1
  ret *= (uv / geo.card(uv))
  return ret

def cross_repulsion(V, adj_V, N=5, C0=1, C1=1, C2=1):
  W = V.copy()
  for i in range(N):
    # calculate attractive forces to original positioning
    F = np.subtract(V, W)
    F = np.multiply(C0, F)
    # re-calculate the intersection points per iteration
    ints = geo.get_intersects(W, adj_V)
    for ist in ints:
      p = W[ist['e1'][0]]
      q = W[ist['e1'][1]]
      r = W[ist['e2'][0]]
      s = W[ist['e2'][1]]
      c = ist['intersection']

      is_pr = True # smallest angle is formed from vectors p and r
      ang = geo.get_angle(c, p, r)
      if ang > (pi / 2): 
        is_pr = False # smallest angle is formed from p and s
        ang = geo.get_angle(c, p, s)
      f_ps = C1 * cos(ang)

      pc = np.subtract(p, c)
      pc_u = np.divide(pc, norm(pc))
      if is_pr:
        ec = np.subtract(r, c)
      else:
        ec = np.subtract(s, c)
      ec_u = np.divide(ec, norm(ec))
      u_m = np.add(pc_u, ec_u)
      u_m = np.divide(u_m, norm(u_m))

      # expand_dims and squeeze for proper vector representation
      pc_norm = np.dot([[0, -1], [1, 0]], np.expand_dims(pc_u, 1)) # turn 90 deg
      ec_norm = np.dot([[0, 1], [-1, 0]], np.expand_dims(ec_u, 1)) # turn -90 deg
      pc_norm = np.squeeze(pc_norm) 
      ec_norm = np.squeeze(ec_norm) 

      if not np.dot(u_m[0], pc_u[1]) - np.dot(pc_u[0], u_m[1]) > 0: # if pc is not left
        pc_norm = np.multiply(-1, pc_norm) # turn pc_norm the other way
        ec_norm = np.multiply(-1, ec_norm) 

      W[ist['e1'][0]] += np.multiply(f_ps, pc_norm)
      W[ist['e1'][1]] -= np.multiply(f_ps, pc_norm)
      if not is_pr:
        ec_norm = np.multiply(-1, ec_norm)
      W[ist['e2'][0]] += np.multiply(f_ps, ec_norm)
      W[ist['e2'][1]] -= np.multiply(f_ps, ec_norm)

  F = np.multiply(C2, F)
  W = np.add(W, F)
  return W
    
def ee_repulsion(V, adj_V, cross=False):
  W = V.copy()
  F = np.zeros((len(V), 2))
  C1 = 1 
  C2 = 1
  C3 = 1
  C4 = 1
  C5 = 1
  C6 = 0.05 # set to higher value for more freedom of placement
  for i in range(len(V)):
    # calculate repulsive force
    if len(adj_V[i]) < 2:
      continue
    v = W[i]
    nbr_order = geo.get_nbr_order(v, V, adj_V[i])
    for j in range(0, len(nbr_order)):
      k = (j + 1) % len(nbr_order)
      u = W[nbr_order[j]]
      w = W[nbr_order[k]]
#       print(i, nbr_order[j], nbr_order[k])

      vu = np.subtract(u, v)
      vu_u = np.divide(vu, geo.card(vu))
      vw = np.subtract(w, v)
      u_m = vu / geo.card(vu) + vw / geo.card(vw) 
      u_m = u_m / geo.card(u_m)
      u_f = np.dot([[0, 1], [-1, 0]], u_m)
      vu_left = True if ((u_m[0] * vu_u[1]) - (vu_u[0] * u_m[1])) > 0 else False
      if vu_left:
        # change repulsion direction outwards from vu, vw
        u_f = np.multiply(-1, u_f)
      F[nbr_order[j]] += u_f
      F[nbr_order[k]] -= u_f

  for i in range(len(V)):
    W[i] += C6 * F[i]
  return W
    
if __name__ == "__main__":
  V, adj_V = gd.gen_graph()
  W = ee_repulsion(V, adj_V)
  gd.draw(V, adj_V, color='r')
  print(f'Angular resolution: {geo.get_angular_res(V, adj_V)}')
  gd.draw(W, adj_V)
  print(f'Angular resolution: {geo.get_angular_res(W, adj_V)}')

