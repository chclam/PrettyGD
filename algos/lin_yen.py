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
  ret = norm(uv)
  ret /= C2
  ret = np.log(ret)
  ret *= C1
  ret *= (uv / norm(uv))
  return ret

def ev_repulsion(V, adj_V, N=5, C1=1, C2=1, C3=1):
  W = V.copy()
  for a in range(N):
    F = np.subtract(V, W)
    F = np.multiply(C1, F)
    for i in range(len(V)):
      v = W[i]
      for j in range(len(V)):
        if i == j:
          continue
        w = W[j]
        for k in adj_V[j]:
          if i == k:
            continue
          u = W[k]
          vu = np.subtract(v, u)
          vw = np.subtract(v, w)
          vu_u = np.divide(vu, norm(vu))
          vw_u = np.divide(vw, norm(vw))
          bis = np.add(vu_u, vw_u)
          F_ang = np.divide(bis, norm(bis))
          uw = np.subtract(u, w)
          uw = np.divide(uw, 2)
          uwu = (w - v) + uw
          F_mag = C2 / norm(v - uwu)
          F[i] += (F_mag * F_ang)
    for i in range(len(V)):
      W[i] += np.multiply(C3, F[i])
  return W

def is_left_edge(e, f):
  '''
  Returns true if e is the left edge incident edge.
  '''
  if np.dot(f[0], e[1]) - np.dot(e[0], f[1]) > 0:
    return True
  return False

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

      angle_pr = True # smallest angle is formed from vectors p and r
      ang = geo.get_angle(c, p, r)
      if ang > (pi / 2): 
        angle_pr = False # smallest angle is formed from p and s
        ang = geo.get_angle(c, p, s)
      f_cos = C1 * cos(ang)

      pc = np.subtract(p, c)
      pc_u = np.divide(pc, norm(pc))
      if angle_pr:
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

      if not is_left_edge(pc_u, u_m):
        pc_norm = np.multiply(-1, pc_norm) # turn pc_norm the other way
        ec_norm = np.multiply(-1, ec_norm) 

      W[ist['e1'][0]] += np.multiply(f_cos, pc_norm)
      W[ist['e1'][1]] -= np.multiply(f_cos, pc_norm)
      if not angle_pr:
        ec_norm = np.multiply(-1, ec_norm)
      W[ist['e2'][0]] += np.multiply(f_cos, ec_norm)
      W[ist['e2'][1]] -= np.multiply(f_cos, ec_norm)

    F = np.multiply(C2, F)
    W = np.add(W, F)
  return W

def bigangle(V, adj_V, N=5, C0=1, C1=1, C2=1, C3=1, F2V=True):
  W = V.copy()
  for i in range(N):
    # calculate attractive forces to original positioning
    if F2V:
      F = np.subtract(V, W)
      F = np.multiply(C0, F)
    else:
      F = np.zeros((len(V), 2))
    # calculate cos forces
    # re-calculate the intersection points per iteration
    ints = geo.get_intersects(W, adj_V)
    for ist in ints:
      p = W[ist['e1'][0]]
      q = W[ist['e1'][1]]
      r = W[ist['e2'][0]]
      s = W[ist['e2'][1]]
      c = ist['intersection']

      angle_pr = True # smallest angle is formed from vectors p and r
      ang = geo.get_angle(c, p, r)
      if ang > (pi / 2): 
        angle_pr = False # smallest angle is formed from p and s
        ang = geo.get_angle(c, p, s)
      f_cos = C1 * cos(ang)

      pc = np.subtract(p, c)
      pc_u = np.divide(pc, norm(pc))
      if angle_pr:
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

      if not is_left_edge(pc_u, u_m):
        pc_norm = np.multiply(-1, pc_norm) # turn pc_norm the other way
        ec_norm = np.multiply(-1, ec_norm) 

      W[ist['e1'][0]] += np.multiply(f_cos, pc_norm)
      W[ist['e1'][1]] -= np.multiply(f_cos, pc_norm)
      if not angle_pr:
        ec_norm = np.multiply(-1, ec_norm)
      W[ist['e2'][0]] += np.multiply(f_cos, ec_norm)
      W[ist['e2'][1]] -= np.multiply(f_cos, ec_norm)

    # calculate sin forces
    for i, v in enumerate(W):
      # calculate repulsive force
      if len(adj_V[i]) < 2:
        continue
      nbr_order = geo.get_nbr_order(v, V, adj_V[i])
      opt_ang = (2 * pi) / len(nbr_order)
      for j in range(0, len(nbr_order)):
        k = (j + 1) % len(nbr_order)
        u = W[nbr_order[j]]
        w = W[nbr_order[k]]
        ang = geo.get_angle(v, u, w)
        f_sin = C2 * np.sin((opt_ang - ang) / 2)

        vu = np.subtract(u, v)
        vu_u = np.divide(vu, norm(vu))
        vw = np.subtract(w, v)

        u_m = np.divide(vu, norm(vu)) 
        u_m += np.divide(vw, norm(vw))
        u_m = u_m / norm(u_m)
        vu_norm = np.divide(vu, norm(vu))
        vu_norm = np.expand_dims(vu_norm, 1)
        vu_norm = np.dot([[0, -1], [1, 0]], vu_norm)
        vu_norm = np.squeeze(vu_norm)

        vw_norm = np.divide(vw, norm(vw))
        vw_norm = np.expand_dims(vw_norm, 1)
        vw_norm = np.dot([[0, 1], [-1, 0]], vw_norm)
        vw_norm = np.squeeze(vw_norm)

        if not is_left_edge(vu_u, u_m):
          # vu is not the left edge, i.e. right edge, invert the directions
          vu_norm = np.multiply(-1, vu_norm)
          vw_norm = np.multiply(-1, vw_norm)
        F[nbr_order[j]] += np.multiply(f_sin, vu_norm)
        F[nbr_order[k]] += np.multiply(f_sin, vw_norm)

    F = np.multiply(C3, F)
    W = np.add(W, F)
  return W
    
def ee_repulsion(V, adj_V, N=5, C0=1, C1=1, F2V=True):
  W = V.copy()
  for a in range(N):
    if F2V:
      F = np.subtract(V, W)
      F = np.multiply(C0, F)
    else:
      F = np.zeros((len(V), 2))
    for i, v in enumerate(W):
      # calculate repulsive force
      if len(adj_V[i]) < 2:
        continue
      nbr_order = geo.get_nbr_order(v, V, adj_V[i])
      for j in range(0, len(nbr_order)):
        k = (j + 1) % len(nbr_order)
        u = W[nbr_order[j]]
        w = W[nbr_order[k]]

        vu = np.subtract(u, v)
        vu_u = np.divide(vu, norm(vu))
        vw = np.subtract(w, v)
        u_m = vu / norm(vu) + vw / norm(vw) 
        u_m = u_m / norm(u_m)
        u_m = np.expand_dims(u_m, 1)
        u_f = np.dot([[0, 1], [-1, 0]], u_m)
        u_f = np.squeeze(u_f)
        vu_left = True if ((u_m[0] * vu_u[1]) - (vu_u[0] * u_m[1])) > 0 else False
        if vu_left:
          # change repulsion direction outwards from vu, vw
          u_f = np.multiply(-1, u_f)
        F[nbr_order[j]] += u_f
        F[nbr_order[k]] -= u_f
    F = np.multiply(C1, F)
    W = np.add(W, F)
  return W
    
if __name__ == "__main__":
  V, adj_V = gd.gen_graph()
  W = ee_repulsion(V, adj_V)
  gd.draw(V, adj_V, color='r')
  print(f'Angular resolution: {geo.get_angular_res(V, adj_V)}')
  gd.draw(W, adj_V)
  print(f'Angular resolution: {geo.get_angular_res(W, adj_V)}')

