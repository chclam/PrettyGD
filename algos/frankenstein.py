from . import graphdrawing as gd
from math import sqrt, pi
import numpy as np

def get_angular_res(V, adj_V):
  ret = []
  for i in range(len(V)):
    if len(adj_V[i]) < 2:
      continue
    v = V[i]
    nbr_order = gd.get_nbr_order(v, V, adj_V[i])
    for j in range(0, len(nbr_order)):
      k = (j + 1) % len(nbr_order)
      u = V[nbr_order[j]]
      w = V[nbr_order[k]]
      a = get_angle(v, w, u)
      ret.append(a)
  ret = min(ret)
  return ret
  
def get_rep_force(v, u, w, C3, C4, C5):
  f_e = C3 * ((np.arctan(d2(v, u) / C4)) + (np.arctan(d2(v, w) / C4)))
  ang = get_angle(v, w, u)
  f_ang = C5 * (np.cos(ang / 2)) / (np.sin(ang / 2))
  ret = f_e + f_ang
  return ret

def get_spring_force(v, u, C1, C2):
  uv = np.subtract(u, v)
  ret = gd.card(uv)
  ret /= C2
  ret = np.log(ret)
  ret *= C1
  ret *= (uv / gd.card(uv))
  return ret

def ev_repulsion(V, adj_V, ):
  W = V.copy()
  F = np.zeros(len(V), 2)
  for i in range(len(V)):
    for j in range(len(V)):
      for k in adj_V[j]:
        u = V[k]
        if i == k or i == j:
          continue
        vu = np.subtract(u, v)
        vw = np.subtract(w, v)
        vu_u = np.divide(vu, gd.card(vu))
        vw_u = np.divide(vw, gd.card(vw))
        bis = np.add(vu_u, vw_u)
        bis = np.divide(bis, 2)
       # bs_n = bis / gd.card(bis)
        bs_n *= -1
        F[i] += bs_n
  return W
        
def frank(V, adj_V, N=5, C0=1, C6=0.05):
  # set C6 to higher value for more freedom of placement
  W = V.copy()
  C1 = 1 
  C2 = 1
  C3 = 1
  C4 = 1
  C5 = 1
  for a in range(N):
    F = np.subtract(V, W)
    F = np.multiply(C0, F)
    for i in range(len(V)):
      # calculate repulsive force
      if len(adj_V[i]) < 2:
        continue
      v = W[i]
      nbr_order = gd.get_nbr_order(v, V, adj_V[i])
      for j in range(0, len(nbr_order)):
        k = (j + 1) % len(nbr_order)
        u = W[nbr_order[j]]
        w = W[nbr_order[k]]
        
        vu = np.subtract(u, v)
        vu_u = np.divide(vu, gd.card(vu))
        vw = np.subtract(w, v)
        
        u_m = vu / gd.card(vu) + vw / gd.card(vw) 
        u_m = u_m / gd.card(u_m)
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
  gd.draw(V, adj_V, color='r')
  W = frank(V, adj_V)
  gd.draw(W, adj_V)

