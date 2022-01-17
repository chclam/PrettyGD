# PrettyGD
PrettyGD makes your geographic graph drawings prettier by optimizing on various aesthetics. It takes as input a graph with initialized vertex positions on the map and reassigns better a position for each vertex on the map. Vertices have better placements with regard to various "aesthetic criteria" such as vertex or edge overlaps. PrettyGD currently supports the optimization on the following aesthetic criteria:

  1. Displacement: the displacement of each vertex from its initial position;
  2. Angular Resolution: the angles formed by adjacent edges of a vertex;
  3. Cross Angular Resolution: the angles formed by edge crossings;
  4. Gabriel Graph Property: the distance between vertices and unadjacent edges on the map;
  5. Vertex Resolution: the distance between vertices on the map.


## Simple example
The syntax of PrettyGD is similar to libraries in _sklearn_.

```python
from prettygd import PrettyGD

vert_coords = [[0., 0.], [1., 0.5], [0., 1.]]
adjacency_list = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

pgd = PrettyGD(lr=0.001, weights={"displacement": 1., "vertex_res": 0.5, "angular_res": 0.7})
pgd.fit(vert_coords, adjacency_list)
pgd.train(N=50)
new_coords = pgd.get_graph_coords()

print("Old coordinates:", vert_coords)
print("New coordinates:", new_coords)

```

