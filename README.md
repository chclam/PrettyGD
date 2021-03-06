# PrettyGD
PrettyGD makes your geographic graph drawings prettier by optimizing on various aesthetics. It takes as input a graph with initialized vertex positions on the map and reassigns a better position for each vertex. Vertices have better placements with respect to various "aesthetic criteria" such as vertex or edge overlaps. PrettyGD uses the Gradient Descent algorithm powered by PyTorch. It currently supports the optimization on the following aesthetic criteria:

  1. **Displacement**: the displacement of each vertex from its initial position;
  2. **Angular Resolution**: the angles formed by adjacent edges of a vertex;
  3. **Cross Angular Resolution**: the angles formed by edge crossings;
  4. **Gabriel Graph Property**: the distance between vertices and unadjacent edges on the map;
  5. **Vertex Resolution**: the distance between vertices on the map.

This project is the result of a research internship at Statistics Netherlands.

## Prerequisites
PrettyGD is written in ``Python3`` and is tested in version 3.9.9. Although it should also work with version > 3.8.0. 
Run the following shell command for a minimum install of the Python dependencies using ``pip``:

```shell
pip3 install numpy torch torchvision torchaudio sklearn tqdm
```
You might have to install the following in order to execute ``commute_nl.py``:

```shell
pip3 install matplotlib plotly pyproj
```

## Usage
The syntax of PrettyGD is designed to be similar to libraries in _scikit-learn_. See the following example:

```python
from PrettyGD import PrettyGD

vert_coords = [[0,0], [1,0.5], [0,1]]
adjacency_list = {0: [1,2], 1: [0,2], 2: [0,1]}

pgd = PrettyGD(lr=0.1, weights={"displacement": 1, "vertex_res": 0.5})
pgd.fit(vert_coords, adjacency_list)
pgd.train(N=50)
new_coords = pgd.get_graph_coords()

print("Old coordinates:", vert_coords)
print("New coordinates:", new_coords)

```
The example above takes the graph as defined by ``vert_coords`` and ``adjacency_list`` and runs ``N = 50`` iterations of Gradient Descent.

### Input
PrettyGD takes as input a list of vertex coordinates (``vert_coords`` in the example) and a dictionary representing the adjacency list.

The ``adjacency_list`` must be of type ``dict``. Each key in ``adjacency_list`` corresponds to a vertex at the same index in ``vert_coords``. Note that the keys of the dictionary must be of type ``int``.

### Adjusting the Optimization Weights
To adjust the weights of the optimization criteria, simply pass a dictionary of weights during initialization of PrettyGD (see example). Every weight is set to ``1`` by default. Therefore, weight definitions that were not passed on are set to ``1`` by default. The following keys can be set:


```python
# Accepted keys for optimization weights
weights = {
  "displacement": 1,
  "crossing_ang_res": 1,
  "angular_res": 1,
  "gabriel": 1,
  "vertex_res": 1
}
pgd = PrettyGD(lr=0.01, weights)
```

### Losses
Every instance of PrettyGD keeps a list of losses. It can be accessed as follows:

```python
pgd_losses = pgd.losses
```

## Example: Commute Data of the Netherlands
Run the following line to see an application on real-life data, along with various visualizations. This program uses the library ``mapbox`` to plot the graphs on a geographic map and also creates a file in the same folder for further statistics.

```shell
./commute_nl.py
```
