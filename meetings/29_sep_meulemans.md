
Preserve to original positioning
---
Birchfield & Overbye (2018) impose forces of attraction on newly placed nodes
and their original position. This is a variation on the classical spring
algorithm in which adjacent nodes are attracted to each other.

Algorithm combining multiple aesthetic criteria.
---
Algorithm BIGANGLE (Huang et al.,2012) is the first recorded algorithm to optimize on two aesthetic criteria:
(1) cross angular resolution; (2) angular resolution, i.e. resolution of incident angles. They have found that BIGANGLE also improved on other aesthetic criteria
that were not explicitly considered. It empirically scored better in all measures compared to the classical spring-embedder algorithm.

The following criteria are most important in descending order (Purchase, 1997):
  1. reducing crossing edges 
  2. minimum bends (not relevant)
  3. maximizing symmetry (not relevant)
  
The following are statistically insignificant:
  1. maximizing the minimum angle between incident edges
  2. maximizing crossing angle is also shown to be insignificant for **shortest
  path tracing** (Ware et al., 2002)
  
Multicriteria optimization using Gradient Descent
---
Ahmed et al. (2020) propose a method to optimize on multiple aesthetic criteria using gradient descent,
i.e. find local minimum while considering multiple contradicting criteria. 
  

