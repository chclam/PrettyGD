class Node:
  def __init__(self, x):
    self.x = x
    self.left = None
    self.right = None

class BinaryTree:
  def __init__(self, root):
    self.root = root

  def insert(self, node, current=None):
    if current is None:
      current = self.root
    if current.left is None and current.right is None:
      if node.x > current.x:
        current.left = node
      else:
        current.right = node
    else:
      if current.left is None:
        if node.x > current.x:
          current.left = node
        else:
          self.add(node, current.right)
      else:
        if node.x <= current.x:
          current.right = node
        else:
          self.add(node, current.left) 
  
  def delete(self, node, current=None):
    pass

