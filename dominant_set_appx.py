import networkx as nx
from networkx.algorithms.approximation import min_weighted_dominating_set
import numpy as np

INPUT_FILE = "inputs/0.in"

def parse_input(input_file):
  with open(input_file, 'r') as inFile:
    lines = inFile.readlines()

  n = int(lines[0])
  kingdoms = lines[1][:-1].split(" ")
  matrix = np.zeros((n, n))
  for i, line in enumerate(lines[3:]):
    if i < n - 1:
      line = line[:-1]

    costs = line.split(' ')
    for j, cost in enumerate(costs):
      if cost and not cost == 'x':
        matrix[i, j] = int(cost)

  return kingdoms, matrix

G = nx.Graph()
kingdoms, matrix = parse_input(INPUT_FILE)

n, _ = matrix.shape
G.add_nodes_from(list(range(n)))

for i in range(n):
  # Node cost represents the cost of conquering that node
  node_cost = matrix[i, i]
  for j in range(n):
    if not i == j:
      # Only talk about actual neighbours
      if matrix[i, j] > 0:
        # Cost goes "down" by "benefit"
        node_cost -= matrix[j, j]
        G.add_edge(i, j, weight=matrix[i, j])

  G.node[i]['cost'] = node_cost

dominanting_set = min_weighted_dominating_set(G, weight='cost')


