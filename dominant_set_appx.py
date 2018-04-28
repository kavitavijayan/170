import networkx as nx
from networkx.algorithms.approximation import min_weighted_dominating_set
import numpy as np
import pytspsa
import collections
from student_utils_sp18 import *
import warnings
from os import listdir
from os.path import isfile, join

warnings.filterwarnings("ignore")
INPUT_FILE = "inputs/1.in"
OUTPUT_FILE = "outputs/1.out"
NUM_MONTE_CARLO = 1

np.random.seed(42)


def create_solution(input_file, output_file):
  n, kingdoms, starting_kingdom, matrix = parse_input(input_file)
  start_k = kingdoms.index(starting_kingdom)

  G = adjacency_matrix_to_graph(matrix.tolist())

  for i in range(n):
    # Node cost represents the cost of conquering that node
    node_cost = matrix[i, i]
    for j in range(n):
      if not i == j:
        # Only talk about actual neighbours
        if matrix[i, j] > 0:
          # Cost goes "down" by "benefit"
          node_cost -= matrix[j, j]

    G.node[i]['surplus'] = node_cost

  # Upto logarithmic factors, contains the best dominating set as per surplus
  dominating_set = min_weighted_dominating_set(G, weight='surplus')
  if start_k not in dominating_set:
    dominating_set.add(start_k)

  dominating_list = sorted(list(dominating_set))
  k = len(dominating_list)


  # Compute all pairs shortest paths
  lengths = dict(nx.all_pairs_shortest_path_length(G))

  # Create a new reduced graph
  G_reduced = nx.Graph()
  G_reduced.add_nodes_from(list(dominating_set))

  for i in range(k):
    for j in range(i):
        n, m = dominating_list[i], dominating_list[j]
        weight = lengths[n][m]
        G_reduced.add_edge(n, n, weight=weight)
        G_reduced.add_edge(m, n, weight=weight)

  weights_reduced = nx.adjacency_matrix(G_reduced).astype('float32')
  weights_reduced.setdiag(np.ones(k))

  for _ in range(NUM_MONTE_CARLO):
    solver = pytspsa.Tsp_sa()
    solver.set_num_nodes(k)
    solver.add_by_distances(weights_reduced.todense())
    solver.set_t_v_factor(10.0)
    solver.sa(2018)

    solution = solver.getBestSolution()

  route = solution.getRoute().split("-")
  route = [dominating_list[int(i)] for i in route]
  print('Path= {}'.format(route))


  final_route = [route[0]]
  for i in range(len(route) - 1):
    source, target = route[i: i+2]
    path = nx.shortest_path(G, source=source, target=target)
    print(path)
    final_route.extend(path[1:])

  print('Length={}'.format(solution.getlength()))
  print(final_route)

  route = final_route[:-1]
  index = route.index((start_k))
  d = collections.deque(route)
  d.rotate(len(route) - index)
  route = list(d)
  route.append(start_k)
  print('Path= {}'.format(route))

  write_output(output_file, route, dominating_list, kingdoms)

def write_output(output_file, path, conquering_kingdoms, kingdoms):
  kingdoms_path = list(map(lambda x: kingdoms[x], path))
  kingdoms_conquer = list(map(lambda x: kingdoms[x], conquering_kingdoms))

  str1 = " ".join(kingdoms_path) + "\n"
  str2 = " ".join(kingdoms_conquer) + "\n"

  with open(output_file, 'w+') as outFile:
    outFile.write(str1)
    outFile.write(str2)

def parse_input(input_file):
  with open(input_file, 'r') as inFile:
    lines = inFile.readlines()

  n = int(lines[0])
  kingdoms = lines[1][:-1].split(" ")
  start = lines[2].strip()
  matrix = np.zeros((n, n))
  for i, line in enumerate(lines[3:]):
    if i < n - 1:
      line = line[:-1]

    costs = line.split(' ')
    for j, cost in enumerate(costs):
      if cost and not cost == 'x':
        matrix[i, j] = float(cost)

  return n, kingdoms, start, matrix

mypath = "inputs/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for input_file in onlyfiles[:10]:
  output_file = input_file.replace("in", "out")
  create_solution("inputs/" + input_file, "outputs/" + output_file)





