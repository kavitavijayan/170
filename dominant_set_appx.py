import networkx as nx
from networkx.algorithms.approximation import min_weighted_dominating_set
import numpy as np
import pytspsa
from pytsp import atsp_tsp, run, dumps_matrix
import collections
from student_utils_sp18 import *
import warnings
import os
from os import listdir, chdir
from os.path import isfile, join
import unicodedata
from multiprocessing import Process

warnings.filterwarnings("ignore")
NUM_MONTE_CARLO = 1
OVERWRITE = False
TIMEOUT = 15

np.random.seed(42)

def get_tour_by_annealing(k, weights_reduced):
  """
  Gets the best tour by simulated annealing.
  """
  weights_reduced.setdiag(np.ones(k))
  min_cost = np.inf
  solution = None

  for _ in range(NUM_MONTE_CARLO):
    solver = pytspsa.Tsp_sa()
    solver.set_num_nodes(k)
    solver.add_by_distances(weights_reduced.todense())
    solver.set_t_v_factor(10.0)
    solver.sa(12)

    cur_soln = solver.getBestSolution()
    if cur_soln.getlength() < min_cost:
      min_cost = cur_soln.getlength()
      solution = cur_soln

  route = solution.getRoute().split("-")
  return route

def get_tour_concorde(weights_reduced):
  """
  Runs the Concord algorithm to solve a heuristic approach to the TSP
  problem.
  """
  dir_path = os.path.dirname(os.path.realpath(__file__))

  matrix_sym = atsp_tsp(weights_reduced.todense(), strategy="avg")
  outf = "/tmp/myroute.tsp"
  with open(outf, 'w') as dest:
      dest.write(dumps_matrix(matrix_sym, name="My Route"))

  tour = run(outf, start=0, solver="concorde")
  os.chdir(dir_path)
  route = tour['tour']
  route.append(route[0])

  return route

def create_solution(input_file, output_file):
  n, kingdoms, starting_kingdom, matrix = parse_input(input_file)
  start_k = kingdoms.index(starting_kingdom)
  conquer_start = True

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
    conquer_start = False

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

  # route = get_tour_concorde(weights_reduced)
  route = get_tour_by_annealing(k, weights_reduced)
  route = [dominating_list[int(i)] for i in route]

  final_route = [route[0]]
  for i in range(len(route) - 1):
    source, target = route[i: i+2]
    path = nx.shortest_path(G, source=source, target=target)
    final_route.extend(path[1:])

  route = final_route[:-1]
  index = route.index((start_k))
  d = collections.deque(route)
  d.rotate(len(route) - index)
  route = list(d)
  route.append(start_k)

  if not conquer_start:
    dominating_list.remove(start_k)
    print("Removing starting kingdom to conquer.")

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
    lines = [unicodedata.normalize("NFKD", line) for line in inFile.readlines()]

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
        try:
          matrix[i, j] = float(cost)
        except:
          pass

  return n, kingdoms, start, matrix

mypath = "inputs/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for input_file in onlyfiles:
  output_file = input_file.replace("in", "out")

  # Don't want to overwrite a solution
  if not OVERWRITE and isfile("outputs/" + output_file):
    continue

  print("Working on file: " + input_file)
  try:
    action_process = Process(target=create_solution, args=(
      "inputs/" + input_file, "outputs/" + output_file, ))
    action_process.start()
    action_process.join(timeout=TIMEOUT)

    if action_process.is_alive():
      print("Killing process due to timeout.")
      action_process.terminate()
  except Exception as e:
    print("Error: ", e)
    print("Could not solve file: " + input_file)





