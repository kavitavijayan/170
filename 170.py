import numpy as np
import random

def generate(n):

  #Pick n points in 3 dimensional cube
  pts = np.random.uniform(low=1, high=3, size=(n, 3))


  mtx = np.zeros((n, n))
  for i in range(n):
    for j in range(i, n):
        if random.random() > 0.4:
          mtx[i, j] = round(np.linalg.norm(pts[i, :] - pts[j, :]), 4)
          mtx[j, i] = mtx[i, j]

  for i in range(n):
    mtx[i, i] = round(np.random.uniform(low=1, high=15), 4)

  for i in range(n):
    mtx[i, (i+1)%n] = round(np.linalg.norm(pts[i, :] - pts[(i+1)%n, :]), 4)
    mtx[(i+1)%n, i] = mtx[i, (i+1)%n]

  return mtx

def check_metric(mtx):
  n = mtx.shape[0]
  for i in range(n):
    for j in range(n):
      dij = mtx[i, j]
      for k in range(n):
        if k != i and k != j and i != j:
          if dij > mtx[i, k] + mtx[k, j] and mtx[i,k]>0 and mtx[k, j]>0:
            print(i, j, k)

mtx50 = generate(50)
mtx100 = generate(100)
mtx200 = generate(200)
mtxs = {50: mtx50, 100: mtx100, 200: mtx200}

for i in [50, 100, 200]:
  with open("inputs/" + str(i) + ".in", 'w+') as f:
    f.write(str(i) + "\n")
    sec_str = str(i)
    for j in range(1, i):
      sec_str += " " + str(j)
    f.write(sec_str + "\n")
    f.write("1\n")
    mtx = mtxs[i]
    for j in range(i):
      if mtx[j, 0] > 0:
        mtx_str = str(mtx[j, 0])
      else:
        mtx_str = 'x'

      for k in range(1, i):
        if mtx[j, k] == 0:
          tmp = 'x'
        else:
          tmp = str((mtx[j, k]))
        mtx_str += " " + tmp

      f.write(mtx_str + "\n")

for i in [50, 100, 200]:
  with open("outputs/" + str(i) + ".out", 'w+') as f:
    sec_str = "1"
    for j in range(2, i):
      sec_str += " " + str(j)

    f.write(sec_str + " 1\n")
    sec_str = "1"
    for j in range(2, i):
      if j % 2 == 1:
        sec_str += " " + str(j)

    f.write(sec_str + "\n")

