import numpy as np
from sklearn.neighbors import KDTree
import sys

for i in range(len(sys.argv)):
    if (sys.argv[i] == '--input_path'):
        input_path = sys.argv[i+1]
    if (sys.argv[i] == '--output_path'):
        output_path = sys.argv[i+1]
    if (sys.argv[i] == '-k'):
        k = int(sys.argv[i+1])
    if (sys.argv[i] == '-cutoff'):
        cutoff = float(sys.argv[i+1])

f = open(input_path, 'r')
data = []
for line in f.readlines():
    line = line.strip()
    line = line[line.find(',') + 1:]
    datum = line.split(",")
    for i in range(len(datum)):
        datum[i] = float(datum[i])
    data.append(datum)
f.close()

data_count = len(data)
max_dist = []
unfinished_points = []
for i in range(data_count):
    unfinished_points.append(i)
neighbors = []
data = np.array(data)

j = 1
tree = KDTree(data, leaf_size = 20)

while True:

    unfinished_points_new = []
    if len(unfinished_points) == 0:
        break

    dist, ind = tree.query(data[unfinished_points], k = k + j)

    for i in range(len(dist)):

        if j == 1:
            max_dist.append(dist[i][-1])
            neighbors.append(ind[i])

        if max_dist[unfinished_points[i]] == dist[i][-1]:
            neighbors[unfinished_points[i]] = ind[i]
            unfinished_points_new.append(unfinished_points[i])

    j += 1
    unfinished_points = unfinished_points_new

AR = []

for i in range(data_count):

    avg_R = 0
    j = 0
    for neighbor in neighbors[i]:
        if neighbor != i:
            distance = np.linalg.norm(data[i] - data[neighbor])
            R = max(distance, max_dist[neighbor])
            j += 1
            avg_R = avg_R + (R - avg_R) / j
    AR.append(avg_R)

f = open(output_path, 'w')

for i in range(data_count):

    LOF = 0
    j = 0

    for neighbor in neighbors[i]:
        if neighbor != i:
            if AR[neighbor] == 0 and AR[i] == 0:
                AR_ratio = 1
            elif AR[neighbor] == 0:
                AR_ratio = 10000
            else:
                AR_ratio = AR[i] / AR[neighbor]

            j += 1
            LOF = LOF + (AR_ratio - LOF) / j

    if LOF > cutoff:
        f.write('{0},{1}\n'.format(i+1, 1))
    else:
        f.write('{0},{1}\n'.format(i+1, 0))