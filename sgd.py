import numpy as np
import sys

for i in range(len(sys.argv)):
    if (sys.argv[i] == '--train_data'):
        train_data = sys.argv[i+1]
    if (sys.argv[i] == '--train_labels'):
        train_labels = sys.argv[i+1]
    if (sys.argv[i] == '--test_data'):
        test_data = sys.argv[i+1]

fp = open(train_data, 'r')
data = []
for line in fp.readlines():
    line = line.strip()
    line = line[line.find(',') + 1:]
    datum = line.split(",")
    for i in range(len(datum)):
        datum[i] = float(datum[i])
    data.append(datum)
fp.close()

fp = open(train_labels, 'r')
labels = []
for line in fp.readlines():
    line = line.strip()
    line = line[line.find(',') + 1:]
    if (int(line)) == 1:
        labels.append(1)
    else:
        labels.append(-1)
fp.close()
data = np.array(data)
labels = np.array(labels).reshape(data.shape[0],1)
w = np.zeros((1, data.shape[1]))
b = 0
k = 0
ita = 0.1 * 1e-3
eps = 0.001
C = 100

f = C * data.shape[0]
delta = 0
cost = [f]

while delta >= eps or k == 0:

    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    labels = labels[perm]

    for i in range(data.shape[0]):

        if labels[i] * (np.dot(data[i], w.T) + b) < 1:
            dw = w - C * labels[i] * data[i]
            db = -C * labels[i]
        else:
            dw = w * 1
            db = 0

        w = w - ita * dw
        b = b - ita * db
        k += 1

    temp = 1 - labels * (np.dot(data, w.T) + b)
    temp = np.max(temp, 0)
    f = np.dot(w, w.T) / 2 + C * np.sum(temp)
    delta = delta/2 + 50 * abs(cost[-1] - f.item()) / cost[-1]
    if delta < eps:
        break
    cost.append(f.item())
    ita *= 0.5

fp = open(test_data, 'r')
test = []
for line in fp.readlines():
    line = line.strip()
    line = line[line.find(',') + 1:]
    datum = line.split(",")
    for i in range(len(datum)):
        datum[i] = float(datum[i])
    test.append(datum)
fp.close()
test = np.array(test)

test_labels = np.dot(test, w.T) + b
test_labels[test_labels > 0] = 1
test_labels[test_labels <= 0] = 0

f = open('sgd.csv', 'w')
f.write('{0},{1}\n'.format("Id", "Expected"))
for i in range(len(test_labels)):
    f.write('{0},{1}\n'.format(i + 80002, int(test_labels[i].item())))