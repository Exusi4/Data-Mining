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
ita = 0.3 * 1e-6
eps = 0.25
C = 100

f = C * w.shape[0]
delta = 100
cost = [f]

while delta >= eps:

    temp = np.where(labels*(np.dot(data, w.T) + b) < 1)[0]
    dw = w + C * np.sum(-labels[temp] * data[temp], axis = 0)
    db = C * np.sum(-labels[temp])

    w = w - ita * dw
    b = b - ita * db
    k += 1

    temp = 1 - labels*(np.dot(data, w.T) + b)
    temp[temp < 0] = 0
    f = np.dot(w, w.T) / 2 + C * np.sum(temp)
    delta = 100 * abs(cost[-1] - f) / cost[-1]
    cost.append(f)

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

f = open('bgd.csv', 'w')
f.write('{0},{1}\n'.format("Id", "Expected"))
for i in range(len(test_labels)):
    f.write('{0},{1}\n'.format(i + 80002, int(test_labels[i].item())))