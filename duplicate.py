import numpy as np
import os
import re

np.random.seed(12)
dic = {}
path = 'data1/'
files = os.listdir(path)

k0 = 3
t = 0
data_count = 0

for file in files:

    if data_count % 200 == 0:
        print(data_count)
    k = 0
    shingle = ""
    f = open(path + file, 'r', encoding = 'UTF-8')
    text = f.read().split()
    for str0 in text:
        str1 = re.sub(r'[^A-Za-z]', '', str0)
        if len(str1) > 0:
            if k >= k0:
                shingle = shingle[shingle.find(' ') + 1:] + " " + str1.lower()
                if shingle not in dic:
                    t += 1
                    dic[shingle] = t
            elif k == 0:
                k += 1
                shingle = str1.lower()
            else:
                k += 1
                shingle = shingle + " " + str1.lower()
    f.close()
    data_count += 1

c = len(dic)
while True:
    flag = True
    for i in range(2, int(np.sqrt(c)) + 1):
        if c % i == 0:
            c += 1
            flag = False
            break
    if flag:
        break

permutation_count = 300
a1 = np.random.randint(c, size = (permutation_count, 1))
a2 = np.random.randint(c, size = (permutation_count, 1))
sig_matrix = np.zeros((permutation_count, data_count))

count = 0

for count in range(data_count):

    if count % 200 == 0:
        print(count)
    k = 0
    datum = []
    shingle = ""
    f = open(path + str(count) + ".txt", 'r', encoding='UTF-8')
    text = f.read().split()
    for str0 in text:
        str1 = re.sub(r'[^A-Za-z]', '', str0)
        if len(str1) > 0:
            if k >= k0:
                shingle = shingle[shingle.find(' ') + 1:] + " " + str1.lower()
                datum.append(dic[shingle])
            elif k == 0:
                shingle = str1.lower()
            else:
                shingle = shingle + " " + str1.lower()
            k += 1
    f.close()
    datum = np.array(datum).reshape(1, k - k0)
    sig_vector = np.dot(a1, datum) + a2
    sig_vector = np.min(sig_vector % c, axis = 1)
    sig_matrix[:,count] = sig_vector
    count += 1

f = open('duplicate.csv', 'w')
f.write('{0},{1}\n'.format("Id", "DocumentList"))
r = 2
b = int(permutation_count / r)
buckets_count = 100000
a1 = np.random.randint(buckets_count, size = (b, r))
a2 = np.random.randint(buckets_count, size = (b, 1))
tao = 0.2
candidate_pairs = {}

for i in range(b):

    buckets = {}
    for j in range(data_count):
        hash_val = (np.dot(a1[i,:], sig_matrix[r*i:r*(i+1), j]) + a2[i]) % buckets_count
        if hash_val[0] not in buckets:
            buckets[hash_val[0]] = [j]
        else:
            buckets[hash_val[0]].append(j)
    for j in range(buckets_count):
        if j not in buckets:
            continue
        if len(buckets[j]) > 1:
            for k in buckets[j]:
                for l in buckets[j]:
                    tmp = np.count_nonzero(sig_matrix[:,l] - sig_matrix[:,k])
                    if k != l and tmp / permutation_count < 1 - tao:
                        if k not in candidate_pairs:
                            candidate_pairs[k] = str(l)
                        elif str(l) not in candidate_pairs[k]:
                            candidate_pairs[k] = candidate_pairs[k] + " " + str(l)

for i in range(data_count):
    if i not in candidate_pairs:
        f.write('{0},{1}\n'.format(i, -1))
    else:
        f.write('{0},{1}\n'.format(i, candidate_pairs[i]))