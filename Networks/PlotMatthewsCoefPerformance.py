from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np

file = open('input(Real).txt', 'r')
raw = file.readlines()
file.close()

dict={1:0, 2:1, 3:2, 5:3, 10:4, 20:5, 30:6}
dictInverse={0:1, 1:2, 2:3, 3:5, 4:19, 5:20, 6:30}

d = [[0.,1., 0.], [0.,1., 0.], [0.,1., 0.], [0.,1., 0.], [0.,1., 0.], [0.,1., 0.], [0.,1., 0.11]]  # 7 terms up to k=30

for line in range(len(raw)): # Go through each line in input.txt (from accuracy.txt)
    if(raw[line]!="\n"):
        k=int(raw[line].split('\t')[0].split()[3][2:])  # Select k value
        index = dict[k]  # lookup k index in d
        # print(k)
        current = raw[line].split('\t')[1]
        temp = float(current.split()[3])  # select MC
        d[index][0] = d[index][0] + temp  # accumulate on k

# NOW ERROR MARCINS -> Retain MIN value and MAX
        if(d[index][1]>temp):
            d[index][1] = temp
        if(d[index][2]<temp):
            d[index][2] = temp

xAxis = []
yAxis = []

for each in range(len(d)):
    d[each][0] = d[each][0]/10
    print(d[each])

    # aah = np.asarray([d[each][0]-d[each][1],d[each][2]-d[each][0]])
    # print(aah_shape)
    # plt.errorbar(dictInver- d[each][9], yerr-jaJ. e , barsabove='true')#, capsize=2)
    plt.plot(dictInverse[each],d[each][2],dictInverse[each],d[each][1])
    plt.plot(dictInverse[each],d[each][2],'r2')
    plt.plot(dictInverse[each],d[each][1],'rl')
    xAxis.append(dictInverse[each])
    yAxis.append(d[each][0])


print(d)

plt.plot(xAxis, yAxis,'#75bbfd')
plt.ylabel("Matthew's Coefficient")
plt.xlabel("ValueS of 'k'")
plt.savefig("Try2.png")
plt.show()