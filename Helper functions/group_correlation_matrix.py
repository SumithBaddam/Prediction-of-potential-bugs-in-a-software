import matplotlib.pyplot as plt
#plt.matshow(dataframe.corr())
import math

def top(d):
    d = d.sort_values(ascending=False)
    if(math.isnan(d[1:4][0])):
        return False, None
    return (True, d[1:4])

def dict_value(l, i):
    for k in l:
        if(i in l[k]):
            return False
    return True

c = test_df.corr()
l = {}

for i in c:
    stat = True
    if(i not in l and dict_value(l, i)):
        print(i)
        s, a = top(c[i])
        if(s):
            l[i] = a

for i in l:
    print(l[i])
