import numpy as np
#%%

l_01 =[9460,4,0,1,0,0,0,0,0]

l_02 =[3337,4,0,1,3,0,0,0,0]

l_03 =[4352,6,0,2,10,0,0,0,0]

l_05 =[8893,18,0,8,20,0,0,1,0]

l_07 =[9227,18,0,8,37,0,0,0,0]

l_1 =[11598,26,0,11,67,4,0,0,4]

l_2 =[23587,89,1,52,280,2,1,2,11]

l_3 =[11954,54,1,33,302,4,0,3,9]

l_5 =[12881,87,0,50,643,4,0,1,24]

l_7 =[7888,57,0,42,733,7,0,4,34]

l_10 =[9238,85,5,48,1209,8,1,6,75]

l_20 =[16868,255,3,169,4161,42,7,28,527]

l_30 =[8861,144,8,131,4558,48,8,39,913]

l_50 =[7530,182,9,176,7503,127,12,129,2922]

l_70 =[2624,91,7,103,6058,127,10,130,4730]

l_90 =[588,43,4,38,2882,113,4,108,5920]

l_100 =[14,3,0,2,348,17,0,25,3296]

freqs = [0.001,0.002,0.003,0.005,0.007,0.01,0.02,
 0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,0.9,1.001]

names = [100*i for i in freqs]

everything = [l_01,l_02,l_03,l_05,l_07,l_1,l_2,l_3,
              l_5,l_7,l_10,l_20,l_30,l_50,l_70,l_90,l_100]

#%%
def get_rsq(data):
    a = []
    b = []
    
    for _ in range(data[0]):
        a.append(0)
        b.append(0)
    for _ in range(data[1]):
        a.append(0)
        b.append(1)
    for _ in range(data[2]):
        a.append(0)
        b.append(2)
    for _ in range(data[3]):
        a.append(1)
        b.append(0)
    for _ in range(data[4]):
        a.append(1)
        b.append(1)
    for _ in range(data[5]):
        a.append(1)
        b.append(2)
    for _ in range(data[6]):
        a.append(2)
        b.append(0)
    for _ in range(data[7]):
        a.append(2)
        b.append(1)
    for _ in range(data[8]):
        a.append(2)
        b.append(2)
    
    corr = np.corrcoef(a,b)
    
    return corr[0,1]

#%%
for i in range(len(everything)):
    print(f"{names[i]:.1f}",everything[i],f"{get_rsq(everything[i]):.3f}","     ",sum(everything[i])/5)
    