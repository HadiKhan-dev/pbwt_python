import matplotlib.pyplot as plt
import numpy as np

with_div = {2: 0.0238791, 3: 0.0393664, 5: 0.0414488, 7: 0.0398990,
10: 0.0338448, 20: 0.0423533, 30: 0.0410647, 50: 0.0395999,
70: 0.0374253, 90: 0.0355451, 100: 0.0304245}

without_div = {2: 0.1069967, 3: 0.1077872, 5: 0.0912748, 7: 0.0905808,
10: 0.0892069, 20: 0.0972171, 30: 0.0960300, 50: 0.0953299,
70: 0.0855385,90: 0.0755117, 100: 0.0673437}

fig,ax = plt.subplots()

with_list = []
without_list = []
names = []

width = 0.4
for name in with_div.keys():
    names.append(str(name))
    with_list.append(with_div[name])
    without_list.append(without_div[name])
    
ax.bar(range(len(names)),without_list,color="r",width=width,label="Without Divergence Info")

ax.bar(np.array(range(len(names)))+width,with_list,width=width,label="With Divergence Info")

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.legend()
ax.set_title("RMSE of XGBoost model on test data")

plt.show()