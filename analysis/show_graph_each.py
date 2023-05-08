import matplotlib.pyplot as plt
import numpy as np
import os
import json
files = ['fb_res_scale.json']#['caffe_scale.json', 'convnext_scale.json' ,'eff_scale.json','fb_res_scale.json']
dir_name = '/net/cs-nfs/home/grad3/keisaito/general_det/tools/grad_evals'

for name in files:
    with open(os.path.join(dir_name, name), 'r') as f:
        data = json.load(f)
    print(len(data))
    keys_data = np.array([i / len(data) for i in range(len(data))])
    #keys = np.array([int(val) / max(data.keys()) for val in data.keys()])
    values_data = [val for val in data.values()]
    values_data = [val / max(values_data) for val in values_data]
    arch_name = name.split("_")[0]
    plt.plot(keys_data, values_data, label=arch_name)
    plt.legend(fontsize=18)
    plt.savefig("scale_{}.png".format(arch_name))
