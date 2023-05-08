import matplotlib.pyplot as plt
import numpy as np
import os
import json
files = ['caffe_scale.json', 'convnext_scale.json' ,'eff_scale.json','fb_res_scale.json', 'augmix_scale.json',
         'eff_b0_scale.json', 'senet_scale.json', 'caffe_res101_scale.json', 'eff_b2_1k_scale.json',
         'convnext_1k_scale.json']
dir_name = '/net/cs-nfs/home/grad3/keisaito/general_det/tools/grad_evals'

for name in files:
    with open(os.path.join(dir_name, name), 'r') as f:
        data = json.load(f)
    print(len(data))
    #data =
    keys_data = np.array([i / len(data) for i in range(len(data))])
    #keys = np.array([int(val) / max(data.keys()) for val in data.keys()])
    values_data = [val for val in data.values()]
    values_data = [val for val in values_data]
    arch_name = name.split("_")[0]
    plt.plot(keys_data, values_data, label=arch_name)
    plt.legend(fontsize=18)
    plt.savefig("compare_scale_grad.png")
    values_data = values_data[:int(len(values_data)*0.5)]
    print(arch_name, sum(values_data) / len(values_data))
