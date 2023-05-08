import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.stats import pearsonr

class Data_Var(object):
    def __init__(self, ood_up, id_up, var_values):
        self.ood_up = ood_up#[:1]
        self.id_up = id_up
        self.var_values = var_values
data_dict = {
'swsl_res50': Data_Var([-6.6, -0.2, -2.4], 8.0, 1.34),
'caffe_res50': Data_Var([-0.4, 2.3, -1.1], 8.9, 0.299),
'eff_b2_jft': Data_Var([4.5, 6.8, 2.9], 13.2,0.06),
'convnext': Data_Var([-0.2, 5.6, 2.9], 20.9,0.130),
'senet': Data_Var([0.8181, -0.0661,2.1343], 12.3098, 0.032),
'caffe_res101': Data_Var([1.1172,8.2131,1.8408] ,10.3422, 0.302),
'augmix': Data_Var([0.3287,7.131,5.2063], 15.0947, 0.574),
'convnext_1k': Data_Var([1.3724,4.991,1.3437], 20.3526, 0.115),
'eff_b0': Data_Var([0.7185,8.0014,5.7382], 10.5, 0.065),
'eff_b2_1k':Data_Var([3.1841,8.090,6.0189], 13.9293, 0.055),}

data_names = ['comic', 'water', 'clipart']

fig, ax = plt.subplots()
dict_corr = {0: [], 1: [], 2:[]}
for key, value in data_dict.items():
    x_val = [value.var_values] * len(value.ood_up)
    ood_over_id = [ood / value.id_up for ood in value.ood_up]
    plt.scatter(x_val, ood_over_id, label=key)
    for i in range(len(ood_over_id)):
        ax.annotate(data_names[i], (x_val[i], ood_over_id[i]))
        dict_corr[i].append([x_val[i], ood_over_id[i]])
print([val for val in dict_corr[0]], [val for val in dict_corr[0][1]])
corr, _ = pearsonr([val[0] for val in dict_corr[0]], [val[1] for val in dict_corr[0]])
print('corr comic', corr)
corr, _ = pearsonr([val[0] for val in dict_corr[1]], [val[1] for val in dict_corr[1]])
print('corr water', corr)
corr, _ = pearsonr([val[0] for val in dict_corr[2]], [val[1] for val in dict_corr[2]])
print('corr clipart', corr)
plt.legend(fontsize=14)
plt.xlabel(" |gradient| / |weight| ", fontsize=18)
plt.ylabel("(OOD_f - OOD_p) / (ID_f - ID_p) ", fontsize=18)
plt.savefig("vis_id_vs_ood.png")



#plt.c
plt.clf()
data_dict = {
'fbnet': Data_Var([-6.6, -0.2, -2.4], 8.0, 1.79),
'caffe': Data_Var([-0.4, 2.3, -1.1], 8.9, 0.36),
'efnet': Data_Var([4.5, 6.8, 2.9], 13.2,0.08),
'convnext': Data_Var([-0.2, 5.6, 2.9], 20.9,0.237),
'senet': Data_Var([0.8181, -0.0661,2.1343], 12.3098, 0.041),
'caffe_res101': Data_Var([1.1172,8.2131,1.8408] ,10.3422, 0.345),
'augmix': Data_Var([0.3287,7.131,5.2063], 15.0947, 0.755),
'convnext_1k': Data_Var([1.3724,4.991,1.3437], 20.3526, 0.21),
'eff_b0': Data_Var([0.7185,8.0014,5.7382], 10.5, 0.085),
'eff_b2_1k':Data_Var([3.1841,8.090,6.0189], 13.9293, 0.064)}

data_names = ['comic', 'water', 'clipart']
#
# for key, value in data_dict.items():
#     x_val = [value.id_up] * 3
#     plt.scatter(x_val, value.ood_up, label=key)
# plt.legend(fontsize=18)
# plt.savefig("vis_id_vs_ood.png")
#
fig, ax = plt.subplots()
#ax.scatter(z, y)

for key, value in data_dict.items():
    x_val = [value.var_values] * len(value.ood_up)
    ood_over_id = [ood / value.id_up for ood in value.ood_up]
    plt.scatter(x_val, ood_over_id, label=key)
    for i in range(len(ood_over_id)):
        ax.annotate(data_names[i], (x_val[i], ood_over_id[i]))
        dict_corr[i].append([x_val[i], ood_over_id[i]])
print([val for val in dict_corr[0]], [val for val in dict_corr[0][1]])
corr, _ = pearsonr([val[0] for val in dict_corr[0]], [val[1] for val in dict_corr[0]])
print('corr comic', corr)
corr, _ = pearsonr([val[0] for val in dict_corr[1]], [val[1] for val in dict_corr[1]])
print('corr water', corr)
corr, _ = pearsonr([val[0] for val in dict_corr[2]], [val[1] for val in dict_corr[2]])
print('corr clipart', corr)

#plt.legend(fontsize=18)
plt.xlabel(" |gradient| / |weight| ", fontsize=18)
plt.ylabel("(OOD_f - OOD_p) / (ID_f - ID_p) ", fontsize=18)
plt.savefig("vis_id_vs_ood_v2.png")

plt.clf()

data_dict = {
'swsl_res50': Data_Var([-6.6, -0.2, -2.4], 8.0, 0.007022742741918634),
'caffe_res50': Data_Var([-0.4, 2.3, -1.1], 8.9, 0.0022593106619925017),
'eff_b2_jft': Data_Var([4.5, 6.8, 2.9], 13.2, 0.0041901809468383785),
'convnext': Data_Var([-0.2, 5.6, 2.9], 20.9, 0.009013429228383576),
'senet': Data_Var([0.8181, -0.0661,2.1343], 12.3098, 0.0006001424502601173),
'caffe_res101': Data_Var([1.1172,8.2131,1.8408] ,10.3422, 0.0014620978535790725),
'augmix': Data_Var([0.3287,7.131,5.2063], 15.0947, 0.00308630621570739),
'convnext_1k': Data_Var([1.3724,4.991,1.3437], 20.3526, 0.008225915161159163),
'eff_b0': Data_Var([0.7185,8.0014,5.7382], 10.5, 0.006416655001745452),
'eff_b2_1k':Data_Var([3.1841,8.090,6.0189], 13.9293, 0.00308630621570739),}

dict_corr = {0: [], 1: [], 2:[]}
for key, value in data_dict.items():
    x_val = [value.var_values] * len(value.ood_up)
    ood_over_id = [ood / value.id_up for ood in value.ood_up]
    plt.scatter(x_val, ood_over_id, label=key)
    for i in range(len(ood_over_id)):
        ax.annotate(data_names[i], (x_val[i], ood_over_id[i]))
        dict_corr[i].append([x_val[i], ood_over_id[i]])
print([val for val in dict_corr[0]], [val for val in dict_corr[0][1]])
corr, _ = pearsonr([val[0] for val in dict_corr[0]], [val[1] for val in dict_corr[0]])
print('corr comic', corr)
corr, _ = pearsonr([val[0] for val in dict_corr[1]], [val[1] for val in dict_corr[1]])
print('corr water', corr)
corr, _ = pearsonr([val[0] for val in dict_corr[2]], [val[1] for val in dict_corr[2]])
print('corr clipart', corr)
plt.legend(fontsize=14)
plt.xlabel(" |gradient| / |weight| ", fontsize=18)
plt.ylabel("(OOD_f - OOD_p) / (ID_f - ID_p) ", fontsize=18)
plt.savefig("vis_id_vs_ood_unscale_grad.png")
