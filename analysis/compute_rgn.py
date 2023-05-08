import sys
import json
name = sys.argv[1]

with open(name, 'r') as f:
    data = json.load(f)

all_rgn = 0
count = 0
for key, value in data.items():
    if 'se_module' not in key:
        all_rgn += value
        count +=1
print('mean rgn ' + str(all_rgn / count))