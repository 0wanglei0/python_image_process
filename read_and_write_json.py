import json
import re

pattern = re.compile('"deviceState":1')
f = open('device1.json', 'w')
with open('device.json') as data_file:
    for line in data_file:
        print(pattern.findall(line))
        result = pattern.findall(line)
        if result is None or result == []:
            f.write(line)
        else:
            print('device is updated')
data_file.close()
f.close()
