
import os
map_list = os.listdir('maps')
try:
    idx = map_list.index('100.png')
    print(f"Index of 100.png: {idx}")
    print(f"Next map: {map_list[(idx+1)%len(map_list)]}")
except ValueError:
    print("100.png not found")
