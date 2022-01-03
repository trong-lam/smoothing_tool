import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
if  __name__ == "__main__":
    path = os.path.join(os.getcwd(),'dataOfEffect.json')
    array = []
    img = cv2.imread(r'C:\Users\Admin\PycharmProjects\mocban_tool\static\uploads\img.png', 0)
    xcz,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    with open(path, 'r+') as jsonFile:
        # check if

        all_data = json.loads(jsonFile.read())
        print(json.dumps(all_data, indent=4, sort_keys=True))