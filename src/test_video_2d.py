import coremltools
import numpy as np
from PIL import Image, ImageDraw
import time
import cv2
import sys

#0: "top",              ---> 1
#1: "neck",             ---> 2, 8, 5, 11
#2: "R shoulder"        ---> 3
#3: "R elbow"           ---> 4
#4: "R wrist"
#5: "L shoulder"        ---> 6
#6: "L elbow"           ---> 7
#7: "L wrist"
#8: "R hip"             ---> 9
#9: "R knee"            ---> 10
#10: "R ankle"
#11: "L hip"            ---> 12
#12: "L knee"           ---> 13
#13: "L ankle"

connections = [[0, 1], [1, 2], [1, 5], [1, 8], [1, 11],
                [2, 3], [3, 4], [5, 6], [6, 7],
                [8, 9], [9, 10], [11, 12], [12, 13]]

DISPLAY = 1

enc_in = np.zeros((1, 64))
enc_in[0] = [0 for i in range(64)]

dt = 0.04
poses2d = []

if __name__ == '__main__':
    # Load the model
    model =  coremltools.models.MLModel("models/graph/mobilenet_thin/graph320x320.mlmodel")

    cap = cv2.VideoCapture(sys.argv[1])

    dim = 320
    out_dim = 40
    scale = 8
    key = "Openpose__concat_stage7__0"
    
    while cap.isOpened():
        _, img = cap.read()
        t = time.time()
        img = Image.fromarray(img)
        resized = img.resize((dim, dim), Image.LINEAR)
        predictions = model.predict({"image__0": resized}, False)
        
        preds = [np.mean(np.argwhere(predictions[key][i] > 0.1), axis=0)[::-1] for i in range(18)]
        preds_t = [tuple(int(scale * x) for x in pred) for pred in preds]
        _data = np.array([[preds_t[i][0], preds_t[i][1]] for i in range(18)]).flatten()
        
        d = ImageDraw.Draw(resized)
        
        for pair in connections:
            d.line([(_data[pair[0] * 2], _data[pair[0] * 2 + 1]),   (_data[pair[1] * 2], _data[pair[1] * 2 + 1])], fill=(0, 0, 255), width=2)

        print "Inference time: ", time.time() - t
        resized = np.array(resized)

        cv2.imshow("window", resized)
        cv2.waitKey(1)
