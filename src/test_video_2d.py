import coremltools
import numpy as np
from PIL import Image, ImageDraw
import time
import matplotlib.pyplot as plt
import cv2
import sys


try:
    from pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess')
    exit(-1)
    

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

DISPLAY = 1

connections = [[0, 1], [1, 2], [1, 5], [1, 8], [1, 11],
                [2, 3], [3, 4], [5, 6], [6, 7],
                [8, 9], [9, 10], [11, 12], [12, 13]]

if __name__ == '__main__':
    
    frame = 0
    
    # Load the model
    model =  coremltools.models.MLModel("/Users/Melanie/tf-pose-estimation/models/graph/mobilenet_thin/graph640.mlmodel")

    cap = cv2.VideoCapture(sys.argv[1])
#    cap = cv2.VideoCapture(0)
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    dim = 640
    out_dim = 80
    scale = 8
    key = "Openpose__concat_stage7__0"
    
    p_top, p_neck = 0, 0
    p_rShoulder, p_lShoulder = 0, 0
    p_rElbow, p_lElbow= 0, 0
    p_rWrist, p_lWrist = 0, 0
    p_rHip, p_lHip = 0, 0
    p_rKnee, p_lKnee = 0, 0
    p_rAnkle, p_lAnkle = 0, 0
    
    plt.figure()
    while cap.isOpened():
        _, img = cap.read()
        t = time.time()
        img = Image.fromarray(img)
        resized = img.resize((dim, dim), Image.LINEAR)
        predictions = model.predict({"image__0": resized}, False)
        
        heatMat2 = predictions[key][:19,:,:]
        heatMat = np.zeros((80, 80, 19))
        for i in range(19):
            heatMat[:,:, i] = heatMat2[i,:,:]
        pafMat = np.zeros((80, 80, 38))
        pafMat2 = predictions[key][19:,:,:]
        for i in range(38):
            pafMat[:,:, i] = pafMat2[i,:,:]
        
        
        print time.time() - t
        
        preds = [np.mean(np.argwhere(heatMat[:,:,i] > 0.1), axis=0)[::-1] for i in range(19)]
        preds_t = []
        for pred in preds:
            try:
                preds_t.append(tuple(int(x) for x in pred))
            except:
                preds_t.append((0, 0))

        peaks = np.zeros((80, 80, 19))
        for i in range(19):
            peaks[preds_t[i][1], preds_t[i][0], i] = heatMat[preds_t[i][1], preds_t[i][0], i]
        
        pafprocess.process_paf(peaks.astype('float32'), heatMat.astype('float32'), pafMat.astype('float32'))
        
        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            body_parts = [(0.0, 0.0)] * 18
            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                body_parts[part_idx] = (pafprocess.get_part_x(c_idx),
                                        pafprocess.get_part_y(c_idx))
            body_parts = np.array([[body_parts[i][0] * scale, body_parts[i][1] * scale] for i in range(18)]).flatten()
            humans.append(body_parts)
        
        d = ImageDraw.Draw(resized)
        
        for _data in humans:
            for pair in connections:
                if (_data[pair[0] * 2], _data[pair[0] * 2 + 1]) != (0, 0) and (_data[pair[1] * 2], _data[pair[1] * 2 + 1]) != (0, 0):
                    d.line([(_data[pair[0] * 2], _data[pair[0] * 2 + 1]), (_data[pair[1] * 2], _data[pair[1] * 2 + 1])], fill=(255, 255, 255), width=2)

        print time.time() - t
        
        if DISPLAY:
            resized = np.array(resized)
    
            plt.subplot(131)
            plt.imshow(np.sum(heatMat[:,:,i] for i in range(19)))
            plt.subplot(132)
            plt.imshow(np.sum(pafMat[:,:,i] for i in range(19)))
            plt.subplot(133)
            plt.imshow(np.sum(peaks[:,:,i] for i in range(19)))
            plt.draw()
            plt.pause(0.001)
    
            cv2.imshow("result", resized)
            cv2.waitKey(1)
                    

