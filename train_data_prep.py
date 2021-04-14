'''
https://towardsdatascience.com/prototyping-an-anomaly-detection-system-for-videos-step-by-step-using-lstm-convolutional-4e06b7dcdd29
^^Thanks to the author^^
'''

from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image
import torch
traindata_path = 'UCSDped1/Train'
testdata_path = 'UCSDped1/Test'

strides = [1,2,3]
train_id = 0

for f in sorted(listdir(traindata_path)):
    
    dir_path = join(traindata_path, f)
    if isdir(dir_path):
        #strides in the paper are  1,2,3
        video = []
        print('train id is ', train_id)
        for frame in sorted(listdir(dir_path)):
            
            img_path = join(dir_path,frame)
            
            if str(img_path)[-3:] == "tif":
                
                img = Image.open(img_path).resize((100, 100))
                img = np.array(img, dtype=np.float32) / 255.0
                img = np.expand_dims(img,axis=0)

                video.append(img)
        
        video_len = len(video)
        cnt = 0
        for stride in strides:
            for start in range(0,stride):
                seq = np.zeros((10,1,100,100))
                for ind in range(start, video_len, stride):
                    
                    seq[cnt,:,:,:] = video[ind]     
                    cnt += 1
                    if cnt==10:
                        data_point = 'data/train/id-'+ str(train_id) + '.pt'
                        torch.save(torch.tensor(seq,dtype=torch.float32),data_point)
                        cnt = 0
                        train_id += 1

val_id = 0

for f in sorted(listdir(testdata_path)):
    
    dir_path = join(testdata_path, f)
        
    if isdir(dir_path):
        video = []
        print('val id is ', val_id)

        for frame in sorted(listdir(dir_path)):

            if str(img_path)[-3:] == "tif":
            
                img = Image.open(img_path).resize((100, 100))
                img = np.array(img, dtype=np.float32) / 255.0
                img = np.expand_dims(img,axis=0)

                video.append(img)
    video_len = len(video)
    cnt = 0

    seq = np.zeros((10,1,100,100))
    for ind in range(video_len):
        seq[cnt,:,:,:] = video[ind]     
        cnt += 1
        if cnt==10:
            data_point = 'data/val/id-'+ str(val_id) + '.pt'
            torch.save(torch.tensor(seq,dtype=torch.float32),data_point)
            cnt = 0
            val_id += 1
    
