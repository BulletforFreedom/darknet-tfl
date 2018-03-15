import numpy as np
import os
import glob

import pdb

active_classes = ["beer","beverage","instantnoodle","redwine","snack","springwater","yogurt"]

trainSet_list = []
valSet_list = []
testSet_list = []

val_train_ratio = [1,9]

# --
wd = "/mnt/nas/tanfulun/Data/GoodsID"

raw_images_dir = os.path.join(wd,'images-raw')
save_dir = os.path.join(wd,'ImageSets','Main')

for cla in active_classes:
    img_dir = os.path.join(raw_images_dir,cla)
    paths = glob.glob(os.path.join(img_dir,'*.jpg'))
    imgNum = len(paths)
    if imgNum<10:
        valNum = 1
        trainNum = imgNum - valNum
    else:
        tot = sum(val_train_ratio)
        valNum = int(float(val_train_ratio[0])/tot*imgNum)
        #trainNum = int(float(val_train_ratio[1])/tot*imgNum)
	trainNum = imgNum-valNum

    #pdb.set_trace()
    
    print("Class %s, trainset:%d valset:%d tot:%d"%(cla,trainNum,valNum,imgNum))
    trainSet_list = trainSet_list + paths[0:trainNum]
    valSet_list = valSet_list + paths[trainNum:]

print("TrainNum:%d ,ValNum:%d"%(len(trainSet_list),len(valSet_list)))

#--save
trainf = open(os.path.join(save_dir,'train.txt'),'w')
for path in trainSet_list:
    trainf.write(path.split('/')[-1].strip()[0:-4]+'\n')
trainf.close()

valf = open(os.path.join(save_dir,'val.txt'),'w')
for path in valSet_list:
    valf.write(path.split('/')[-1].strip()[0:-4]+'\n')
valf.close()

