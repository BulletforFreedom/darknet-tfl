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
wd = "/mnt/lvmhdd/tanfulun/workspaces/Data/GoodsID"

dataset_dir = os.path.join(wd,'ImageSets','Main')
image_dir = os.path.join(wd,'JPEGImages')
save_dir = wd

# --training set
train_set_file_path = os.path.join(dataset_dir,'train.txt')
f = open(train_set_file_path,'r')
name_list = f.readlines()
name_list = [x.strip() for x in name_list]
f.close()
#pdb.set_trace()
save_f = open(os.path.join(save_dir,'train.txt'),'w')
for name in name_list:
    save_f.write(image_dir+'/'+name+'.jpg'+'\n')
save_f.close()

# --val set
val_set_file_path = os.path.join(dataset_dir,'val.txt')
f = open(val_set_file_path,'r')
name_list = f.readlines()
name_list = [x.strip() for x in name_list]
f.close()
#pdb.set_trace()
save_f = open(os.path.join(save_dir,'val.txt'),'w')
for name in name_list:
    save_f.write(image_dir+'/'+name+'.jpg'+'\n')
save_f.close()




