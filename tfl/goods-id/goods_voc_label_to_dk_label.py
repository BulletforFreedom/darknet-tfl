#coding: utf-8
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
import json

import pdb

sets=[('2012', 'train'),('2018','val')]

classes = ["beer","beverage","instantnoodle","redwine","snack","springwater","yogurt"]
classes_ch = ["啤酒","饮料","泡面","红酒","零食","矿泉水","酸奶"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(workingdir,voc_xml_path, image_id):
    in_file = open(voc_xml_path)
    out_file = open('%s/dk_labels/%s.txt'%(workingdir, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    #pdb.set_trace()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        
        cls_utf = cls.encode('utf-8')
        if cls_utf not in classes_ch or int(difficult)==1:
            continue
        cls_id = classes_ch.index(cls_utf)
       
        print ("%s  %s %s"%(cls,image_id,cls_id))

        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

# VOC data abs dir
wd = '/mnt/nas/tanfulun/Data/GoodsID'

dk_label_abs_dir = os.path.join(wd,'dk_labels')
if not os.path.exists(dk_label_abs_dir):
    os.makedirs(dk_label_abs_dir)

voc_xml_dir = os.path.join(wd,"Annotations")
voc_xml_paths = glob.glob(os.path.join(voc_xml_dir,'*.xml'))
#voc_xml_paths = [voc_xml_dir+'/frames_00057.xml',voc_xml_dir+'/frames_00155.xml',voc_xml_dir+'/frames_00026.xml',voc_xml_dir+'/frames_00555.xml',voc_xml_dir+'/frames_00001.xml',voc_xml_dir+'/frames_00306.xml',voc_xml_dir+'/frames_00046.xml',]

image_dir = os.path.join(wd,"JPEGImages")
image_paths = glob.glob(os.path.join(image_dir,'*.jpg'))

#diff between .jpg .xml
diff=False

jpg_names = []
xml_names = []
miss = []
for imgpath in image_paths:
    name = imgpath.split('/')[-1].strip()[0:-4]
    jpg_names.append(name)
for xml in voc_xml_paths:
    xml_name = xml.split('/')[-1].strip()[0:-4]
    xml_names.append(xml_name)

for i in jpg_names:
    if i not in xml_names:
        miss.append(i)
        diff = True
if diff:
    print("file missing!")
    print miss
    exit()
else:
    pass

#pdb.set_trace()

for year,dataset in sets:
    set_path = os.path.join(wd,'ImageSets/Main/%s.txt'%dataset)
    infile = open(set_path,'r')
    names = infile.read().strip().split('\n')
    infile.close()

    outfile = open(os.path.join(wd,'%s.txt'%dataset),'w')
    for name in names:
        if name in jpg_names:
            img_path = os.path.join(wd,'JPEGImages/%s.jpg'%name)
            outfile.write(img_path+'\n')
        else:
            print("file missing in JPEGImages")
            print name
            exit()
        voc_xml_path = os.path.join(wd,'Annotations/%s.xml'%name)
        convert_annotation(wd,voc_xml_path,name)

    outfile.close()
    #pdb.set_trace()
'''
for path in voc_xml_paths:
    image_id = path.split('/')[-1].strip()[0:-4]
    #pdb.set_trace()
    convert_annotation(wd, path, image_id)
    #pdb.set_trace()
'''
# dataset spliting
#os.chdir(wd)
#os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > %s/train.txt")
#os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

