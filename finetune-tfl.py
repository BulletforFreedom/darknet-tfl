import os
import time

training_argments = []

restart_training = True

dataset_conf = 'tfl/goods-id/cfg/goodid.data'

# finetune from darknet19-coco
#model_struct_path = 'tfl/goods-id/cfg/yolo.cfg'
#model_weights_path = 'pretrain_models/yolo.weights.30'

# finetune from darknet19-imagenet12
#model_struct_path = 'tfl/goods-id/cfg/yolo-voc.cfg'
#model_weights_path = 'pretrain_models/darknet19_448.conv.23'

# finetune from darknet19-imagenet12-608
model_struct_path = 'tfl/goods-id/cfg/yolo-voc-800.cfg'
model_weights_path = 'pretrain_models/darknet19_448.conv.23'

#start-training-time
print time.ctime()
stime = time.time()

if not restart_training:
    #os.system('./darknet detector train'+' '+dataset_conf+' '+model_struct_path+' '+model_weights_path+' -gpus 0,3')
    #os.system('nohup ./darknet detector train'+' '+dataset_conf+' '+model_struct_path+' '+model_weights_path+' -gpus 0,3'+' >goodsid.log 2>&1')
    os.system('./darknet detector train'+' '+dataset_conf+' '+model_struct_path+' '+model_weights_path+' -gpus 0,3')
else:
    suffix = '20000'
    model_weights_path = 'tfl/goods-id/backup/yolo-voc-800_20000.weights'
    os.system('nohup ./darknet detector train'+' '+dataset_conf+' '+model_struct_path+' '+model_weights_path+' -gpus 0,3'+' >goodsid%s.log 2>&1'%suffix)
    #os.system('./darknet detector train'+' '+dataset_conf+' '+model_struct_path+' '+model_weights_path+' -gpus 0,3')

#end-training-time
print time.ctime()
etime = time.time()

# Training Time
dur_sec = etime-stime

hour = int(dur_sec/3600)
minute = int( (dur_sec-3600*hour)/60 )

print ('%d h, %d m'%(hour,minute))

