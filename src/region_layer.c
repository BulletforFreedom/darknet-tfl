#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = {0};
    l.type = REGION;

    l.n = n;                                            // anchor的个数（一个cell多少个box）
    l.batch = batch;                                    // batch_size(one gpu on forward batch)
    l.h = h;                                            // input feature height
    l.w = w;                                            // input feature width
    l.c = n*(classes + coords + 1);                     // 输入通道数= anchor数*(类别+(tx,ty,tw,th)+置信度(t0))
    l.out_w = l.w;                                      // 输入的 W,H,C 与输出的一致，该层不改变数据的结构
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;                                // 类别数
    l.coords = coords;                                  // 指（tx,ty,tw,th）
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));              // box长宽的偏置; 为anchor boxes的宽和高赋初值(论文中表述为prior)
    l.bias_updates = calloc(n*2, sizeof(float));        // box长宽偏置的更新值
    l.outputs = h*w*n*(classes + coords + 1);           // feature-map大小 * anchor * 每个anchor对应的参数
    l.inputs = l.outputs;
    l.truths = 30*(l.coords + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

/**
 * @details 获取每个 grid cell 的 anchor boxes
 *
 *          [i,j] 是 grid cell 的左上角位置坐标
 *
 * @param x: l.output
 * @param biases: l.biars
 * @param n: anchor type(default 5)
 * @param index: boxes 参数在 l.output 中的起始位置
 * @param i: col of grid cell
 * @param j: row of grid cell
 * @param w: width of grid cell
 * @param h: height of grid cell
 * @param stride: stride of boxes params [x,y,w,h]，即 l.w*l.h
 *
 * */
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}


void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        if (delta[index] && tag){
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);
            return;
        }
        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            if(n == class) *avg_cat += output[index + stride*n];
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

/**
 * @details 会在 forward_region_layer 中反复调用
 *          用于确定需要更新的参数的位置
 *
 *          可以推出 l.output 中存储的数据结构如下：
 *
 *          #batch0#-&anchor type 0&-[xxx-yyy-www-hhh-ccc-C0C0C0-C1C1C1-C2C2C2]-&anchor type 1&-......#batch1#... ...
 *          其中[x,y,w,h,c,C0,... ...,CN]每个元素都有 l.w*l.h 个
 *
 *
 * @param l: type layer(一个 region_layer实例)
 * @param batch: batch_size
 * @param location:
 * @param entry: classes coord conf 中的一个
 *
 * @return entry position
 * */
int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);     /// 计算是属于哪类anchor（共5类）
    int loc = location % (l.w*l.h);     /// 在当前anchor内的偏移量
    return batch*l.outputs + n*l.w*l.h*(l.coords+l.classes+1) + entry*l.w*l.h + loc;
}

/**
 * @details region_layer 的前向，即使在GPU模式下也会调用；
 *          输出BB的预测结果，
 *
 *
 * */
void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    /// 将net.input中的元素全部拷贝至l.output中; net.input中存储着上一层的输出;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    /// 遍历一个batch中的所有图片，对 dx，dy，conf，classes 做 logistic_regressio
    for (b = 0; b < l.batch; ++b){
        /// 遍历每种anchor(一共 n 种类型的anchor)
        for(n = 0; n < l.n; ++n){
            /// entry_index的第三个输入参数为"n*l.w*l.h"，目的是保证格式的统一；
            /// 其实有效数据只有第三个参数中的 n，以及第四个参数
            /// 这一步，找到[dx,dy,dw,dh]的起始位置，[dx,dy,dw,dh]在数组最前面，所以偏移量为0；
            int index = entry_index(l, b, n*l.w*l.h, 0);
            /// 第三个参数为 2*l.w*l.h，表示只对 dx 和 dy 做logistic的操作
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            /// 找到 confident 的起始位置(前面为[dx,dy,dw,dh]，所以 conf 偏移量为l.coords);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array(l.output + index,   l.w*l.h, LOGISTIC);
            /// 找到 classes 的起始位置(前面为[dx,dy,dw,dh,confident]，所以偏移量为l.coords+1);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int i;
        int count = l.coords + 1;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    } else if (l.softmax){
        /// =====计算分类的输出(SOFTMAX)=====
        /// l.background 默认为0，!l.background = 1
        /// l.coords + !l.background 使得指针起始位置指向 "classes"
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        /// ==softmax_cpu 输入解析==
        /// 目的是计算一个 grid cell 的所有 classes 的 softmax
        /// 1.输入起始位置(input):net.input + index；已经指向classes的第一个元素
        /// 2.每次softmax元素个数(n):l.classes + l.background；
        /// 3.进行softmax运算次数(batch):l.batch*l.n；作者把一个anchor_type的数据作为一个大batch
        /// 4.大batch之间的间隔(batch_offset):其实就是两个anchor_type数据之间的间隔 l.inputs/l.n
        /// 5.(group):针对于一个anchor_type，有 l.w*l.h 个 grid cell
        /// 6.(group_offset):1
        /// 7.softmax运算相邻元素间隔(stride): 同一个 cell 不同 classes 之间的间隔 = l.w*l.h；就是 grid cells 的个数
        /// 8.(temp):
        /// 9.输出起始位置(output):l.output + index
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }
#endif

    /// 为 l.delta 赋初值 0
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;   // 当前图片是否检测到物体
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    /// 对于特定的anchor_type 遍历所有的 grid cell
                    /// 每次得到一个 grid cell 起始位置的索引
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    /// 获取当前grid cell，当前anchor_type的 BB 预测结果
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    /// 当前 pred 与 GT 的最大IOU
                    float best_iou = 0;
                    /// "30"这个参数,限制了一张输入图片最多有30个GT_BB
                    /// TODO: 确认一下，在label数据输入的时候是否强制最多有30个BB的输入！！！！
                    for(t = 0; t < 30; ++t){
                        /// net.truth是指向 label 的指针
                        /// 第一个参数为GT中，每个BB的起始位置；第二个参数为BB中每个参数的stride；
                        box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                        /// 如果发现truth的值为空，则跳出
                        if(!truth.x) break;
                        /// 将每个 preds 与当前 truth 比较，获取最大IOU，看是否超出阈值；判定前景背景；
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    /// 获取 IOU_confidence,论文中表述为:Pr(object)*IOU(b,object);作为前景背景判断依据;
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
                    avg_anyobj += l.output[obj_index];
                    ///
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    /// l.thresh 在 .cfg 配置文件中定义;default=0.6
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }
                    /// 当输入的图片数量(net.seen) < 12800(一个batch为64的话，大概10个batch);固定truth的值为 grid cell 的大小
                    /// 估计这步是在batch比较小的时候加快网络收敛速度的trick
                    if(*(net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;     // 归一化 grid cell 中心点 x
                        truth.y = (j + .5)/l.h;     // 归一化 grid cell 中心点 y
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
            }
        }
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = 0; n < l.n; ++n){
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);

            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            if(l.coords > 4){
                int mask_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, mask_index, l.delta, l.w*l.h, l.mask_scale);
            }
            if(iou > .5) recall += 1;
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            avg_obj += l.output[obj_index];
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
            if (l.rescore) {
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
            }
            if(l.background){
                l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
            }

            int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
            if (l.map) class = l.map[class];
            int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}

void get_region_boxes(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, float **masks, int only_objectness, int *map, float tree_thresh, int relative)
{
    int i,j,n,z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w/2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for(z = 0; z < l.classes + l.coords + 1; ++z){
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                probs[index][j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            if(masks){
                for(j = 0; j < l.coords - 4; ++j){
                    masks[index][j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    probs[index][j] = (scale > thresh) ? scale : 0;
                    probs[index][l.classes] = scale;
                }
            } else {
                float max = 0;
                for(j = 0; j < l.classes; ++j){
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                    float prob = scale*predictions[class_index];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                    if(prob > max) max = prob;
                    // TODO REMOVE
                    // if (j == 56 ) probs[index][j] = 0; 
                    /*
                       if (j != 0) probs[index][j] = 0; 
                       int blacklist[] = {121, 497, 482, 504, 122, 518,481, 418, 542, 491, 914, 478, 120, 510,500};
                       int bb;
                       for (bb = 0; bb < sizeof(blacklist)/sizeof(int); ++bb){
                       if(index == blacklist[bb]) probs[index][j] = 0;
                       }
                     */
                }
                probs[index][l.classes] = max;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
    correct_region_boxes(boxes, l.w*l.h*l.n, w, h, netw, neth, relative);
}

#ifdef GPU

void forward_region_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            if(l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
            }
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) activate_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if(!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
    /*
        int mmin = 9000;
        int mmax = 0;
        int i;
        for(i = 0; i < l.softmax_tree->groups; ++i){
            int group_size = l.softmax_tree->group_size[i];
            if (group_size < mmin) mmin = group_size;
            if (group_size > mmax) mmax = group_size;
        }
        //printf("%d %d %d \n", l.softmax_tree->groups, mmin, mmax);
        */
        /*
        // TIMING CODE
        int zz;
        int number = 1000;
        int count = 0;
        int i;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
        int group_size = l.softmax_tree->group_size[i];
        count += group_size;
        }
        printf("%d %d\n", l.softmax_tree->groups, count);
        {
        double then = what_time_is_it_now();
        for(zz = 0; zz < number; ++zz){
        int index = entry_index(l, 0, 0, 5);
        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
        }
        cudaDeviceSynchronize();
        printf("Good GPU Timing: %f\n", what_time_is_it_now() - then);
        } 
        {
        double then = what_time_is_it_now();
        for(zz = 0; zz < number; ++zz){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
        int group_size = l.softmax_tree->group_size[i];
        int index = entry_index(l, 0, 0, count);
        softmax_gpu(net.input_gpu + index, group_size, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
        count += group_size;
        }
        }
        cudaDeviceSynchronize();
        printf("Bad GPU Timing: %f\n", what_time_is_it_now() - then);
        }
        {
        double then = what_time_is_it_now();
        for(zz = 0; zz < number; ++zz){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
        int group_size = l.softmax_tree->group_size[i];
        softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
        count += group_size;
        }
        }
        cudaDeviceSynchronize();
        printf("CPU Timing: %f\n", what_time_is_it_now() - then);
        }
         */
        /*
           int i;
           int count = 5;
           for (i = 0; i < l.softmax_tree->groups; ++i) {
           int group_size = l.softmax_tree->group_size[i];
           int index = entry_index(l, 0, 0, count);
           softmax_gpu(net.input_gpu + index, group_size, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
           count += group_size;
           }
         */
    } else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        //printf("%d\n", index);
        softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_layer(l, net);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_region_layer_gpu(const layer l, network net)
{
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            if(l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            }
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if(!l.background) gradient_array_gpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
    }
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}

