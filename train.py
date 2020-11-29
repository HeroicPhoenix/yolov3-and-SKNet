from timeit import default_timer as timer

start = timer()
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from nets.yolo3 import yolo_body
from nets.loss import yolo_loss
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_random_data


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


# ---------------------------------------------------#
#   训练数据生成器
# ---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


# ---------------------------------------------------#
#   读入xml文件，并输出y_true
# ---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors,
                          num_classes):  # y_true就是一个真实图像中，它的每个真实框对应的(13,13)、(26,26)、(52,52)网格上的偏移位置、长宽与种类。其仍需要编码才能与y_pred的结构一致
    # 在yolo3中，其使用了一个专门的函数用于处理读取进来的图片的框的真实情况。
    # true_boxes：shape为(m, T, 5)代表m张图T个框的x_min、y_min、x_max、y_max、class_id。
    # input_shape：输入的形状，此处为416、416
    # anchors：代表9个先验框的大小
    # num_classes：种类的数量。
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors) // 3
    # 先验框
    # 678为116,90,  156,198,  373,326
    # 345为30,61,  62,45,  59,119
    # 012为10,13,  16,30,  33,23,  
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')  # 416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)    取框的真实值，获取其框的中心及其宽高，除去input_shape变成比例的模式。
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy / input_shape[:]
    true_boxes[..., 2:4] = boxes_wh / input_shape[:]

    # m张图
    m = true_boxes.shape[0]
    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       # 建立全为0的y_true，y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
                       dtype='float32') for l in range(num_layers)]
    # [1,9,2]   先验框维度扩增
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算真实框和哪个先验框最契合   对每一张图片处理，将每一张图片中的真实框的wh和先验框的wh对比，计算IOU值，选取其中IOU最高的一个，得到其所属特征层及其网格点的位置，在对应的y_true中将内容进行保存。
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 维度是(n) 感谢 消尽不死鸟 的提醒
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # floor用于向下取整
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    # 标签的位置
    annotation_path = '2007_train.txt'
    # 获取classes和anchor的位置
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    # ------------------------------------------------------#
    weights_path = 'model_data/yolo_weights.h5'  # 预训练好的模型的位置
    # 获得classes和anchor
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    # 一共有多少类
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # 训练后的模型保存的位置
    log_dir = 'logs2/'
    # 输入的shape大小
    input_shape = (416, 416)

    # 清除session
    K.clear_session()

    # 输入的图像为
    image_input = Input(shape=(416, 416, 3))
    h, w = input_shape

    # 创建yolo模型
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)  # yolo_body会输出三个特征层的内容

    # 载入预训练权重   迁移学习的思想
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # y_true为13,13,3,85
    # 26,26,3,85
    # 52,52,3,85
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    # 输入为*model_body.input, *y_true
    # 输出为model_loss
    loss_input = [*model_body.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)

    print(model.summary())
    print('total {} layers.'.format(len(model.layers)))

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False)  # 每过两个周期进行保存
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=2)  # 如果两次val_loss没有下降，lr就会下降
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
                                   verbose=2)  # val_loss一直不下降说明这个东西快过拟合了，要提前停止
    csvlogger = CSVLogger(log_dir + 'train.log')

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    model.compile(optimizer=Adam(lr=1e-4), loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 4
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors,
                                                       num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=100,
                        initial_epoch=0,
                        verbose=2,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping, csvlogger])
    model.save_weights(log_dir + 'last1.h5')

end = timer()
print(end - start)
