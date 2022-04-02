import xml.etree.ElementTree as ET
import numpy as np
import cv2
import random
import os

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
NUMBBOX = 2
NUMGRID = 7
DATASET_PATH = r'/home/zjl/code/Deep-Learning/YOLO/YOLOv1/VOC2012'
STATIC_DEBUG = False  # 调试用

"""将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
并进行归一化"""
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

"""把图像image_id的xml文件转换为目标检测的label文件(txt)
其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
并将四个物理量归一化"""
def convert_annotation(anno_dir, image_id, labels_dir):
    in_file = open(os.path.join(anno_dir,'Annotations/%s'%(image_id)))
    image_id = image_id.split('.')[0]
    tree = ET.parse(in_file)
    root = tree.getroot()
    '''获取xml中size，width和height'''
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text #比较难识别的，eval中不计
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text),float(xmlbox.find('xmax').text),
                  float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text))
        bb = convert((w,h),points)
        with open(os.path.join(labels_dir,'%s.txt'%(image_id)),'a') as out_file :
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
"""在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
def make_label_txt(anno_dir, labels_dir):
    filenames = os.listdir(os.path.join(anno_dir,'Annotations'))
    for file in filenames:
        convert_annotation(anno_dir,file,labels_dir)

def img_augument(img_dir,save_img_dir,labels_dir):
    imgs_list = [x.split('.')[0]+'.jpg' for x in os.listdir(labels_dir)]
    for img_name in imgs_list:
        print("process %s"%os.path.join(img_dir, img_name))
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w = img.shape[0:2]
        input_size = 448  # 输入YOLOv1网络的图像尺寸为448x448
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw,padh = 0,0
        if (h > w) :
            padw = (h-w) // 2
            #((0,0),(padw,padw),(0,0)) 指在第1维(列)（图片为3维，0，1，2）边界的数前后各加padw个0
            img = np.pad(img,((0,0),(padw,padw),(0,0)),'constant',constant_values=0)
        elif (w >h):
            padh = (w-h) // 2
            img = np.pad(img,((padh,padh),(0,0),(0,0)),'constant',constant_values=0)
        img = cv2.resize(img,(input_size,input_size))
        cv2.imwrite(os.path.join(save_img_dir, img_name), img)
        # resize了图片，对应的bbox也会改变，需要进行变化
        with open(os.path.join(labels_dir,img_name.split('.')[0] + ".txt"), 'r') as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox) % 5 != 0:
            raise ValueError("File:"
                             + os.path.join(labels_dir,img_name.split('.')[0] + ".txt") + "——bbox Extraction Error!")
        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据
        # ps bbox里均为归一化之后的参数
        if padw != 0:
            for i in range(len(bbox) // 5):
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
        elif padh != 0:
            for i in range(len(bbox) // 5):
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
        with open(os.path.join(labels_dir, img_name.split('.')[0] + ".txt"), 'w') as f:
            for i in range(len(bbox) // 5):
                bbox_temp = [str(x) for x in bbox[i*5:(i*5+5)]]
                str_context = " ".join(bbox_temp)+'\n'
                f.write(str_context)

"""将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
def convert_bbox2labels(bbox):
    gridsize = 1.0/NUMGRID
    labels = np.zeros((7,7,5*NUMBBOX+len(CLASSES)))  # 此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox)//5):
        gridx = int(bbox[i*5+1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i*5+2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx
        gridpy = bbox[i * 5 + 2] / gridsize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10+int(bbox[i*5])] = 1
    labels = labels.reshape(1, -1)
    return labels

"""
将img_dir文件夹内的图片按实际需要处理后，存入save_dir
最终得到图片文件夹及所有图片对应的标注(train.csv/test.csv)和图片列表文件(train.txt, test.txt)
"""
def create_csv_txt(img_dir, anno_dir, save_root_dir, train_val_ratio=0.9):
    labels_dir = os.path.join(anno_dir, "labels")
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)
    #labels xml2txt
    make_label_txt(anno_dir, labels_dir)
    print("labels done.")

    save_img_dir = os.path.join(os.path.join(anno_dir, "voc2012_forYolov1"), "img")
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    #resize img and bbox
    img_augument(img_dir, save_img_dir, labels_dir)
    #split train and val data
    imgs_list = os.listdir(save_img_dir)
    n_trainval = len(imgs_list)
    shuffle_id = list(range(n_trainval))
    random.shuffle(shuffle_id)
    n_train = int(n_trainval*train_val_ratio)
    train_id = shuffle_id[:n_train]
    test_id = shuffle_id[n_train:]

    traintxt = open(os.path.join(save_root_dir, "train.txt"), 'w')
    traincsv = np.zeros((n_train, NUMGRID*NUMGRID*(5*NUMBBOX+len(CLASSES))),dtype=np.float32)
    for i,id in enumerate(train_id):
        img_name = imgs_list[id]
        img_path = os.path.join(save_img_dir, img_name)+'\n'
        traintxt.write(img_path)
        with open(os.path.join(labels_dir,"%s.txt"%img_name.split('.')[0]), 'r') as f:
            bbox = [float(x) for x in f.read().split()]
            traincsv[i,:] = convert_bbox2labels(bbox)
    np.savetxt(os.path.join(save_root_dir, "train.csv"), traincsv)
    print("Create %d train data." % (n_train))

    testtxt = open(os.path.join(save_root_dir, "test.txt"), 'w')
    testcsv = np.zeros((n_trainval - n_train, NUMGRID*NUMGRID*(5*NUMBBOX+len(CLASSES))),dtype=np.float32)
    for i,id in enumerate(test_id):
        img_name = imgs_list[id]
        img_path = os.path.join(save_img_dir, img_name)+'\n'
        testtxt.write(img_path)
        with open(os.path.join(labels_dir,"%s.txt"%img_name.split('.')[0]), 'r') as f:
            bbox = [float(x) for x in f.read().split()]
            testcsv[i,:] = convert_bbox2labels(bbox)
    np.savetxt(os.path.join(save_root_dir, "test.csv"), testcsv)
    print("Create %d test data." % (n_trainval-n_train))


if __name__ == '__main__':
    random.seed(0)
    np.set_printoptions(threshold=np.inf)
    img_dir = os.path.join(DATASET_PATH, "JPEGImages")  # 原始图像文件夹
    anno_dirs = [DATASET_PATH]  # 标注文件
    save_dir = os.path.join(DATASET_PATH, "voc2012_forYolov1")  # 保存处理后的数据(图片+标签)的文件夹
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # 分别处理
    for anno_dir in anno_dirs:
        create_csv_txt(img_dir, anno_dir, save_dir)
