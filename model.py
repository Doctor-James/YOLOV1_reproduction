import torch
import torch.nn as nn
import torchvision.models as tvmodel
from prepare_data import CLASSES, NUMBBOX, NUMGRID
#from util import calculate_iou

"""计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
def calculate_iou(bbox1,bbox2):
    if bbox1[2]<=bbox1[0] or bbox1[3]<=bbox1[1] or bbox2[2]<=bbox2[0] or bbox2[3]<=bbox2[1]:
        return 0  #error
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的重合区域的(x1,y1,x2,y2)
    intersect_bbox[0] = max(bbox1[0],bbox2[0])
    intersect_bbox[1] = max(bbox1[1], bbox2[1])
    intersect_bbox[2] = min(bbox1[2], bbox2[2])
    intersect_bbox[3] = min(bbox1[3], bbox2[3])
    width = max(intersect_bbox[2] - intersect_bbox[0],0)
    height = max(intersect_bbox[3] - intersect_bbox[1], 0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    area_intersect = width * height  # 交集面积
    iou = area_intersect / (area1 + area2 - area_intersect + 1e-6)  # 防止除0
    return iou

class YOLOV1_Net(nn.Module):
    def __init__(self):
        super(YOLOV1_Net,self).__init__()
        # 此处做一点修改，用resnet34的预训练模型来做特征提取，代替源论文中的20个卷积层，省去了训练特征提取层的时间
        resnet = tvmodel.resnet34(pretrained=True) # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel,1024,3,padding=1),
            nn.BatchNorm2d(1024), # 为了加快训练，在激活函数前增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024,1024,3,padding=1,stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True)
        )
        self.Conn_layers = nn.Sequential(
            nn.Linear(NUMGRID*NUMGRID*1024,4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096,NUMGRID*NUMGRID*(5*NUMBBOX+len(CLASSES))),
            # nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )
    def forward(self,inputs):
        x = self.resnet(inputs)
        x = self.Conv_layers(x)
        # label数据经过pytorch的toTensor()函数转换后，数据会变成 batchsize × 30 × 7 × 7，所以网络的输出也应当对应改成 batchsize × 30 × 7 × 7
        # 中间有参数-1，这表明对out进行reshape，reshape成x.size()[0]行，列数由行数决定,x.size()[0]为batchsize
        # batchsize*7*7*1024 -> batchsize*50176
        x = x.view(x.size()[0], -1)
        x = self.Conn_layers(x)
        self.pred = x.reshape(-1, (5 * NUMBBOX + len(CLASSES)), NUMGRID, NUMGRID)  # reshape一下输出数据
        return self.pred
    def cal_loss(self,labels):
        self.pred = self.pred.double()
        labels = labels.double()
        num_gridx, num_gridy = NUMGRID, NUMGRID  # 划分网格数量
        noobj_confi_loss = 0.  # 不负责检验物体的bbox的confidence误差（iou之差）
        coor_loss = 0.  # 负责检验物体的bbox的定位误差（包含中心点坐标，宽高误差）
        obj_confi_loss = 0.  # 负责检验物体的bbox的confidence误差（iou之差）
        class_loss = 0.  # 负责检验物体的grid cell分类误差
        n_batch = labels.size()[0]  # batchsize的大小
        for i in range(n_batch):  # batchsize循环
            for n in range(num_gridx):  # x方向网格循环
                for m in range(num_gridy):  # y方向网格循环
                    if labels[i, 4, m, n] == 1:  # 如果包含物体
                        # 将数据(px,py,w,h)转换为(x1,y1,x2,y2)
                        bbox1_pred_xyxy = ((self.pred[i, 0, m, n] + n) / num_gridx - self.pred[i, 2, m, n] / 2,
                                           (self.pred[i, 1, m, n] + m) / num_gridy - self.pred[i, 3, m, n] / 2,
                                           (self.pred[i, 0, m, n] + n) / num_gridx + self.pred[i, 2, m, n] / 2,
                                           (self.pred[i, 1, m, n] + m) / num_gridy + self.pred[i, 3, m, n] / 2)
                        bbox2_pred_xyxy = ((self.pred[i, 5, m, n] + n) / num_gridx - self.pred[i, 7, m, n] / 2,
                                           (self.pred[i, 6, m, n] + m) / num_gridy - self.pred[i, 8, m, n] / 2,
                                           (self.pred[i, 5, m, n] + n) / num_gridx + self.pred[i, 7, m, n] / 2,
                                           (self.pred[i, 6, m, n] + m) / num_gridy + self.pred[i, 8, m, n] / 2)
                        # ground truth
                        bbox_gt_xyxy = ((labels[i, 0, m, n] + n) / num_gridx - labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + m) / num_gridy - labels[i, 3, m, n] / 2,
                                        (labels[i, 0, m, n] + n) / num_gridx + labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + m) / num_gridy + labels[i, 3, m, n] / 2)
                        iou1 = calculate_iou(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy, bbox_gt_xyxy)
                        # iou大的负责检测物体
                        if (iou1 >= iou2):
                            coor_loss = coor_loss + 5 * (torch.sum((self.pred[i,0:2,m,n] - labels[i,0:2,m,n])**2))\
                                                    + 5 * (torch.sum((self.pred[i,2:4,m,n].sqrt() - labels[i,2:4,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (self.pred[i, 4, m, n] - iou1) ** 2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((self.pred[i, 9, m, n] - iou2) ** 2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((self.pred[i,5:7,m,n] - labels[i,5:7,m,n])**2))\
                                                    + 5 * (torch.sum((self.pred[i,7:9,m,n].sqrt() - labels[i,7:9,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (self.pred[i, 9, m, n] - iou2) ** 2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((self.pred[i, 4, m, n] - iou1) ** 2)
                        class_loss = class_loss + torch.sum((self.pred[i, 10:, m, n] - labels[i, 10:, m, n]) ** 2)
                    # 不负责预测的gd只有noobj_confi_loss
                    else:
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(self.pred[i, [4, 9], m, n] ** 2)
        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        return loss/n_batch
if __name__ == '__main__':
    # 自定义输入张量，验证网络可以正常跑通，并计算loss，调试用
    x = torch.zeros(5, 3, 448, 448)
    net = YOLOV1_Net()
    a = net(x)
    labels = torch.zeros(5, 30, 7, 7)
    loss = net.cal_loss(labels)
    print(loss)
    print(a.shape)