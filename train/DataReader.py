import os
import cv2

#wider的训练集描述文件
widerTrainDesc=r'/home/ssbai/data/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
#wider的训练图片目录
widerTrainDir=r'/home/ssbai/data/WIDER/WIDER_train/images'

#celebA的box描述文件
celebATrainBox=r'/home/ssbai/data/CelebA/Anno/list_bbox_celeba.txt'
#celebA的landmark描述文件
celebATrainLandmark=r'/home/ssbai/data/CelebA/Anno/list_landmarks_celeba.txt'
#celebA的训练图片目录
celebATrainDir=r'/home/ssbai/data/CelebA/'

def readWiderDatas():
    """读取wider数据库
    """
    allDatas=list()
    with open(widerTrainDesc,'r') as descFile:
        while True:
            line=descFile.readline()
            if not line:
                break
            dataItem=dict()
            name=line.strip()
            dataItem['filePath']=os.path.join(widerTrainDir,name)
            count=int(descFile.readline())
            if count <= 0:
                descFile.readline()
                continue
            boxes=[]
            for i in range(count):
                line=descFile.readline()
                x1,y1,w,h=[int(x) for x in line.split()[0:4]]
                if max(w,h) < 20 or x1<0 or y1<0:
                    continue
                box=[x1,y1,x1+w,y1+h]
                boxes.append(box)
            dataItem['boxes']=boxes
            dataItem['faceCount']=len(boxes)
            allDatas.append(dataItem)
    return allDatas

def readCelebAData():
    allData=list()
    with open(celebATrainBox,'r') as boxFile:
        with open(celebATrainLandmark,'r') as landmarkFile:
            #跳过开头的信息
            boxFile.readline()
            boxFile.readline()
            landmarkFile.readline()
            landmarkFile.readline()
            while True:
                line=boxFile.readline()
                if not line:
                    break
                part=line.split()
                filePath=os.path.join(celebATrainDir,part[0])
                x1,y1,w,h=[int(x) for x in part[1:5]]
                box=[x1,y1,x1+w,y1+h]
                line=landmarkFile.readline()
                landmark=[int(x) for x in line.split()[1:11]]
                #质量不好的图片
                if min(w,h) <40 or x1<0 or y1 <0:
                    continue
                dataItem=dict()
                dataItem['filePath']=filePath
                dataItem['box']=box
                dataItem['landmark']=landmark
                allData.append(dataItem)
    return allData

if __name__=='__main__':
    # widerData=readWiderDatas()
    # for dataItem in widerData:
    #     image=cv2.imread(dataItem['filePath'])
    #     for box in dataItem['boxes']:
    #         cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,0,255))
    #     cv2.imshow('image',image)
    #     cv2.waitKey()
    #     cv2.destroyWindow('image')
    celebaData=readCelebAData()
    for dataItem in celebaData:
        print(dataItem)