import tensorflow as tf
import cv2

def genTfrecords(inDataFile,dataDir,netType,shuffling=False):
    tfrecordsFile='%s/train_%s_data.tfrecord' % (dataDir,netType)
    if tf.gfile.Exists(tfrecordsFile):
        print('tfrecords file :%s already exist' % (tfrecordsFile))
        return
    dataSet=getDataSet(inDataFile)
    print('read data set complete')
    with tf.python_io.TFRecordWriter(tfrecordsFile) as tfRecordWriter:
        for i,dataItem in enumerate(dataSet):
            print('converting image %d/%d' % (i,len(dataSet)))
            addToTfRecords(dataItem,tfRecordWriter)
    print('convert PNet data finish!')

def addToTfRecords(dataItem,tfRecordWriter):
    imageData=cv2.imread(dataItem['filepath'])
    if imageData is None:
        print('can\'t open file:'+dataItem['filepath'])
        exit(-1)
    imageData=imageData.tostring()
    classLabel=dataItem['label']
    faceBox=dataItem['box']
    roi=[faceBox['x'],faceBox['y'],faceBox['width'],faceBox['height']]
    landmarks=[faceBox['leftEyeX'],
        faceBox['leftEyeY'],
        faceBox['rightEyeX'],
        faceBox['rightEyeY'],
        faceBox['noseX'],
        faceBox['noseY'],
        faceBox['leftMouthX'],
        faceBox['leftMouthY'],
        faceBox['rightMouthX'],
        faceBox['rightMouthY']]
    example=tf.train.Example(features=tf.train.Features(feature={
        'image/encoded':tf.train.Feature(bytes_list=tf.train.BytesList(value=[imageData])),
        'image/label':tf.train.Feature(int64_list=tf.train.Int64List(value=[classLabel])),
        'image/roi':tf.train.Feature(float_list=tf.train.FloatList(value=roi)),
        'image/landmark':tf.train.Feature(float_list=tf.train.FloatList(value=landmarks))
    }))
    tfRecordWriter.write(example.SerializeToString())

def getDataSet(inDataFile):
    imageDesc=open(inDataFile,'r')
    dataSet=[]
    for line in imageDesc.readlines():
        info=line.strip().split(' ')
        dataItem=dict()
        faceBox=dict()
        dataItem['filepath']=info[0]
        if not dataItem['filepath']:
            print('null filepath in:'+line)
            continue
        if len(info)<2:
            dataItem['label']=0
        else:
            dataItem['label']=int(info[1])
        faceBox['x']=0
        faceBox['y']=0
        faceBox['width']=0
        faceBox['height']=0
        faceBox['leftEyeX']=0
        faceBox['leftEyeY']=0
        faceBox['rightEyeX']=0
        faceBox['rightEyeY']=0
        faceBox['noseX']=0
        faceBox['noseY']=0
        faceBox['leftMouthX']=0
        faceBox['leftMouthY']=0
        faceBox['rightMouthX']=0
        faceBox['rightMouthY']=0
        if len(info)==6:
            faceBox['x'] = float(info[2])
            faceBox['y'] = float(info[3])
            faceBox['width'] = float(info[4])
            faceBox['height'] = float(info[5])
        if len(info)==12:
            faceBox['leftEyeX'] = float(info[2])
            faceBox['leftEyeY'] = float(info[3])
            faceBox['rightEyeX'] = float(info[4])
            faceBox['rightEyeY'] = float(info[5])
            faceBox['noseX'] = float(info[6])
            faceBox['noseY'] = float(info[7])
            faceBox['leftMouthX'] = float(info[8])
            faceBox['leftMouthY'] = float(info[9])
            faceBox['rightMouthX'] = float(info[10])
            faceBox['rightMouthY'] = float(info[11])
        dataItem['box']=faceBox
        dataSet.append(dataItem)
    return dataSet

if __name__ == '__main__':
    netType='PNet'
    dataDir='./data'
    dataDescFile='./data/12/train_PNet_data.txt'
    genTfrecords(dataDescFile,dataDir,netType,shuffling=True)
