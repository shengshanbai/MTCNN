#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <time.h>
#include <algorithm>
#include "util.h"

using namespace std;
using namespace cv;

#define WIDER_TRAIN_DIR "./data/WIDER_train"
#define WIDER_TRAIN_DESC_FILE "./data/wider_face_train_bbx_gt.txt"

#define GEN_12_DIR "./data/12"
#define GEN_POS_DIR GEN_12_DIR "/positive"
#define GEN_PART_DIR GEN_12_DIR "/part"
#define GEN_NEG_DIR GEN_12_DIR "/negative"

struct WiderDataDesc{
    string imagePath;
    int boxCount;
    vector<Rect> boxs;
};

int createGenDirs(){
    int result=0;
    result=createDir("./data");
    if(result<0){
        cout<<"create dir:"<<"./data"<<"failed"<<endl;
        return result;
    }
    result=createDir(GEN_12_DIR);
    if(result<0){
        cout<<"create dir:"<<GEN_12_DIR<<"failed"<<endl;
        return result;
    }
    result=createDir(GEN_POS_DIR);
    if(result<0){
        cout<<"create dir:"<<GEN_POS_DIR<<"failed"<<endl;
        return result;
    }
    result=createDir(GEN_PART_DIR);
    if(result<0){
        cout<<"create dir:"<<GEN_PART_DIR<<"failed"<<endl;
        return result;
    }
    result=createDir(GEN_NEG_DIR);
    if(result<0){
        cout<<"create dir:"<<GEN_NEG_DIR<<"failed"<<endl;
        return result;
    }
    return result;
}

vector<WiderDataDesc> readWiderData(fstream& descStream){
    vector<WiderDataDesc> datas;
    string line;
    while(!descStream.eof()){
        WiderDataDesc desc;
        descStream>>desc.imagePath;
        if(desc.imagePath.empty())
            break;
        descStream>>desc.boxCount;
        for(int i=0;i<desc.boxCount;i++){
            Rect rect;
            descStream>>rect.x>>rect.y>>rect.width>>rect.height;
            desc.boxs.push_back(rect);
            getline(descStream,line);
        }
        datas.push_back(desc);
    }
    return datas;
}

void showImage(Mat& image,WiderDataDesc dataDesc){
    Mat display=image.clone();
    for(Rect rect:dataDesc.boxs){
        rectangle(display,rect,Scalar(255,0,0),2);
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", display );    // Show our image inside it.
    waitKey(0);
}


void showImage(Mat& image,Rect rect){
    Mat display=image.clone();
    rectangle(display,rect,Scalar(255,0,0),2);
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", display );    // Show our image inside it.
    waitKey(0);
}

bool saveImage(Mat& image,Rect roi,const string& path){
    cout<<"saving file :"<<path<<endl;
    try{
        Mat imageRoi=image(roi).clone();
        Mat dest;
        resize(imageRoi,dest,Size(12,12));
        imwrite(path,dest);
        return true;
    }catch(cv::Exception e){
        cout<<"save image failed:"<<roi.x<<" "<<roi.y<<" "<<roi.width<<" "<<roi.height<<endl;
    }
    return false;
}

bool isNegativeRect(Rect& rect,vector<Rect>& boxs){
    for(auto box:boxs){
        float iou=IoU(rect,box);
        cout<<iou<<endl;
        if(iou>=0.3)
            return false;
    }
    return true;
}

int main(int argc,char** argv){
    createGenDirs();
    ofstream posStream,partStream,negStream;
    int posIndex=0,partIndex=0,negIndex=0;
    posStream.open(GEN_12_DIR "/positive.txt");
    if(!posStream.is_open()){
        cout<<"can't open file:"<<GEN_12_DIR "/positive.txt"<<endl;
    }
    partStream.open(GEN_12_DIR "/part.txt");
    if(!partStream.is_open()){
        cout<<"can't open file:"<<GEN_12_DIR "/part.txt"<<endl;
    }
    negStream.open(GEN_12_DIR "/negative.txt");
    if(!negStream.is_open()){
        cout<<"can't open file:"<<GEN_12_DIR "/negative.txt"<<endl;
    }
    fstream widerDescStream;
    widerDescStream.open(WIDER_TRAIN_DESC_FILE,fstream::in);
    if(!widerDescStream.is_open()){
        cout<<"can't open file:"<<WIDER_TRAIN_DESC_FILE<<endl;
        return -1;
    }
    vector<WiderDataDesc> dataDescVector=readWiderData(widerDescStream);
    cout<<"data desc count:"<<dataDescVector.size()<<endl;
    widerDescStream.close();
    default_random_engine random(time(NULL));
    for(auto dataDesc:dataDescVector){
        Mat image;
        string imagePath=string(WIDER_TRAIN_DIR "/images/")+ dataDesc.imagePath;
        cout<<"processing "<<imagePath<<endl;
        image = imread(imagePath, IMREAD_COLOR);
        //showImage(image,dataDesc);
        int negNumber=0;
        int width=image.size().width;
        int height=image.size().height;
        uniform_int_distribution<int> sizeDist(12,min(width,height)/2);
        //generate 50 random negative image
        while(negNumber<50){
            Rect negRect;
            int size=sizeDist(random);
            negRect.width=size;
            negRect.height=size;
            uniform_int_distribution<int> dist(0,width-size-1);
            negRect.x=dist(random);
            dist=uniform_int_distribution<int>(0,height-size-1);
            negRect.y=dist(random);
            if(isNegativeRect(negRect,dataDesc.boxs)){
                string path=string(GEN_NEG_DIR)+"/"+to_string(negIndex)+".jpg";
                if(saveImage(image,negRect,path)){
                    negStream<<path<<endl;
                    negIndex++;
                    negNumber++;
                }
            }
        }
        //generate positive image
        for(auto box:dataDesc.boxs){
            // ignore small faces
            // in case the ground truth boxes of small faces are not accurate
            if (std::max(box.width,box.height)<40 || box.x<0 || box.y<0){
                    continue;
            }
            //try 5 time to generate negtive image;
            for(int i=0;i<5;i++){
                int size=sizeDist(random);
                uniform_int_distribution<int> tDist(max(-size, -box.x), box.width);
                // deltaX and deltaX are offsets of (x, y)
                int deltaX = tDist(random);
                tDist=uniform_int_distribution<int>(max(-size, -box.y), box.height);
                int deltaY = tDist(random);
                Rect negRect;
                negRect.x=std::max(0,box.x+deltaX);
                negRect.y=std::max(0,box.y+deltaY);
                negRect.width=box.width;
                negRect.height=box.height;
                if (negRect.x+negRect.width > width || negRect.y+negRect.height>height)
                    continue;
                if(isNegativeRect(negRect,dataDesc.boxs)&&negRect.x>=0
                    &&negRect.y>=0&&negRect.x+negRect.width<=width
                    &&negRect.y+negRect.height<=height){
                    string path=string(GEN_NEG_DIR)+"/"+to_string(negIndex)+".jpg";
                    if(saveImage(image,negRect,path)){
                        negStream<<path<<endl;
                        negIndex++;
                        negNumber++;
                    }
                }
            }
            //try 20 times to generate positive and part image
            uniform_int_distribution<int> posSizeDist((int)std::min(box.width,box.height)*0.8,
                                                      std::ceil(std::max(box.width,box.height)*1.25));
            uniform_int_distribution<int> posXDist(-box.width*0.2,box.width*0.2);
            uniform_int_distribution<int> posYDist(-box.height*0.2,box.height*0.2);
            for(int i=0;i<25;i++){
                int size=posSizeDist(random);
                int deltaX=posXDist(random);
                int deltaY=posYDist(random);
                Rect posRect;
                posRect.x=std::max(box.x+box.width/2+deltaX-size/2,0);
                posRect.y=std::max(box.y+box.height/2+deltaY-size/2,0);
                posRect.width=size;
                posRect.height=size;
                if (posRect.x+posRect.width >width ||
                        posRect.y+posRect.height>height)
                    continue;
                float offsetX = (box.x - posRect.x) / float(size);
                float offsetY = (box.y-posRect.y) / float(size);
                float offsetW = (box.width - size)/ float(size);
                float offsetH = (box.height-size) /float(size);
                if(IoU(posRect,box)>=0.65){
                    string path=string(GEN_POS_DIR)+"/"+to_string(posIndex)+".jpg";
                    saveImage(image,posRect,path);
                    posStream<<path<<" 1 "<<offsetX<<" "
                            <<offsetY<<" "<<offsetW<<" "
                            <<offsetH<<" "
                            <<endl;
                    posIndex++;
                } else if(IoU(posRect,box)>=0.4){
                    string path=string(GEN_PART_DIR)+"/"+to_string(partIndex)+".jpg";
                    saveImage(image,posRect,path);
                    partStream<<path<<" -1 "<<offsetX<<" "
                            <<offsetY<<" "<<offsetW<<" "
                            <<offsetH<<" "
                            <<endl;
                    partIndex++;
                }
            }
        }
    }
    posStream.close();
    partStream.close();
    negStream.close();
    return 0;
}
