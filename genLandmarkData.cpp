#include<iostream>
#include<getopt.h>
#include<string>
#include<cstring>
#include "util.h"
#include<fstream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<time.h>
#include<random>
#include<sstream>

using namespace std;
using namespace cv;

#define LANDMARK_FILE "./data/testImageList.txt"

void printHelp(){
    cout<<"Usage: genLandmarkData [OPTION]..."<<endl
       <<"-n network type (PNet RNet ONet) "<<endl
      <<"-e extend the data 1:true 0:false default true"<<endl;
}

enum NetType{
    UNKOWN,
    PNet,
    RNet,
    ONet
};

struct LandmarkData
{
    string imagePath;
    Rect box;
    vector<Point2f> landmarks;
};

void replaceDelimiter(string& path){
    auto iter=path.begin();
    while (iter!=path.end()) {
        if((*iter)=='\\') {
           path.replace(iter,iter+1,"/");
        }
        iter++;
    }
}


vector<LandmarkData> readLandmarks(const string& path){
    vector<LandmarkData> result;
    ifstream markStream(path);
    if(!markStream.is_open()){
        cout<<"can't open landmark file:"<<path<<endl;
        return result;
    }
    while (!markStream.eof()) {
        LandmarkData landmarkData;
        int x1,x2,y1,y2;
        markStream>>landmarkData.imagePath
                >>x1>>x2>>y1>>y2;
        if(landmarkData.imagePath.empty())
            break;
        replaceDelimiter(landmarkData.imagePath);
        landmarkData.box.x=x1;
        landmarkData.box.y=y1;
        landmarkData.box.width=x2-x1;
        landmarkData.box.height=y2-y1;
        while (landmarkData.landmarks.size()<5) {
            Point2f point;
            markStream>>point.x>>point.y;
            landmarkData.landmarks.push_back(point);
        }
        result.push_back(landmarkData);
    }
    return result;
}

void showImage(Mat& image,Rect& box,vector<Point2f>& landmarks){
    Mat display=image.clone();
    rectangle(display,box,Scalar(255,0,0),2);
    for(Point2f point:landmarks){
        circle(display,point,2,Scalar(0,255,0));
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", display );    // Show our image inside it.
    waitKey(0);
}

void saveImage(Mat& image,Rect& roi,string& path,const Size& size){
    Mat imageRoi=image(roi).clone();
    Mat dest;
    resize(imageRoi,dest,size);
    imwrite(path,dest);
}

vector<Point2f> flipLandmark(Rect& box,vector<Point2f> landmarks){
    for(auto& point:landmarks){
        point.x=(box.width-(point.x-box.x))+box.x;
    }
    std::swap(landmarks[0],landmarks[1]); //left eye<->right eye
    std::swap(landmarks[3],landmarks[4]); //#left mouth<->right mouth
   return landmarks;
}

vector<Point2f> rotateLandmark(Mat& roteMatrix,vector<Point2f> landmarks){
    for(auto& point:landmarks){
        float x=roteMatrix.at<double>(0,0)*point.x+roteMatrix.at<double>(0,1)*point.y+roteMatrix.at<double>(0,2);
        float y=roteMatrix.at<double>(1,0)*point.x+roteMatrix.at<double>(1,1)*point.y+roteMatrix.at<double>(1,2);
        point.x=x;
        point.y=y;
    }
    return landmarks;
}

void generateData(NetType netType,bool extend){
    //prepare output dir;
    string outDir;
    string outDescFile;
    int outSize;
    switch (netType) {
    case PNet:
        outDir="./data/12/train_PNet_landmark";
        outDescFile="./data/12/train_PNet_landmark.txt";
        outSize=12;
        break;
    case RNet:
        outDir="./data/24/train_RNet_landmark";
        outDescFile="./data/24/train_RNet_landmark.txt";
        outSize=24;
        break;
    case ONet:
        outDir="./data/48/train_ONet_landmark";
        outDescFile="./data/48/train_ONet_landmark.txt";
        outSize=48;
        break;
    default:
        break;
    }
    if(createDir(outDir.c_str())!=0){
        cout<<"can't create dir:"<<outDir;
        return;
    }
    ofstream outDescStream(outDescFile);
    vector<LandmarkData> datas=readLandmarks(LANDMARK_FILE);
    int imageId=0;
    default_random_engine random(time(NULL));
    for(LandmarkData landmark:datas){
        Mat image=imread("./data/"+landmark.imagePath,IMREAD_COLOR);
        string savePath=outDir+"/"+to_string(imageId)+".jpg";
        saveImage(image,landmark.box,savePath,Size(outSize,outSize));
        outDescStream<<savePath<<" -2 ";
        for(Point2f& point:landmark.landmarks){
            float x=(point.x-landmark.box.x)/(float)landmark.box.width;
            float y=(point.y-landmark.box.y)/(float)landmark.box.height;
            outDescStream<<x<<" "<<y<<" ";
        }
        outDescStream<<endl;
        imageId++;
        if(!extend)
            return;
        uniform_int_distribution<int> binaryDist(0,1);
        //random shift box
        for(int i=0;i<10;i++){
            uniform_int_distribution<int> sizeDist(std::min(landmark.box.width,landmark.box.height)*0.8,
                                                   std::ceil(std::max(landmark.box.width,landmark.box.height)*1.25));
            uniform_int_distribution<int> deltaXDist(-landmark.box.width*0.2,landmark.box.width*0.2);
            uniform_int_distribution<int> deltaYDist(-landmark.box.height*0.2,landmark.box.height*0.2);
            Rect newBox;
            int size=sizeDist(random);
            int deltaX=deltaXDist(random);
            int deltaY=deltaYDist(random);
            newBox.width=size;
            newBox.height=size;
            newBox.x=std::max(landmark.box.x+landmark.box.width/2-size/2+deltaX,0);
            newBox.y=std::max(landmark.box.y+landmark.box.height/2-size/2+deltaY,0);
            if(newBox.x+newBox.width>image.size().width||
                            newBox.y+newBox.height>image.size().height)
                continue;
            float iou=IoU(newBox,landmark.box);
            if(iou>0.65){
                //normalize
                string savePath=outDir+"/"+to_string(imageId)+".jpg";
                ostringstream outString;
                outString<<savePath<<" -2 ";
                for(Point2f& point:landmark.landmarks){
                    float x=(point.x-newBox.x)/(float)newBox.width;
                    float y=(point.y-newBox.y)/(float)newBox.height;
                    if(x<0||x>1||y<0||y>1){
                        continue;
                    }
                    outString<<x<<" "<<y<<" ";
                }
                saveImage(image,newBox,savePath,Size(outSize,outSize));
                outDescStream<<outString.str()<<endl;
                cout<<"normalize image:"<<outString.str()<<endl;
                imageId++;
                //mirror
                if(binaryDist(random)>0){
                    string savePath=outDir+"/"+to_string(imageId)+".jpg";
                    ostringstream outString;
                    outString<<savePath<<" -2 ";
                    auto flipMarks=flipLandmark(newBox,landmark.landmarks);
                    for(Point2f& point:flipMarks){
                        float x=(point.x-newBox.x)/(float)newBox.width;
                        float y=(point.y-newBox.y)/(float)newBox.height;
                        outString<<x<<" "<<y<<" ";
                    }
                    Mat imageT=image.clone();
                    Mat roi=imageT(newBox);
                    cv::flip(roi,roi,1);
                    saveImage(imageT,newBox,savePath,Size(outSize,outSize));
                    outDescStream<<outString.str()<<endl;
                    cout<<"mirror image:"<<outString.str()<<endl;
                    imageId++;
                }
                //rotate
                if(binaryDist(random)>0){
                    string savePath=outDir+"/"+to_string(imageId)+".jpg";
                    ostringstream outString;
                    outString<<savePath<<" -2 ";
                    Mat rotateMat=getRotationMatrix2D(Point2f(newBox.x+newBox.width/2,newBox.y+newBox.height/2),
                                                      5,1);
                    auto rotateMarks=rotateLandmark(rotateMat,landmark.landmarks);
                    for(Point2f& point:rotateMarks){
                        float x=(point.x-newBox.x)/(float)newBox.width;
                        float y=(point.y-newBox.y)/(float)newBox.height;
                        if(x<0||x>1||y<0||y>1){
                            continue;
                        }
                        outString<<x<<" "<<y<<" ";
                    }
                    Mat imageT=image.clone();
                    warpAffine(imageT,imageT,rotateMat,imageT.size());
                    saveImage(imageT,newBox,savePath,Size(outSize,outSize));
                    outDescStream<<outString.str()<<endl;
                    cout<<"rotate image:"<<outString.str()<<endl;
                    imageId++;
                }
                //inverse clockwise rotation
                if(binaryDist(random)>0){
                    string savePath=outDir+"/"+to_string(imageId)+".jpg";
                    ostringstream outString;
                    outString<<savePath<<" -2 ";
                    Mat rotateMat=getRotationMatrix2D(Point2f(newBox.x+newBox.width/2,newBox.y+newBox.height/2),
                                                      -5,1);
                    auto rotateMarks=rotateLandmark(rotateMat,landmark.landmarks);
                    for(Point2f& point:rotateMarks){
                        float x=(point.x-newBox.x)/(float)newBox.width;
                        float y=(point.y-newBox.y)/(float)newBox.height;
                        if(x<0||x>1||y<0||y>1){
                            continue;
                        }
                        outString<<x<<" "<<y<<" ";
                    }
                    Mat imageT=image.clone();
                    warpAffine(imageT,imageT,rotateMat,imageT.size());
                    saveImage(imageT,newBox,savePath,Size(outSize,outSize));
                    outDescStream<<outString.str()<<endl;
                    cout<<"inverse rotate image:"<<outString.str()<<endl;
                    imageId++;
                }
            }
        }
    }
    outDescStream.close();
}

int main(int argc,char** argv){
    NetType netType=UNKOWN;
    int extend=1;
    int opt;
    const char* optString="n:e:";
    while((opt=getopt(argc,argv,optString))!=-1){
        switch (opt) {
        case 'n':
            if(strcmp("PNet",optarg)==0){
                netType=PNet;
            }else if(strcmp("RNet",optarg)==0){
                netType=RNet;
            }else if(strcmp("ONet",optarg)==0){
                netType=ONet;
            }else{
                printHelp();
                return -1;
            }
            break;
        case 'e':
            extend=stoi(optarg);
            if(extend!=0&&extend!=1){
                printHelp();
                return -1;
            }
            break;
        case '?':
            printHelp();
            return -1;
        default:
            break;
        }
    }

    if(netType==UNKOWN){
        cout<<"must input network type"<<endl;
        printHelp();
        return -1;
    }
    generateData(netType,extend);
    return 0;
}
