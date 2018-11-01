#include "util.h"
#include <sys/stat.h>
#include <unistd.h>
using namespace cv;

int createDir(const   char   *sPathName)
{
    if(access(sPathName,NULL)!=0){
        if(mkdir(sPathName,0755)==-1){
            return -1;
        }
    }
    return 0;
}

float IoU(Rect& rect1,Rect& rect2){
    Rect inRect=rect1 & rect2;
    Rect outRect=rect1 | rect2;
    float iou=(float)inRect.area()/(float)outRect.area();
    return iou;
}
