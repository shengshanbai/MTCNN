#include<iostream>
#include<getopt.h>
#include<string>
#include<cstring>

using namespace std;

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

    return 0;
}
