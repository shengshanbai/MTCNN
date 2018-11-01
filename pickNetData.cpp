#include <iostream>
#include "util.h"
#include<getopt.h>
#include<fstream>
#include<string>
#include<random>
#include<time.h>
#include<set>
#include<algorithm>

using namespace std;

void printHelp(){
    cout<<"Usage: pickNetData [OPTION]..."<<endl
       <<"-n network type (PNet RNet ONet) "<<endl;
}

vector<string> getFileContent(const string& path){
    string line;
    ifstream inStream(path);
    vector<string> allLines;
    if(!inStream.is_open()){
        cout<<"can't open file:"<<path;
        return allLines;
    }
    while (getline(inStream,line)) {
        allLines.push_back(line);
    }
    inStream.close();
    return allLines;
}

vector<string> randomChoice(vector<string>& inVector,int count){
    if(inVector.size()<count){
        return inVector;
    }
    set<int> indexSet;
    default_random_engine random(time(NULL));
    uniform_int_distribution<int> dist(0,inVector.size()-1);
    while (indexSet.size()<count) {
        int index=dist(random);
        indexSet.insert(index);
    }
    vector<string> result;
    for(const int& index:indexSet){
        result.push_back(inVector.at(index));
    }
    return result;
}

int main(int argc,char** argv){
    char* netType=NULL;
    int extend=1;
    int opt;
    const char* optString="n:";
    while((opt=getopt(argc,argv,optString))!=-1){
        switch (opt) {
        case 'n':
            if(strcmp("PNet",optarg)==0){
                netType=optarg;
            }else if(strcmp("RNet",optarg)==0){
                netType=optarg;
            }else if(strcmp("ONet",optarg)==0){
                netType=optarg;
            }else{
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

    if(netType==NULL){
        cout<<"must input network type"<<endl;
        printHelp();
        return -1;
    }

    vector<string> negLines=getFileContent("./data/12/negative.txt");
    vector<string> partLines=getFileContent("./data/12/part.txt");
    vector<string> posLines=getFileContent("./data/12/positive.txt");
    vector<string> landmarkLines=getFileContent("./data/12/train_"+string(netType)+"_landmark.txt");
    cout<<"negLines:"<<negLines.size()<<" partLines:"<<partLines.size()<<"posLines:"<<posLines.size()
       <<"landmarkLines:"<<landmarkLines.size()<<endl;
    int base_num = std::min(std::min(negLines.size()/3,partLines.size()),posLines.size());
    vector<string> picks=randomChoice(negLines,base_num*3);
    ofstream PNetStream("./data/12/train_"+string(netType)+"_data.txt");
    for(string& line:picks){
        PNetStream<<line<<endl;
        cout<<line<<endl;
    }
    picks=randomChoice(partLines,base_num);
    for(string& line:picks){
        PNetStream<<line<<endl;
        cout<<line<<endl;
    }
    picks=randomChoice(posLines,base_num);
    for(string& line:picks){
        PNetStream<<line<<endl;
        cout<<line<<endl;
    }
    picks=randomChoice(landmarkLines,base_num*2);
    for(string& line:picks){
        PNetStream<<line<<endl;
        cout<<line<<endl;
    }
    PNetStream.close();
    return 0;
}
