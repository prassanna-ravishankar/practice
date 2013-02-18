#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"

using namespace std;

void read_iris(drwnClassifierDataset& dataset, const char* feats, const char* labels){
    ifstream f(feats);
    ifstream l(labels);
    double feature;
    int label;
    const int nInstance = 150;
    const int nFeature = 4;
    vector<double> fv(nFeature);

    for(int i = 0; i < nInstance; ++i){
        fv.clear();
        l >> label;
        for(int j = 0; j < nFeature; ++j){
            f >> feature;
            fv.push_back(feature);
        }
        dataset.append(fv,label);
    }
}

int read_libsvm(drwnClassifierDataset& dataset, const char* filename, int nInstance, int nFeature){
    ifstream file(filename);
    vector<double> fv(nFeature,0);
    int nClasses = 0;
    
    string buf;
    for(int i = 0; i < nInstance;  i++){
        getline(file,buf);      
        fv.clear();
        istringstream is(buf);
        int label, id;
        char sep;
        double val;
        is >> label;
        if(label > nClasses){
            nClasses = label;
        }
        while(is >> id >> sep >> val){
            fv[id-1] = val;
        }


        dataset.append(fv,label-1);
    }

    return nClasses;
}

int main(int argc, char **argv)
{
    const double SIGNAL2NOISE = 1.5;
    const int NUMPOSSAMPLES = 10;
    const int NUMNEGSAMPLES = 20;

    drwnClassifierDataset dataset;
    //    read_iris(dataset,argv[1],argv[2]);
    //    const int nFeatures = dataset.numFeatures();
    //    const int nClasses = dataset.maxTarget() + 1;
    const int nInstance = atoi(argv[2]);
    const int nFeatures = atoi(argv[3]);
    const int nClasses = read_libsvm(dataset,argv[1],nInstance,nFeatures);


     drwnClassifier *model;
     for (int i = 0; i < 3; i++) {
         if (i == 0) {
             DRWN_LOG_MESSAGE("logistic classifier");
             model = new drwnTMultiClassLogistic<drwnBiasJointFeatureMap>(nFeatures, nClasses);
         } else if (i == 1) {
             DRWN_LOG_MESSAGE("boosted classifier");
             model = new drwnBoostedClassifier(nFeatures, nClasses);
         } else if (i == 2){
             DRWN_LOG_MESSAGE("random forest classifier");
             model = new drwnRandomForest(nFeatures, nClasses);
         }


         model->train(dataset);

         drwnConfusionMatrix confusion(nClasses, nClasses);
         confusion.accumulate(dataset, model);
         confusion.printCounts();

         delete model;
     }

    return 0;
}
