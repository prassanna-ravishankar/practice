#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include "Eigen/Core"
using namespace std;
using namespace Eigen;

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

MatrixXd readData(const char *file){
    std::ifstream ifs(file);
    std::vector<double> data;
    std::string line;
    std::vector<std::string> strings;
    int n_sample = 0;
    int dim;
    while(std::getline(ifs,line)){
        strings.clear();
        boost::split(strings,line,boost::is_any_of(" \n\s"));
        dim = strings.size();
        BOOST_FOREACH(std::string s, strings){
           boost::trim_right(s);
           if(s.size() == 0) continue;
           double v = boost::lexical_cast<double>(s);
           data.push_back(v);
        }
        ++n_sample;
    }
    Eigen::Map<MatrixXd> X(&data[0], dim-1, n_sample);
    return X.transpose();
}

MatrixXi readData2(const char *file){
    std::ifstream ifs(file);
    std::vector<int> data;
    std::string line;
    std::vector<std::string> strings;
    int n_sample = 0;
    int dim;
    while(std::getline(ifs,line)){
        strings.clear();
        boost::split(strings,line,boost::is_any_of(" \n\s"));
        dim = strings.size();
        BOOST_FOREACH(std::string s, strings){
           boost::trim_right(s);
           if(s.size() == 0) continue;
           int v = boost::lexical_cast<int>(s);
           data.push_back(v);
        }
        ++n_sample;
    }
    Eigen::Map<MatrixXi> X(&data[0], dim-1, n_sample);
    return X.transpose();
}

void scale(MatrixXd& X, VectorXd& mean, VectorXd& std_div){
    const int n = X.rows();
    mean = X.colwise().mean();
    std_div = (X.array().square().colwise().sum() / n - mean.transpose().array().square()).sqrt();
    X.rowwise() -= mean.transpose();
    X.array().rowwise() /= std_div.transpose().array();
}

int main(int argc, char **argv)
{

    MatrixXd X = readData(argv[1]);
    VectorXi label = readData2(argv[2]).col(0);
    // std::cout << label << endl;
    VectorXd mean,std_div;
    scale(X, mean, std_div);

    const int n_classes = atoi(argv[5]);
    cout << n_classes << endl;
    vector<vector<double> > features(X.rows(), vector<double>(X.cols()));
    vector<int> targets(X.rows());

    Map<VectorXi>(&targets[0], X.rows()) = label;
    for (int i = 0; i < X.rows(); ++i)
    {
        Map<VectorXd>(&features[i][0], X.cols()) = X.row(i);
    }

    drwnClassifierDataset dataset;
    const int nInstance = X.rows();
    const int nFeatures = X.cols();
    const int nClasses = 2;

    dataset.features = features;
    dataset.targets = targets;
    drwnClassifier *model =  new drwnRandomForest(nFeatures, n_classes);

    model->train(dataset);

    drwnConfusionMatrix confusion(n_classes,n_classes);
           confusion.accumulate(dataset, model);
           confusion.printCounts();

    MatrixXd X_test = readData(argv[3]);
    VectorXi label_test = readData2(argv[4]);

    vector<vector<double> > features_test(X_test.rows(), vector<double>(X.cols()));
    vector<int> correct(X_test.rows());

    X_test.rowwise() -= mean.transpose();
    X_test.array().rowwise() /= std_div.transpose().array();

    Map<VectorXi>(&correct[0], X_test.rows()) = label_test;
    for (int i = 0; i < X_test.rows(); ++i)
    {
        Map<VectorXd>(&features_test[i][0], X.cols()) = X_test.row(i);
    }

    dataset.features = features_test;
    vector<int> predictions;
    model->getClassifications(dataset.features, predictions);

    Map<VectorXi> predicted(&predictions[0], predictions.size());

    int c = 0;
    for(int i = 0; i < predictions.size(); ++i)
    {
        if (predictions[i] == label_test(i)) c++;
    }

    cout << (float)c / predictions.size() << endl;
    delete model;

    return 0;
}
