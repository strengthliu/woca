#include "modelconfiger.h"

CHeadModels::CHeadModels(){
}
CHeadModels::~CHeadModels(){
}

MP CHeadModels::createMP(string modelFileName,Point* ps){
	MP ret;
	Mat m = imread(modelFileName,-1);
	ret.model = m;
	ret.points = ps;
	return ret;
}

SExpression CHeadModels::createExpression(string name,MP leftEye,MP rightEye,MP leftEyeBrow,MP rightEyeBrow,Mat leftPupilModel,Mat rightPupilModel,MP mouth){
	SExpression ret;
	return ret;
}

SHeadModel CHeadModels::createHeadModel(string name,string faceModelFileName,MP nose,vector<SExpression> expressions){
	SHeadModel ret;
	return ret;
}

SHeadModel CHeadModels::createHeadModel(string name){
	SHeadModel shmt;
	shmt.name = name;
	return shmt;
}
void CHeadModels::setFaceModel(SHeadModel shm,string faceModelFileName){
	Mat faceModel = imread(faceModelFileName,-1);
	shm.faceModel = faceModel;
}
void CHeadModels::release(){

}

void CHeadModels::setCurrentHead(string headName){
	for(int i=0;i<headModels.size();i++){
		SHeadModel shm = headModels.at(i);
		if(shm.name == headName){
			currentHead = shm;
			return;
		}
	}
}
void CHeadModels::setCurrentExpression(string expressionName){
	vector<SExpression> expressionst = currentHead.expressions;
	for(int i=0;i<expressionst.size();i++){
		SExpression setmp = expressionst.at(i);
		if(setmp.name == expressionName){
			currentExpression = setmp;
			return;
		}
	}
}
void CHeadModels::init(){
	if(currentHead.name.length()<1 && headModels.size()>0){
		headModels.begin();
		currentHead = headModels.at(0);

		vector<SExpression> expressionst = currentHead.expressions;
		if(expressionst.size()>0)
			currentExpression = expressionst.at(0);
	}
}