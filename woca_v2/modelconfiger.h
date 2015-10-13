/*
这里定义模板配置文件的数据结构。
*/
#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include <opencv2/opencv.hpp>


using namespace cv;

struct MP{
	Mat model;
	Point* points;
};

struct SExpression {
	string name;

	MP leftEye;
	MP rightEye;

	Mat leftEyeBrow;
	Mat rightEyeBrow;

	MP leftBrow;
	MP rightBrow;

	Mat leftPupilModel;
	Mat rightPupilModel;

	MP mouth;
};

struct SHeadModel{
	string name;
	Mat faceModel;

	MP nose;

	vector<SExpression> expressions;
};

class CHeadModels
{
public:
	CHeadModels();
	virtual~ CHeadModels();

	SHeadModel currentHead;
	SExpression currentExpression;
	vector<SHeadModel> headModels;

	void addHeadConfig(SHeadModel);
	void addExpression(string name,SHeadModel shm,SExpression se);

	MP createMP(string modelFileName,Point* ps);
	SExpression createExpression(string name,MP leftEye,MP rightEye,MP leftEyeBrow,MP rightEyeBrow,Mat leftPupilModel,Mat rightPupilModel,MP mouth);
	SHeadModel createHeadModel(	string name,string faceModelFileName,MP nose,vector<SExpression> expressions);
	SHeadModel createHeadModel(	string name);
	void setFaceModel(SHeadModel shm,string faceModelFileName);
	
	void setCurrentHead(string headName);
	void setCurrentExpression(string headName);
	void init();
	void release();
};
