#include "appFace.h"
#include "sharedmatting.h"
#include "xmlParser.h"
#include "dataconvert.h"
//#include "modelconfiger.h"
#include "atlstr.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[]){

	if (argc < 2)
	{
		return 0;
	}

	string replaceModelOnly = argv[1];

	cout << "input file name:";
	string inputFileName = argv[2];
	if (inputFileName.length() == 0)
	{
		cin >> inputFileName;
	}
	else 
		cout << inputFileName << endl;
	

	Mat imageOrigine = imread("OrigineData//"+inputFileName);
	//imshow("imageOrigine", imageOrigine);
	
	cout << "input trimap file name:";
	string inputTrimapFileName = argv[3];
	//int isHair = 0;
	if (inputTrimapFileName.length() == 0)
	{
		cin >> inputTrimapFileName;
	}
	else 
		cout << inputTrimapFileName << endl;

	string inputFaceFileName = argv[4];;//= "trimap_test12.jpg";
	Mat trimapFaceImage = imread("OrigineData//" + inputFaceFileName);
	//resize(trimapFaceImage, trimapFaceImage, Size(imageOrigine.cols, imageOrigine.rows));

	//-- long hair or short hair
	cout << "long hair or short hair :" ;
	string isHair1 = argv[5];//= "trimap_test12.jpg";
	double d=atof(isHair1.c_str());
	int isHair = (int)d;
	Mat trimapImage = imread("OrigineData//" + inputTrimapFileName);
	//imshow("trimap", trimapImage);
	//waitKey(0);
	//appFace::autoChangeSize(imageOrigine, trimapImage, 960);

	replaceModelOnly = "yes";

	//-- SharedMatting for contour of human
	SharedMatting sm1;
	Mat imageContour;
	clock_t start, finish;
	start = clock();
	if(replaceModelOnly != "yes"){
		sm1.loadImage(imageOrigine);
		sm1.loadTrimap(trimapImage);
		sm1.solveAlpha();
		sm1.save("MedianData//result.png");
		finish = clock();
		cout <<  double(finish - start) / CLOCKS_PER_SEC << endl;
		imageContour = sm1.saveContourMask(13);
		imwrite("MedianData//result_blur_threshold.png", imageContour);
	}else
		imageContour = imread("MedianData//result_blur_threshold.png", 0);
	
	appFace aF(imageOrigine, imageContour);
	aF.debug = true;



	//-- SharedMatting for contour of face
	SharedMatting sm2;
	sm2.loadImage(imageOrigine);
	//-- 这里是生成脸的Trimap，要改成抠图。
	if(inputFaceFileName == "none"){
		aF.FaceDetection();
		trimapFaceImage = aF.generateTrimapOfFace(); //生成脸的Trimap。
	}
	if(replaceModelOnly != "yes"){
		sm2.loadTrimap(trimapFaceImage);
		sm2.solveAlpha();
		sm2.save("MedianData//2SM_result.png");
		Mat imageFaceContour = sm2.saveContourMask(13);//
		imwrite("MedianData//result_blur_threshold_face.png", imageFaceContour);
		aF.imageFaceContourSM = imageFaceContour;
	}else{
		aF.imageFaceContourSM = imread("MedianData//result_blur_threshold_face.png", 0);
	}

	aF.imageRealFaceContourSM = imread("MedianData//2SM_result.png",0);
	aF.imageRealContourSM = imread("MedianData//result.png",0);

	//是否是自动生成人脸TRIMAP，后面的处理机制也不同。不能跟上面的那句合并。一定是分在人脸抠图的前后面。
	//下面先注释掉，为了调试读XML文件。
	if(inputFaceFileName != "none")
		aF.FaceDetection();
	string dict = argv[6];
	string configFileName = ".\\OrigineData\\"+dict+"\\config.xml";
	dict = "OrigineData//" + dict +"//";

	Mat faceModel ;
	Mat leftEyeModel;
	Mat rightEyeModel;
	Mat leftEyeWithBrowModel;
	Mat rightEyeWithBrowModel;
	Mat leftEyePupilModel;
	Mat rightEyePupilModel;
	Mat mouthModel;
	Mat noseModel;

	CDataConverter dc = CDataConverter();
	wchar_t* configfilename = dc.StringToWchar(configFileName);
	XMLNode xMainNode=XMLNode::openFileHelper(configfilename,L"config");
	//if(xMainNode.isDeclaration()) // 如果打开了模板配置文件，就读配置文件
	{
		aF.chm = CHeadModels(); //初始化模板库
		int nHead = xMainNode.nChildNode(L"head");//有多少个头配置 head

		vector<SHeadModel> headModels(0);//nHead);
		for (int i = 0;i< nHead ;i++){ //循环每个头
			XMLNode xHeadNode = xMainNode.getChildNode(i);//取第I个头
			if(!xHeadNode.isEmpty()){ //如果有值
				string cHeadName =  dc.WstringToString(xHeadNode.getAttribute(L"name"));//	取脸模板文件 facemodelfile
				SHeadModel shmt = aF.chm.createHeadModel(cHeadName);
				XMLNode faceNode = xHeadNode.getChildNode(L"face");//取脸 face
				string cfaceModelFileName = dict + dc.WstringToString(faceNode.getAttribute(L"facemodelfile"));//	取脸模板文件 facemodelfile
				shmt.faceModel = imread(cfaceModelFileName,-1);
	imwrite("ResultData//BodyWithoutBackground1-2.png",shmt.faceModel);
				int nExpression = xHeadNode.nChildNode(L"expression");
				vector<SExpression> expressions(0);//nExpression);//初始化nExpression个表情
				for(int iExpression=0;iExpression<nExpression;iExpression++)	{
					SExpression sep;
					XMLNode expressionNode = xHeadNode.getChildNode(L"expression",iExpression);//一个个取表情 expression
					string expressionName = dc.WstringToString(expressionNode.getAttribute(L"expressionname"));//	取表情名称 expressionname
					sep.name = expressionName;//为这个表情填上名字。

						XMLNode eyeNode = expressionNode.getChildNode(L"eyes"); // 取该表情下的眼睛
					
							XMLNode leftEyeNode = eyeNode.getChildNode(L"lefteye");//	取左眼lefteye
								string cleftEyeModelFileName = dict + dc.WstringToString(leftEyeNode.getAttribute(L"eyemodelfile"));//		取左眼珠模板文件
								string cleftEyePupilModelFileName = dict + dc.WstringToString(leftEyeNode.getAttribute(L"pupilmodelfile"));//		取左眼珠模板文件
								string cleftEyeBrowModelFileName = dict + dc.WstringToString(leftEyeNode.getAttribute(L"eyewithbrowmodelfile"));//		取左眼眉模板文件
								string cleftBrowModelFileName = dict + dc.WstringToString(leftEyeNode.getAttribute(L"browmodelfile"));//		取左眼眉模板文件
								XMLNode leftEyePointsNode = leftEyeNode.getChildNode(L"eyepoint");//		取左眼点 eyepoint
									XMLNode leftEyePoint_left = leftEyePointsNode.getChildNode(L"left");//			取左点
									XMLNode leftEyePoint_right = leftEyePointsNode.getChildNode(L"right");//			取右点
									XMLNode leftEyePoint_top = leftEyePointsNode.getChildNode(L"top");//			取上点
									XMLNode leftEyePoint_buttom = leftEyePointsNode.getChildNode(L"buttom");//			取下点
								Point leftEyeP[4];
								leftEyeP[2] = Point(dc.WstringToInt(leftEyePoint_left.getAttribute(L"x")),dc.WstringToInt(leftEyePoint_left.getAttribute(L"y")));
								leftEyeP[3] = Point(dc.WstringToInt(leftEyePoint_right.getAttribute(L"x")),dc.WstringToInt(leftEyePoint_right.getAttribute(L"y")));
								leftEyeP[0] = Point(dc.WstringToInt(leftEyePoint_top.getAttribute(L"x")),dc.WstringToInt(leftEyePoint_top.getAttribute(L"y")));
								leftEyeP[1] = Point(dc.WstringToInt(leftEyePoint_buttom.getAttribute(L"x")),dc.WstringToInt(leftEyePoint_buttom.getAttribute(L"y")));
								MP leftEye = aF.chm.createMP(cleftEyeModelFileName,leftEyeP);//左眼MP
								sep.leftEye = leftEye;

								XMLNode leftBrowPointsNode = leftEyeNode.getChildNode(L"browpoint");//取眉毛
									XMLNode leftBrowPoint_left = leftBrowPointsNode.getChildNode(L"left");//			取左点
									XMLNode leftBrowPoint_right = leftBrowPointsNode.getChildNode(L"right");//			取右点
									XMLNode leftBrowPoint_top = leftBrowPointsNode.getChildNode(L"top");//			取上点
									XMLNode leftBrowPoint_buttom = leftBrowPointsNode.getChildNode(L"buttom");//			取下点
								Point leftBrowP[4];
								leftBrowP[2] = Point(dc.WstringToInt(leftBrowPoint_left.getAttribute(L"x")),dc.WstringToInt(leftBrowPoint_left.getAttribute(L"y")));
								leftBrowP[3] = Point(dc.WstringToInt(leftBrowPoint_right.getAttribute(L"x")),dc.WstringToInt(leftBrowPoint_right.getAttribute(L"y")));
								leftBrowP[0] = Point(dc.WstringToInt(leftBrowPoint_top.getAttribute(L"x")),dc.WstringToInt(leftBrowPoint_top.getAttribute(L"y")));
								leftBrowP[1] = Point(dc.WstringToInt(leftBrowPoint_buttom.getAttribute(L"x")),dc.WstringToInt(leftBrowPoint_buttom.getAttribute(L"y")));
								MP leftBrow = aF.chm.createMP(cleftBrowModelFileName,leftBrowP);//左眼眉MP
								sep.leftBrow = leftBrow;
								Mat leftPupilModel = imread(cleftEyePupilModelFileName,-1);
								sep.leftPupilModel = leftPupilModel;
								Mat leftEyeBrow = imread(cleftEyeBrowModelFileName,-1);
								sep.leftEyeBrow = leftEyeBrow;

							XMLNode rightEyeNode = eyeNode.getChildNode(L"righteye");//	取右眼lefteye
								string crightEyeModelFileName = dict + dc.WstringToString(rightEyeNode.getAttribute(L"eyemodelfile"));//		取右眼模板文件 eyemodelfile
								string crightEyeBrowModelFileName = dict + dc.WstringToString(rightEyeNode.getAttribute(L"eyewithbrowmodelfile"));//		取右眼眉模板文件 browmodelfile
								string crightBrowModelFileName = dict + dc.WstringToString(rightEyeNode.getAttribute(L"browmodelfile"));//		取右眼眉模板文件 browmodelfile
								string crightEyePupilModelFileName = dict + dc.WstringToString(rightEyeNode.getAttribute(L"pupilmodelfile"));//		取右眼珠模板文件
								XMLNode rightEyePointsNode = rightEyeNode.getChildNode(L"eyepoint");//		取右眼点 eyepoint
										XMLNode rightEyePoint_left = rightEyePointsNode.getChildNode(L"left");//			取左点
										XMLNode rightEyePoint_right = rightEyePointsNode.getChildNode(L"right");//			取右点
										XMLNode rightEyePoint_top = rightEyePointsNode.getChildNode(L"top");//			取上点
										XMLNode rightEyePoint_buttom = rightEyePointsNode.getChildNode(L"buttom");//			取下点
								Point rightEyeP[4];
								rightEyeP[2] = Point(dc.WstringToInt(rightEyePoint_left.getAttribute(L"x")),dc.WstringToInt(rightEyePoint_left.getAttribute(L"y")));
								rightEyeP[3] = Point(dc.WstringToInt(rightEyePoint_right.getAttribute(L"x")),dc.WstringToInt(rightEyePoint_right.getAttribute(L"y")));
								rightEyeP[0] = Point(dc.WstringToInt(rightEyePoint_top.getAttribute(L"x")),dc.WstringToInt(rightEyePoint_top.getAttribute(L"y")));
								rightEyeP[1] = Point(dc.WstringToInt(rightEyePoint_buttom.getAttribute(L"x")),dc.WstringToInt(rightEyePoint_buttom.getAttribute(L"y")));
								MP rightEye = aF.chm.createMP(crightEyeModelFileName,rightEyeP);
								sep.rightEye = rightEye;

								XMLNode rightBrowPointsNode = rightEyeNode.getChildNode(L"browpoint");//		取右眼点 eyepoint
									XMLNode rightBrowPoint_left = rightBrowPointsNode.getChildNode(L"left");//			取左点
									XMLNode rightBrowPoint_right = rightBrowPointsNode.getChildNode(L"right");//			取右点
									XMLNode rightBrowPoint_top = rightBrowPointsNode.getChildNode(L"top");//			取上点
									XMLNode rightBrowPoint_buttom = rightBrowPointsNode.getChildNode(L"buttom");//			取下点
								Point rightBrowP[4];
								rightBrowP[2] = Point(dc.WstringToInt(rightBrowPoint_left.getAttribute(L"x")),dc.WstringToInt(rightBrowPoint_left.getAttribute(L"y")));
								rightBrowP[3] = Point(dc.WstringToInt(rightBrowPoint_right.getAttribute(L"x")),dc.WstringToInt(rightBrowPoint_right.getAttribute(L"y")));
								rightBrowP[0] = Point(dc.WstringToInt(rightBrowPoint_top.getAttribute(L"x")),dc.WstringToInt(rightBrowPoint_top.getAttribute(L"y")));
								rightBrowP[1] = Point(dc.WstringToInt(rightBrowPoint_buttom.getAttribute(L"x")),dc.WstringToInt(rightBrowPoint_buttom.getAttribute(L"y")));
								MP rightBrow = aF.chm.createMP(crightBrowModelFileName,rightBrowP);
								sep.rightBrow = rightBrow;
								Mat rightPupilModel = imread(crightEyePupilModelFileName,-1);
								sep.rightPupilModel = rightPupilModel;
								Mat rightEyeBrowModel = imread(crightEyeBrowModelFileName,-1);
								sep.rightEyeBrow = rightEyeBrowModel;

	
						XMLNode mouthNode = expressionNode.getChildNode(L"mouth"); //取嘴
						string cmouthModelFileName = dict + dc.WstringToString(mouthNode.getAttribute(L"mouthmodelfile"));//		取嘴模板文件
						XMLNode mouthPoint = mouthNode.getChildNode(L"mouthpoint"); //		取嘴点
						XMLNode mouthPoint_left = mouthPoint.getChildNode(L"left");//			取左点
						XMLNode mouthPoint_right = mouthPoint.getChildNode(L"right");//			取右点
						XMLNode mouthPoint_top = mouthPoint.getChildNode(L"top");//			取上点
						XMLNode mouthPoint_buttom = mouthPoint.getChildNode(L"buttom");//			取下点
						Point mouthP[4];
						mouthP[2] = Point(dc.WstringToInt(mouthPoint_left.getAttribute(L"x")),dc.WstringToInt(mouthPoint_left.getAttribute(L"y")));
						mouthP[3] = Point(dc.WstringToInt(mouthPoint_right.getAttribute(L"x")),dc.WstringToInt(mouthPoint_left.getAttribute(L"y")));
						mouthP[0] = Point(dc.WstringToInt(mouthPoint_top.getAttribute(L"x")),dc.WstringToInt(mouthPoint_left.getAttribute(L"y")));
						mouthP[1] = Point(dc.WstringToInt(mouthPoint_buttom.getAttribute(L"x")),dc.WstringToInt(mouthPoint_left.getAttribute(L"y")));
						MP mouth = aF.chm.createMP(cmouthModelFileName,mouthP);
						sep.mouth = mouth;

						expressions.push_back(sep);//表情库里增加一套
				}
						
				XMLNode noseNode =xHeadNode.getChildNode(L"nose");//取鼻子
				string cnoseModelFileName = dict + dc.WstringToString(noseNode.getAttribute(L"nosemodelfile"));//	取鼻子模板文件
				XMLNode noseTipPointNode =noseNode.getChildNode(L"tippoint");//	取鼻尖点
				XMLNode noseTipPointNode_left =noseTipPointNode.getChildNode(L"left");//		取左点
				XMLNode noseTipPointNode_right =noseTipPointNode.getChildNode(L"right");//		取右点
				XMLNode noseTipPointNode_buttom =noseTipPointNode.getChildNode(L"buttom");//		取下点
				XMLNode noseTipPointNode_middle =noseTipPointNode.getChildNode(L"middle");//		取中点
				Point noseTipP[5];
				noseTipP[2] = Point(dc.WstringToInt(noseTipPointNode_left.getAttribute(L"x")),dc.WstringToInt(noseTipPointNode_left.getAttribute(L"y")));
				noseTipP[3] = Point(dc.WstringToInt(noseTipPointNode_right.getAttribute(L"x")),dc.WstringToInt(noseTipPointNode_right.getAttribute(L"y")));
				noseTipP[1] = Point(dc.WstringToInt(noseTipPointNode_buttom.getAttribute(L"x")),dc.WstringToInt(noseTipPointNode_buttom.getAttribute(L"y")));
				noseTipP[0] = Point(dc.WstringToInt(noseTipPointNode_middle.getAttribute(L"x")),dc.WstringToInt(noseTipPointNode_middle.getAttribute(L"y")));

				XMLNode noseBridgePointNode =noseNode.getChildNode(L"bridgepoint");//	取鼻梁点
				XMLNode noseBridgePointNode_top =noseBridgePointNode.getChildNode(L"top");//		取下点
				noseTipP[4] = Point(dc.WstringToInt(noseBridgePointNode_top.getAttribute(L"x")),dc.WstringToInt(noseBridgePointNode_top.getAttribute(L"y")));
				MP nose = aF.chm.createMP(cnoseModelFileName,noseTipP);

				shmt.expressions = expressions;
				shmt.nose = nose;

				headModels.push_back(shmt);
			}	
			aF.chm.headModels = headModels;
			aF.chm.init();
		}
	}
	aF.FaceChange(isHair);
	//aF.FaceChange(faceModel,leftEyeModel,rightEyeModel,leftEyeWithBrowModel,rightEyeWithBrowModel,leftEyePupilModel,rightEyePupilModel,mouthModel,noseModel,isHair);
	

}