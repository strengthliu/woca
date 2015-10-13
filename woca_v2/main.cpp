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
	//-- ��������������Trimap��Ҫ�ĳɿ�ͼ��
	if(inputFaceFileName == "none"){
		aF.FaceDetection();
		trimapFaceImage = aF.generateTrimapOfFace(); //��������Trimap��
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

	//�Ƿ����Զ���������TRIMAP������Ĵ������Ҳ��ͬ�����ܸ�������Ǿ�ϲ���һ���Ƿ���������ͼ��ǰ���档
	//������ע�͵���Ϊ�˵��Զ�XML�ļ���
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
	//if(xMainNode.isDeclaration()) // �������ģ�������ļ����Ͷ������ļ�
	{
		aF.chm = CHeadModels(); //��ʼ��ģ���
		int nHead = xMainNode.nChildNode(L"head");//�ж��ٸ�ͷ���� head

		vector<SHeadModel> headModels(0);//nHead);
		for (int i = 0;i< nHead ;i++){ //ѭ��ÿ��ͷ
			XMLNode xHeadNode = xMainNode.getChildNode(i);//ȡ��I��ͷ
			if(!xHeadNode.isEmpty()){ //�����ֵ
				string cHeadName =  dc.WstringToString(xHeadNode.getAttribute(L"name"));//	ȡ��ģ���ļ� facemodelfile
				SHeadModel shmt = aF.chm.createHeadModel(cHeadName);
				XMLNode faceNode = xHeadNode.getChildNode(L"face");//ȡ�� face
				string cfaceModelFileName = dict + dc.WstringToString(faceNode.getAttribute(L"facemodelfile"));//	ȡ��ģ���ļ� facemodelfile
				shmt.faceModel = imread(cfaceModelFileName,-1);
	imwrite("ResultData//BodyWithoutBackground1-2.png",shmt.faceModel);
				int nExpression = xHeadNode.nChildNode(L"expression");
				vector<SExpression> expressions(0);//nExpression);//��ʼ��nExpression������
				for(int iExpression=0;iExpression<nExpression;iExpression++)	{
					SExpression sep;
					XMLNode expressionNode = xHeadNode.getChildNode(L"expression",iExpression);//һ����ȡ���� expression
					string expressionName = dc.WstringToString(expressionNode.getAttribute(L"expressionname"));//	ȡ�������� expressionname
					sep.name = expressionName;//Ϊ��������������֡�

						XMLNode eyeNode = expressionNode.getChildNode(L"eyes"); // ȡ�ñ����µ��۾�
					
							XMLNode leftEyeNode = eyeNode.getChildNode(L"lefteye");//	ȡ����lefteye
								string cleftEyeModelFileName = dict + dc.WstringToString(leftEyeNode.getAttribute(L"eyemodelfile"));//		ȡ������ģ���ļ�
								string cleftEyePupilModelFileName = dict + dc.WstringToString(leftEyeNode.getAttribute(L"pupilmodelfile"));//		ȡ������ģ���ļ�
								string cleftEyeBrowModelFileName = dict + dc.WstringToString(leftEyeNode.getAttribute(L"eyewithbrowmodelfile"));//		ȡ����üģ���ļ�
								string cleftBrowModelFileName = dict + dc.WstringToString(leftEyeNode.getAttribute(L"browmodelfile"));//		ȡ����üģ���ļ�
								XMLNode leftEyePointsNode = leftEyeNode.getChildNode(L"eyepoint");//		ȡ���۵� eyepoint
									XMLNode leftEyePoint_left = leftEyePointsNode.getChildNode(L"left");//			ȡ���
									XMLNode leftEyePoint_right = leftEyePointsNode.getChildNode(L"right");//			ȡ�ҵ�
									XMLNode leftEyePoint_top = leftEyePointsNode.getChildNode(L"top");//			ȡ�ϵ�
									XMLNode leftEyePoint_buttom = leftEyePointsNode.getChildNode(L"buttom");//			ȡ�µ�
								Point leftEyeP[4];
								leftEyeP[2] = Point(dc.WstringToInt(leftEyePoint_left.getAttribute(L"x")),dc.WstringToInt(leftEyePoint_left.getAttribute(L"y")));
								leftEyeP[3] = Point(dc.WstringToInt(leftEyePoint_right.getAttribute(L"x")),dc.WstringToInt(leftEyePoint_right.getAttribute(L"y")));
								leftEyeP[0] = Point(dc.WstringToInt(leftEyePoint_top.getAttribute(L"x")),dc.WstringToInt(leftEyePoint_top.getAttribute(L"y")));
								leftEyeP[1] = Point(dc.WstringToInt(leftEyePoint_buttom.getAttribute(L"x")),dc.WstringToInt(leftEyePoint_buttom.getAttribute(L"y")));
								MP leftEye = aF.chm.createMP(cleftEyeModelFileName,leftEyeP);//����MP
								sep.leftEye = leftEye;

								XMLNode leftBrowPointsNode = leftEyeNode.getChildNode(L"browpoint");//ȡüë
									XMLNode leftBrowPoint_left = leftBrowPointsNode.getChildNode(L"left");//			ȡ���
									XMLNode leftBrowPoint_right = leftBrowPointsNode.getChildNode(L"right");//			ȡ�ҵ�
									XMLNode leftBrowPoint_top = leftBrowPointsNode.getChildNode(L"top");//			ȡ�ϵ�
									XMLNode leftBrowPoint_buttom = leftBrowPointsNode.getChildNode(L"buttom");//			ȡ�µ�
								Point leftBrowP[4];
								leftBrowP[2] = Point(dc.WstringToInt(leftBrowPoint_left.getAttribute(L"x")),dc.WstringToInt(leftBrowPoint_left.getAttribute(L"y")));
								leftBrowP[3] = Point(dc.WstringToInt(leftBrowPoint_right.getAttribute(L"x")),dc.WstringToInt(leftBrowPoint_right.getAttribute(L"y")));
								leftBrowP[0] = Point(dc.WstringToInt(leftBrowPoint_top.getAttribute(L"x")),dc.WstringToInt(leftBrowPoint_top.getAttribute(L"y")));
								leftBrowP[1] = Point(dc.WstringToInt(leftBrowPoint_buttom.getAttribute(L"x")),dc.WstringToInt(leftBrowPoint_buttom.getAttribute(L"y")));
								MP leftBrow = aF.chm.createMP(cleftBrowModelFileName,leftBrowP);//����üMP
								sep.leftBrow = leftBrow;
								Mat leftPupilModel = imread(cleftEyePupilModelFileName,-1);
								sep.leftPupilModel = leftPupilModel;
								Mat leftEyeBrow = imread(cleftEyeBrowModelFileName,-1);
								sep.leftEyeBrow = leftEyeBrow;

							XMLNode rightEyeNode = eyeNode.getChildNode(L"righteye");//	ȡ����lefteye
								string crightEyeModelFileName = dict + dc.WstringToString(rightEyeNode.getAttribute(L"eyemodelfile"));//		ȡ����ģ���ļ� eyemodelfile
								string crightEyeBrowModelFileName = dict + dc.WstringToString(rightEyeNode.getAttribute(L"eyewithbrowmodelfile"));//		ȡ����üģ���ļ� browmodelfile
								string crightBrowModelFileName = dict + dc.WstringToString(rightEyeNode.getAttribute(L"browmodelfile"));//		ȡ����üģ���ļ� browmodelfile
								string crightEyePupilModelFileName = dict + dc.WstringToString(rightEyeNode.getAttribute(L"pupilmodelfile"));//		ȡ������ģ���ļ�
								XMLNode rightEyePointsNode = rightEyeNode.getChildNode(L"eyepoint");//		ȡ���۵� eyepoint
										XMLNode rightEyePoint_left = rightEyePointsNode.getChildNode(L"left");//			ȡ���
										XMLNode rightEyePoint_right = rightEyePointsNode.getChildNode(L"right");//			ȡ�ҵ�
										XMLNode rightEyePoint_top = rightEyePointsNode.getChildNode(L"top");//			ȡ�ϵ�
										XMLNode rightEyePoint_buttom = rightEyePointsNode.getChildNode(L"buttom");//			ȡ�µ�
								Point rightEyeP[4];
								rightEyeP[2] = Point(dc.WstringToInt(rightEyePoint_left.getAttribute(L"x")),dc.WstringToInt(rightEyePoint_left.getAttribute(L"y")));
								rightEyeP[3] = Point(dc.WstringToInt(rightEyePoint_right.getAttribute(L"x")),dc.WstringToInt(rightEyePoint_right.getAttribute(L"y")));
								rightEyeP[0] = Point(dc.WstringToInt(rightEyePoint_top.getAttribute(L"x")),dc.WstringToInt(rightEyePoint_top.getAttribute(L"y")));
								rightEyeP[1] = Point(dc.WstringToInt(rightEyePoint_buttom.getAttribute(L"x")),dc.WstringToInt(rightEyePoint_buttom.getAttribute(L"y")));
								MP rightEye = aF.chm.createMP(crightEyeModelFileName,rightEyeP);
								sep.rightEye = rightEye;

								XMLNode rightBrowPointsNode = rightEyeNode.getChildNode(L"browpoint");//		ȡ���۵� eyepoint
									XMLNode rightBrowPoint_left = rightBrowPointsNode.getChildNode(L"left");//			ȡ���
									XMLNode rightBrowPoint_right = rightBrowPointsNode.getChildNode(L"right");//			ȡ�ҵ�
									XMLNode rightBrowPoint_top = rightBrowPointsNode.getChildNode(L"top");//			ȡ�ϵ�
									XMLNode rightBrowPoint_buttom = rightBrowPointsNode.getChildNode(L"buttom");//			ȡ�µ�
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

	
						XMLNode mouthNode = expressionNode.getChildNode(L"mouth"); //ȡ��
						string cmouthModelFileName = dict + dc.WstringToString(mouthNode.getAttribute(L"mouthmodelfile"));//		ȡ��ģ���ļ�
						XMLNode mouthPoint = mouthNode.getChildNode(L"mouthpoint"); //		ȡ���
						XMLNode mouthPoint_left = mouthPoint.getChildNode(L"left");//			ȡ���
						XMLNode mouthPoint_right = mouthPoint.getChildNode(L"right");//			ȡ�ҵ�
						XMLNode mouthPoint_top = mouthPoint.getChildNode(L"top");//			ȡ�ϵ�
						XMLNode mouthPoint_buttom = mouthPoint.getChildNode(L"buttom");//			ȡ�µ�
						Point mouthP[4];
						mouthP[2] = Point(dc.WstringToInt(mouthPoint_left.getAttribute(L"x")),dc.WstringToInt(mouthPoint_left.getAttribute(L"y")));
						mouthP[3] = Point(dc.WstringToInt(mouthPoint_right.getAttribute(L"x")),dc.WstringToInt(mouthPoint_left.getAttribute(L"y")));
						mouthP[0] = Point(dc.WstringToInt(mouthPoint_top.getAttribute(L"x")),dc.WstringToInt(mouthPoint_left.getAttribute(L"y")));
						mouthP[1] = Point(dc.WstringToInt(mouthPoint_buttom.getAttribute(L"x")),dc.WstringToInt(mouthPoint_left.getAttribute(L"y")));
						MP mouth = aF.chm.createMP(cmouthModelFileName,mouthP);
						sep.mouth = mouth;

						expressions.push_back(sep);//�����������һ��
				}
						
				XMLNode noseNode =xHeadNode.getChildNode(L"nose");//ȡ����
				string cnoseModelFileName = dict + dc.WstringToString(noseNode.getAttribute(L"nosemodelfile"));//	ȡ����ģ���ļ�
				XMLNode noseTipPointNode =noseNode.getChildNode(L"tippoint");//	ȡ�Ǽ��
				XMLNode noseTipPointNode_left =noseTipPointNode.getChildNode(L"left");//		ȡ���
				XMLNode noseTipPointNode_right =noseTipPointNode.getChildNode(L"right");//		ȡ�ҵ�
				XMLNode noseTipPointNode_buttom =noseTipPointNode.getChildNode(L"buttom");//		ȡ�µ�
				XMLNode noseTipPointNode_middle =noseTipPointNode.getChildNode(L"middle");//		ȡ�е�
				Point noseTipP[5];
				noseTipP[2] = Point(dc.WstringToInt(noseTipPointNode_left.getAttribute(L"x")),dc.WstringToInt(noseTipPointNode_left.getAttribute(L"y")));
				noseTipP[3] = Point(dc.WstringToInt(noseTipPointNode_right.getAttribute(L"x")),dc.WstringToInt(noseTipPointNode_right.getAttribute(L"y")));
				noseTipP[1] = Point(dc.WstringToInt(noseTipPointNode_buttom.getAttribute(L"x")),dc.WstringToInt(noseTipPointNode_buttom.getAttribute(L"y")));
				noseTipP[0] = Point(dc.WstringToInt(noseTipPointNode_middle.getAttribute(L"x")),dc.WstringToInt(noseTipPointNode_middle.getAttribute(L"y")));

				XMLNode noseBridgePointNode =noseNode.getChildNode(L"bridgepoint");//	ȡ������
				XMLNode noseBridgePointNode_top =noseBridgePointNode.getChildNode(L"top");//		ȡ�µ�
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