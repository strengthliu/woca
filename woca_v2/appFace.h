#ifndef APPFACE_H
#define APPFACE_H
#include "parameters.h"
#include "modelconfiger.h"
#include "face.h"

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>



using namespace cv;
struct EyeCircle{
	Point2i c;
	int radius;
};

class appFace{
public:
	Mat imageOrigine;
	Mat imageContourSM;
	Mat imageFaceContourSM;
	Mat imageHairContourSM; 
	Mat imageRealHairContourSM;
	Mat imageRealFaceContourSM;
	Mat imageRealContourSM;
	appFace(Mat _imageOrigine, Mat _imageContour);

	Mat oSourceImage;
	Mat lineImage;
	static void autoChangeSize(Mat &image, Mat &trimap, int longer = 640);//640*480
	
	void FaceChange(Mat faceModel, Mat leftEye, Mat rightEye,Mat leftEyeWithBrowModel,Mat rightEyeWithBrowModel,Mat leftEyePupilModel,Mat rightEyePupilModel,Mat mouthModel,Mat noseModel,int isHair);
	void FaceChange(int isHair);
	void FaceDetection();
	Mat generateTrimapOfFace(int dSize = 7);
	
	//REALMODE��дʵ�档��ʵ����ٴ�С��λ�ã����Խӽ�1��ϵ��������١�
	static const int REALMODE = 1;
	//REALMODEPLUS��дʵ��ǿ�档��پ�ȷ���ϡ�
	static const int REALMODEPLUS = 8; 
	//QMODE����ģ���������������Ŵ�ģ����١�
	static const int QMODE = 2;
	//QFITMODE������ģ���񣬻��ˣ������ı����Σ���ģ�峬������ʱ����ͷ���ڣ������ڵ����֣����ϱ��ܽ��ϣ�����ģ���������Ƿ�������խ����������ģ�����Σ�
	static const int QFITMODE = 3;
	//QRESIZEMODE,ͷ�������Ӧģ�塣�������������ڻ�û����á�
	static const int QRESIZEMODE = 4;
	//LINEMODE,������
	static const int LINEMODE = 5;
	//BLOCKMODE,����������
	static const int BLOCKMODE = 6;
	//DRAWMODE,ת�ֻ�
	static const int DRAWMODE = 7; 

	Mat debugFrame;//����չʾ���Կ�ı���ͼ
	bool debug;
	void debugFace();
	void debugEyes();
	void debugNose();
	void debugMouth();

	//=======================================================================
	//the rectangles detected by opencv.
	Point eyesPoint[8] ;//��ʵ�۾��������
	Point browPoint[8] ;//��ʵüë�������
	Point mouthPoint[4];
	Point nosePoint[5];
	//Point* eyesPoint ;
	CHeadModels chm;//������װ�����е�ģ����顣�°湦�ܡ�
	void setHeadModel(string headName,string expressionName); //�������ģ�����
	//***************************************************************

	Point* getEyePoint(Mat leftEyeROI_l,Mat leftEyeROI_r,Mat rightEyeROI_r,Mat rightEyeROI_l,Mat leftEyeROI_,Mat rightEyeROI_);
	Point* getBrowPoint(Mat bmask1,Mat bmask11); 
	Point getEyeModelPoint(Mat model,int lr);
	//Mat resizeModel(Mat modelImg,Point ml,Point mr,Point mt,Point el,Point er,Point et);
	Mat resizeModel(Mat modelImg,Point ml,Point mr,Point mt,Point mb,Point el,Point er,Point et,Point eb,int mode);
	Mat resizeModel2(Mat modelImg,Point ml,Point mr,Point mt,Point mb,Point el,Point er,Point et,Point eb,int mode);
	Point* getNosePoint(Mat noseROI);
	Point* getMouthPoint(Mat mouthROI);

	//***************************************************************
	//=======================================================================
	

private:
	CascadeClassifier face_cascade;//for detecting face
	CascadeClassifier eyes_cascade;//for detecting eyes
	CascadeClassifier mouth_cascade;//for detecting mouth
	CascadeClassifier nose_cascade;//for detecting nose
	int loadCascadeFile();


	//***************************** ���Ĵ��� ******************************
	//ͼ���е���
	vector<Rect> faces;
	Rect faceChangeRect;//in this rectangle change face.
	Rect faceDetectedRect;//the rectangle detected by opencv.
	Rect faceSampleRect;

	Rect calFaceChangeRect(Mat _maskFace);
	void calFaceParameters();
	void resizeFaceModel(int mode);
	Mat replaceFace(Mat faceModel,Mat &resultImage,int mode);//��resultImageͼ�У���modeģʽ����������faceModel��
	Mat replaceFaceByREALMODEPLUS(Mat faceModel,Mat &resultImage);
	Mat replaceFaceByQMODE(Mat faceModel,Mat &resultImage);
	Mat replaceFaceByDefault(Mat faceModel,Mat &resultImage);
	void replaceFaceAndEyes(Mat &resultImage,int mode);
	void replaceFaceAndEyes(Mat faceModel, Mat leftEye, Mat rightEye,Mat leftEyeWithBrowModel,Mat rightEyeWithBrowModel,
		Mat &resultImage,Mat leftEyePupilModel,Mat rightEyePupilModel,Mat mouthModel,Mat noseModel,int mode);

	Rect faceMiddleRect;
	//ȡ�������
	int getLeftFaceWidth();
	//ȡ�������
	int getRightFaceWidth();


	//***************************** �۵Ĵ��� ******************************
	//eyes
	int eyeNumber;
	Rect eyeDetectedRects[2];//the rectangles detected by opencv.
	Rect browDetectedRects[2];//üë
	//======================================= RECT ========================
	Rect leftEyeRect;
	Rect rightEyeRect;
	Rect dtLeftEyeRect;
	Rect dtRightEyeRect;
	Rect leftBrowRect;
	Rect rightBrowRect;
	Rect leftEyeWithBrowRect;
	Rect rightEyeWithBrowRect;
	Rect leftEyePupilRect;
	Rect rightEyePupilRect;
	EyeCircle eyeCircles[2];//
	bool aroundEye(int _row, int _col);//give a position
	vector<Rect> detectEyes(Mat _face);//�����м���۾�
	vector<Rect> detectEyes(Mat _face,Mat frame);//�����м���۾�
	void setEyesParameters(vector<Rect> __eyes, Rect faces,Mat frame);
	void setEyesParameters(vector<Rect> __eyes, Rect faces);
	void setEyesParameters(vector<Rect> __eyes);
	void resizeEyes(int mode);
	//void replaceEyes(Mat _bgraFrameLight,Mat noseSample,Rect noseRect,Mat maskRealFace,int mode);
	void replaceEyes(Mat _bgraFrameLight,Mat leftEye, Mat rightEye,Mat leftEyeWithBrowModel,
						Mat rightEyeWithBrowModel, Mat leftEyePupilModel,Mat rightEyePupilModel,Mat grayData,int mode);
	Mat replaceEyes(int mode,Mat face);
	
	//***************************** ��Ĵ��� ******************************
	//mouth
	Rect mouthDetectedRect;
	vector<Rect> detectMouth(Mat _face);//�����м����
	void setMouthsParameter(Vector<Rect> mouths);
	void setMouthsParameter(Vector<Rect> mouths,Rect mouthRegion);
	Mat resizeMouth(int mode,Mat face);
	Mat replaceMouth(Mat face,Mat mouthModel,Rect mouthRect,Mat grayData);

	//***************************** ���ӵĴ��� ******************************
	//nose
	Rect noseDetectedRect;
	vector<Rect> detectNose(Mat _face);//�����м�����
	void setNoseParameter(Vector<Rect> noses);
	void setNoseParameter(Vector<Rect> noses,Rect nr);
	//Mat resizeNoseModel(int mode);
	Mat resizeNoseModel(int mode,Mat face);
	Mat replaceNose(Mat _bgraFrameLight,Mat noseModel,Rect noseRect,Mat grayData);

	void postProcessus(Mat &);
	
	//***************************** ͼ����������� ******************************
	//ԭͼ����ת���ٽǶȣ����ܼ�⵽��������˫����ƽ�����û�м�⵽�����ÿ�ͼ������⣬ת���ٶȣ�˫����ƽ��
	double rotateAngle;
	int rotateDetectFaces();//��ת����������û�з���0��
	void simpleFaceDetection1();//first detection of face (not accurate), using opencv
	void simpleFaceDetection();//first detection of face (not accurate), using opencv
	void setFaceParameters(Mat frame);
	int leftFace(Mat img);//�ж�ͼ�е������ĸ�����ƫ��1����2���ң�0û�м�⵽����-1ֻ��һֻ��
	//������ͼ��ת����ʼ��������
	void rotate();

	Mat replaceEyesByREALMODEPLUS(Mat face,Rect &left,Rect &right);
	//void _Rotation(Mat& src, Mat& dst, float TransMat[3][3]);
	//������ͼ�ָ���ԭͼ����.
	void reRotate();


	//for detecting face in YCrCb color space.
	bool approximateFaceColor(int _row, int _col);
	Mat YCrCbImage;
	Mat HSVImage;
	Scalar faceMeanColorHSV;//HSV
	void hsvCompromise(Mat &modelFace);//in hsv color space compromise model and real face
	void bgrCompromise(Mat &modelFace);//in bgr color space compromise model and real face
	//void superimposingTransparent(Vec4b colData,Vec4b bgra_frame_data,int _transparent);
	int superimposingTransparent(int c1,int c2,int a1,int a2);

	//
	int calFirstRowOfContourHuman();
	int firstRowOfContourHuman;


	/*
	int middlePointX(Rect r);
	int middlePointY(Rect r);
	bool betwin(int a,int r1,int r2);
	*/

	void colorBasedFaceDetection();//seconde detection of face
	
	
	void insideComponent(Mat &_maskImage, int min_dif = 0);
	void insideComponent2(Mat &_maskImage, int min_dif = 0);
	void saveImages(Mat resultImage,String fileName);
	void saveImages1(Mat resultImage);



	void changeFace(Mat &_bgraFrameLight,uchar* mask_face_replace_data,Mat faceSample);
	void generateThresholdImage(IplImage *graySource,IplImage *grayDst,double max,int method, int type, double* parameters);
	Mat fillImageWithMask(Mat imgS,Mat imgT,Mat mask,Rect r);
	//================================== ��������е���ʱ�ļ� ===========================
	void initCounter();
	 
	//=====================================  MASK  ==================
	Mat maskFace;
	Mat maskFaceReplace;
	Mat maskRealFace;
	Mat maskHair;
	Mat maskHairReplace; 
	Mat maskRealHair;
	Mat maskContourHuman;

	Mat mask7,mask1,mask6,mask2;
	Mat mask71,mask11,mask61,mask21;
	Mat bmask2,bmask1;
	Mat bmask21,bmask11;
	//Mat bodyLine;
	//======================================  FRAME ======================
	void initTempMat();
	//Mat _bgraFrameLight;
	//Mat _bgraFrame;
	Mat frame;

	//===================================== MODEL  =========================
	Mat faceModel;
	Mat leftEyeModel;
	Mat rightEyeModel;
	Mat leftEyeWithBrowModel;
	Mat rightEyeWithBrowModel;
	Mat leftBrowModel;
	Mat rightBrowModel;
	Mat leftEyePupilModel;
	Mat rightEyePupilModel;
	Mat mouthModel;
	Mat noseModel;

	//====================================== SAMPLE ======================
	Mat faceSample;//��ģ�͵���ʱ�ļ�
	Mat leftEyeSample,rightEyeSample,rightBrowSample,leftBrowSample,leftEyeWithBrowSample,rightEyeWithBrowSample,leftEyePupilSample, rightEyePupilSample;
	Mat faceSampleBGR;
	Mat noseSample;
	Mat bodyWithoutBackground,bodyWithoutBackgroundLight,origionFace,origionFaceLight;
	Mat origionBodyWithoutFace,origionBodyWithoutFaceLight,faceChanged,faceChangedLight;
	Mat bodyWithoutFace,bodyWithoutFaceLight;

	//=======================================VALUE ========================
	double fWidth ;
	double fHeight;
	double resizeEyeRate;

	Mat removeFace(Mat body,Mat face,Mat bodyWithoutFace);
	Mat removeBackground(Mat srcFrame,Mat body);

	//=================================================================================


	//#define picture_name "004.jpg"//ͼƬ���� ���޸�ͼƬ���ƣ�����ֻ���޸���һ�����
	#define template_size 3 //ģ���С ���޸�ģ���С������ֻ���޸���һ�����
	int m,n,height,width;//����ͼ���С
	int p,q;//����ģ���С��Ϊ��̺��޸ķ��㣬��ֱ����template_size
	int result_median,result_average,result_weighted,logo;//����ÿ������Ľ�� �㷨��־λ
	int a[template_size*template_size],b[template_size*template_size];//a��������ð������ b�������ڴ�ģ��ϵ��
	int c[template_size][template_size];//c[][]���ڴ洢��Ȩ(��˹)ģ��ϵ��

	//--------------------------------------------------------------------------------------------------------------------------------
void PrintStar()//-----------------------------------------------------------------------��ӡ�ָ���-------------------------------
{
	for(int i=0;i<80;i++)
		printf("*");
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Weighted()//------------------------------------------------------------------����ģ��ϵ�������ֵ���浽����median��--
{
	//�������ȸ���Ȩ(��˹)ģ��ϵ�� 
	int t,i,j,*y;
	for (t=1;t<=(p+1)/2;t++)//Ϊģ�������м��е�q��Ԫ�ظ�ֵ
	{
		c[(p-1)/2][(q-1)/2-(t-1)]=pow(2.0,(p-t));
		c[(p-1)/2][(q-1)/2+(t-1)]=pow(2.0,(p-t));
	}
	for (t=1;t<=(p-1)/2;t++)//Ϊģ����������ÿ�е�q��Ԫ�ظ�ֵ
	{
		for (j=0;j<p;j++)
		{
			c[(p-1)/2-((p+1)/2-t)][j]=c[(p-1)/2][j]/pow(2.0,((p+1)/2-t));
			c[(p-1)/2+((p+1)/2-t)][j]=c[(p-1)/2][j]/pow(2.0,((p+1)/2-t));
		}
	}
	y=&b[0];                        //ָ�����ָ��һά�����׵�ַ���Դ���һ�δ���
	for (i=0;i<p;i++)
	{
		for (j=0;j<q;j++)
		{
			*y=c[i][j];             //�Ѹ�˹ģ��ϵ���浽һά����b[]
			y++;
		}
	}
	//�������ȸ���Ȩ(��˹)ģ��ϵ��
	int sum=0,sum_template=0;
	for (i=0;i<p*q;i++)
	{
		sum+=a[i]*b[i];              //��Ȩ���
		sum_template+=b[i];          //��ģ��ϵ����
	}
	result_weighted=sum/sum_template;//�ѽ���ݴ浽result_weighted�У���׼��ͨ��Img_Weightedָ����ͼ�������и�ֵ
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Average()//------------------------------------------------------------------����ģ��ϵ�������ֵ���浽����median��---
{
	//�������ȸ���ֵģ��ϵ��
	for (int k=0;k<p*q;k++)
	{
		b[k]=1;//��ֵ�˲�ģ��ϵ����Ϊ1                          
	}
	//�������ȸ���ֵģ��ϵ��
	int i,sum=0,sum_template=0;
	for (i=0;i<p*q;i++)
	{
		sum+=a[i]*b[i];              //��Ȩ���
		sum_template+=b[i];          //��ģ��ϵ����
	}
	result_average=sum/sum_template;//�ѽ���ݴ浽result_average�У���׼��ͨ��Img_Averageָ����ͼ�������и�ֵ
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Median()//-------------------------------------------------------------------ȡ����ֵ���浽����median��---------------
{
	if ((p*q)%2==1)                         //ģ�峤��˻�Ϊ����
	{
		result_median=a[(p*q-1)/2];//�ѽ���ݴ浽result_median�У���׼��ͨ��Img_Medianָ����ͼ�������и�ֵ
	} 
	else                                    //ģ�峤��˻�Ϊż��  ����else����ȥ���ˣ���Ϊ�Ѿ��涨ģ��Ϊ3*3 �˻�������ż��
	{
		result_median=(a[p*q/2]+a[p*q/2-1])/2;
	}
}
//--------------------------------------------------------------------------------------------------------------------------------
void Sort()//----------------------------------------------------------------------------����-------------------------------------
{
	int i,j,k;
	for(k=0;k<p*q-1;k++)                    //ð������
	{
		for(i=0;i<p*q-1-k;i++)
		{
			if(a[i]>a[i+1])
			{
				j=a[i];
				a[i]=a[i+1];
				a[i+1]=j;
			}
		}
	}
	//cout<<"�����"<<a[0]<<" "<<a[1]<<" "<<a[2]<<" "<<a[3]<<" "<<a[4]<<" "<<a[5]<<" "<<a[6]<<" "<<a[7]<<" "<<a[8]<<" "<<endl;
	Select_Median();
}
//--------------------------------------------------------------------------------------------------------------------------------
void Judge_Method()//--------------------------------------------------------------------�ж������ַ����˲�-----------------------
{
	switch(logo)//����3����������У����Լ�break���ѡ��
	{
	case 0: Sort();//logo=0Ϊ��ֵ�˲� �ȵ�Sort����ȥ�����ٵ�Select_Median����ѡ��ֵ
	case 1: Select_Average();//logo=1Ϊ��ֵ�˲�
	case 2: Select_Weighted();//logo=2Ϊ��Ȩ�˲�
	}
}
//--------------------------------------------------------------------------------------------------------------------------------
double Input()//-------------------------------------------------------------------------��ʼ��һЩ��Ҫ����-----------------------
{
	PrintStar();
	printf("ģ��Ĵ�СΪ%d*%d\n",template_size,template_size);
	printf("���ڱ��⣺Original Image-------------------ԭʼͼ��\n");
	printf("���ڱ��⣺Median Filter Image--------------��ֵ�˲����ͼ��\n");
	printf("���ڱ��⣺Average Filter Image-------------��ֵ�˲����ͼ��\n");
	printf("���ڱ��⣺Weighted(Gaussian) Filter Image--��Ȩ(��˹)�˲����ͼ��\n");
	PrintStar();
	p=template_size;//ģ�����
	q=template_size;//ģ�����
	logo=0;
	return 0;
}

//--------------------------------------------------------------------------------------------------------------------------------

IplImage*  removeNoise(IplImage* pImg,int mode)//----------------------------------------------------------------------------������-----------------------------------
{
	; //����IplImageָ��
    //pImg = cvLoadImage("E:\\30%.bmp", 0);//����ͼ��

	IplImage* Img_Median;
	IplImage* Img_Average;
	IplImage* Img_Weighted;

	Img_Median=cvCloneImage(pImg);
	Img_Average=cvCloneImage(pImg);
	Img_Weighted=cvCloneImage(pImg);

	int height;//���������׳�����ֱ��дint height=pImg->height;�������⣬���Զ���͸�ֵ�ֿ�д
	int width;

	height=pImg->height;    //height������ǰ�浥������
	width=pImg->width;
	Input();
	m=height;
	n=width;
	int i,j,g,h,*z;
	for (i=0;i<m-p+1;i++)                   //ģ�����Ͻ���ͼ�������������������ƶ�
	{
		for (j=0;j<n-q+1;j++)
		{
			z=&a[0];                        //ָ�����ָ��һά�����׵�ַ���Դ���һ�δ��������������׳���
			for (g=i;g<i+p;g++)
			{
				for (h=j;h<j+q;h++)
				{
					*z=((uchar *)(pImg->imageData+g*pImg->widthStep))[h];//��ģ���е�����ֵ�浽һ��һά������������׳���
					z++;                    //�޸�ָ���ַ
				}
			}
			Judge_Method();
			((uchar *)(Img_Median->imageData+(i+(p-1)/2)*Img_Median->widthStep))[j+(q-1)/2]=result_median; //��ģ���м�λ��Ԫ���滻Ϊ���
			((uchar *)(Img_Average->imageData+(i+(p-1)/2)*Img_Average->widthStep))[j+(q-1)/2]=result_average; //��ģ���м�λ��Ԫ���滻Ϊ���
			((uchar *)(Img_Weighted->imageData+(i+(p-1)/2)*Img_Weighted->widthStep))[j+(q-1)/2]=result_weighted; //��ģ���м�λ��Ԫ���滻Ϊ���
		}
	}
	printf("��Ȩ(��˹)�˲���ģ��Ϊ��\n");
	for (i=0;i<p;i++)
	{
		for (j=0;j<q;j++)
		{
			printf("c[%d][%d]=%2d ",i,j,c[i][j]);
		}
		printf("\n");
	}

	switch(mode){
	case 1: //��ֵ�˲�
		{
			cvReleaseImage( &Img_Median ); //�ͷ�ͼ��
			return Img_Median; 
		}
	case 2: //��ֵ�˲�
		{
			cvReleaseImage( &Img_Median ); //�ͷ�ͼ��
			cvReleaseImage( &Img_Weighted ); //�ͷ�ͼ��
			return Img_Average;
		}
	case 3: //��Ȩ(��˹)�˲�
		{
			cvReleaseImage( &Img_Median ); //�ͷ�ͼ��
			cvReleaseImage( &Img_Average ); //�ͷ�ͼ��
			return Img_Weighted;
		}
	}

	cvReleaseImage( &Img_Median ); //�ͷ�ͼ��
	cvReleaseImage( &Img_Average ); //�ͷ�ͼ��
	cvReleaseImage( &Img_Weighted ); //�ͷ�ͼ��
}
//--------------------------------------------------------------------------------------------------------------------------------

};






#endif