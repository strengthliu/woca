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
	
	//REALMODE，写实版。按实际五官大小、位置，乘以接近1的系数，放五官。
	static const int REALMODE = 1;
	//REALMODEPLUS，写实增强版。五官精确贴合。
	static const int REALMODEPLUS = 8; 
	//QMODE，按模板捏脸，按比例放大模板五官。
	static const int QMODE = 2;
	//QFITMODE，按照模板风格，画人，包括改变脸形：当模板超出人脸时（在头发内，不是遮挡部分，与上边能接上），按模板制作。是否拉宽或变窄人像，以适配模板脸形？
	static const int QFITMODE = 3;
	//QRESIZEMODE,头变大，以适应模板。包括比例。现在还没有想好。
	static const int QRESIZEMODE = 4;
	//LINEMODE,线条化
	static const int LINEMODE = 5;
	//BLOCKMODE,脸萌那样的
	static const int BLOCKMODE = 6;
	//DRAWMODE,转手绘
	static const int DRAWMODE = 7; 

	Mat debugFrame;//用于展示调试框的背景图
	bool debug;
	void debugFace();
	void debugEyes();
	void debugNose();
	void debugMouth();

	//=======================================================================
	//the rectangles detected by opencv.
	Point eyesPoint[8] ;//真实眼睛的坐标点
	Point browPoint[8] ;//真实眉毛的坐标点
	Point mouthPoint[4];
	Point nosePoint[5];
	//Point* eyesPoint ;
	CHeadModels chm;//这里面装了所有的模板表情。新版功能。
	void setHeadModel(string headName,string expressionName); //设置五官模板表情
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


	//***************************** 脸的处理 ******************************
	//图像中的脸
	vector<Rect> faces;
	Rect faceChangeRect;//in this rectangle change face.
	Rect faceDetectedRect;//the rectangle detected by opencv.
	Rect faceSampleRect;

	Rect calFaceChangeRect(Mat _maskFace);
	void calFaceParameters();
	void resizeFaceModel(int mode);
	Mat replaceFace(Mat faceModel,Mat &resultImage,int mode);//在resultImage图中，按mode模式，将脸换成faceModel。
	Mat replaceFaceByREALMODEPLUS(Mat faceModel,Mat &resultImage);
	Mat replaceFaceByQMODE(Mat faceModel,Mat &resultImage);
	Mat replaceFaceByDefault(Mat faceModel,Mat &resultImage);
	void replaceFaceAndEyes(Mat &resultImage,int mode);
	void replaceFaceAndEyes(Mat faceModel, Mat leftEye, Mat rightEye,Mat leftEyeWithBrowModel,Mat rightEyeWithBrowModel,
		Mat &resultImage,Mat leftEyePupilModel,Mat rightEyePupilModel,Mat mouthModel,Mat noseModel,int mode);

	Rect faceMiddleRect;
	//取左脸宽度
	int getLeftFaceWidth();
	//取右脸宽度
	int getRightFaceWidth();


	//***************************** 眼的处理 ******************************
	//eyes
	int eyeNumber;
	Rect eyeDetectedRects[2];//the rectangles detected by opencv.
	Rect browDetectedRects[2];//眉毛
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
	vector<Rect> detectEyes(Mat _face);//在脸中检测眼睛
	vector<Rect> detectEyes(Mat _face,Mat frame);//在脸中检测眼睛
	void setEyesParameters(vector<Rect> __eyes, Rect faces,Mat frame);
	void setEyesParameters(vector<Rect> __eyes, Rect faces);
	void setEyesParameters(vector<Rect> __eyes);
	void resizeEyes(int mode);
	//void replaceEyes(Mat _bgraFrameLight,Mat noseSample,Rect noseRect,Mat maskRealFace,int mode);
	void replaceEyes(Mat _bgraFrameLight,Mat leftEye, Mat rightEye,Mat leftEyeWithBrowModel,
						Mat rightEyeWithBrowModel, Mat leftEyePupilModel,Mat rightEyePupilModel,Mat grayData,int mode);
	Mat replaceEyes(int mode,Mat face);
	
	//***************************** 嘴的处理 ******************************
	//mouth
	Rect mouthDetectedRect;
	vector<Rect> detectMouth(Mat _face);//在脸中检测嘴
	void setMouthsParameter(Vector<Rect> mouths);
	void setMouthsParameter(Vector<Rect> mouths,Rect mouthRegion);
	Mat resizeMouth(int mode,Mat face);
	Mat replaceMouth(Mat face,Mat mouthModel,Rect mouthRect,Mat grayData);

	//***************************** 鼻子的处理 ******************************
	//nose
	Rect noseDetectedRect;
	vector<Rect> detectNose(Mat _face);//在脸中检测鼻子
	void setNoseParameter(Vector<Rect> noses);
	void setNoseParameter(Vector<Rect> noses,Rect nr);
	//Mat resizeNoseModel(int mode);
	Mat resizeNoseModel(int mode,Mat face);
	Mat replaceNose(Mat _bgraFrameLight,Mat noseModel,Rect noseRect,Mat grayData);

	void postProcessus(Mat &);
	
	//***************************** 图像处理基础方法 ******************************
	//原图像旋转多少角度，可能检测到脸，并且双眼相平。如果没有检测到脸，用抠图的脸检测，转多少度，双眼相平。
	double rotateAngle;
	int rotateDetectFaces();//旋转检测脸。如果没有返回0。
	void simpleFaceDetection1();//first detection of face (not accurate), using opencv
	void simpleFaceDetection();//first detection of face (not accurate), using opencv
	void setFaceParameters(Mat frame);
	int leftFace(Mat img);//判断图中的脸向哪个方向偏。1向左，2向右，0没有检测到脸，-1只有一只眼
	//将所有图旋转，开始做漫画。
	void rotate();

	Mat replaceEyesByREALMODEPLUS(Mat face,Rect &left,Rect &right);
	//void _Rotation(Mat& src, Mat& dst, float TransMat[3][3]);
	//将所有图恢复回原图方向.
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
	//================================== 程序过程中的临时文件 ===========================
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
	Mat faceSample;//脸模型的临时文件
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


	//#define picture_name "004.jpg"//图片名称 若修改图片名称，程序只需修改这一处便可
	#define template_size 3 //模板大小 若修改模板大小，程序只需修改这一处便可
	int m,n,height,width;//定义图像大小
	int p,q;//定义模版大小，为编程和修改方便，不直接用template_size
	int result_median,result_average,result_weighted,logo;//定义每次运算的结果 算法标志位
	int a[template_size*template_size],b[template_size*template_size];//a数组用于冒泡排序 b数组用于存模版系数
	int c[template_size][template_size];//c[][]用于存储加权(高斯)模板系数

	//--------------------------------------------------------------------------------------------------------------------------------
void PrintStar()//-----------------------------------------------------------------------打印分割线-------------------------------
{
	for(int i=0;i<80;i++)
		printf("*");
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Weighted()//------------------------------------------------------------------根据模版系数计算均值并存到变量median中--
{
	//以下是先赋加权(高斯)模板系数 
	int t,i,j,*y;
	for (t=1;t<=(p+1)/2;t++)//为模板中最中间行的q个元素赋值
	{
		c[(p-1)/2][(q-1)/2-(t-1)]=pow(2.0,(p-t));
		c[(p-1)/2][(q-1)/2+(t-1)]=pow(2.0,(p-t));
	}
	for (t=1;t<=(p-1)/2;t++)//为模板中其余行每行的q个元素赋值
	{
		for (j=0;j<p;j++)
		{
			c[(p-1)/2-((p+1)/2-t)][j]=c[(p-1)/2][j]/pow(2.0,((p+1)/2-t));
			c[(p-1)/2+((p+1)/2-t)][j]=c[(p-1)/2][j]/pow(2.0,((p+1)/2-t));
		}
	}
	y=&b[0];                        //指针从新指到一维数组首地址，以待下一次存数
	for (i=0;i<p;i++)
	{
		for (j=0;j<q;j++)
		{
			*y=c[i][j];             //把高斯模板系数存到一维数组b[]
			y++;
		}
	}
	//以上是先赋加权(高斯)模板系数
	int sum=0,sum_template=0;
	for (i=0;i<p*q;i++)
	{
		sum+=a[i]*b[i];              //加权求和
		sum_template+=b[i];          //其模板系数和
	}
	result_weighted=sum/sum_template;//把结果暂存到result_weighted中，并准备通过Img_Weighted指针往图像像素中赋值
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Average()//------------------------------------------------------------------根据模版系数计算均值并存到变量median中---
{
	//以下是先赋均值模板系数
	for (int k=0;k<p*q;k++)
	{
		b[k]=1;//均值滤波模板系数都为1                          
	}
	//以上是先赋均值模板系数
	int i,sum=0,sum_template=0;
	for (i=0;i<p*q;i++)
	{
		sum+=a[i]*b[i];              //加权求和
		sum_template+=b[i];          //其模板系数和
	}
	result_average=sum/sum_template;//把结果暂存到result_average中，并准备通过Img_Average指针往图像像素中赋值
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Median()//-------------------------------------------------------------------取出中值并存到变量median中---------------
{
	if ((p*q)%2==1)                         //模板长宽乘积为奇数
	{
		result_median=a[(p*q-1)/2];//把结果暂存到result_median中，并准备通过Img_Median指针往图像像素中赋值
	} 
	else                                    //模板长宽乘积为偶数  现在else可以去掉了，因为已经规定模板为3*3 乘积不会是偶数
	{
		result_median=(a[p*q/2]+a[p*q/2-1])/2;
	}
}
//--------------------------------------------------------------------------------------------------------------------------------
void Sort()//----------------------------------------------------------------------------排序-------------------------------------
{
	int i,j,k;
	for(k=0;k<p*q-1;k++)                    //冒泡排序
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
	//cout<<"排序后"<<a[0]<<" "<<a[1]<<" "<<a[2]<<" "<<a[3]<<" "<<a[4]<<" "<<a[5]<<" "<<a[6]<<" "<<a[7]<<" "<<a[8]<<" "<<endl;
	Select_Median();
}
//--------------------------------------------------------------------------------------------------------------------------------
void Judge_Method()//--------------------------------------------------------------------判断用哪种方法滤波-----------------------
{
	switch(logo)//这里3种情况都运行，可以加break语句选择
	{
	case 0: Sort();//logo=0为中值滤波 先到Sort函数去排序，再到Select_Median函数选中值
	case 1: Select_Average();//logo=1为均值滤波
	case 2: Select_Weighted();//logo=2为加权滤波
	}
}
//--------------------------------------------------------------------------------------------------------------------------------
double Input()//-------------------------------------------------------------------------初始化一些必要参数-----------------------
{
	PrintStar();
	printf("模版的大小为%d*%d\n",template_size,template_size);
	printf("窗口标题：Original Image-------------------原始图像\n");
	printf("窗口标题：Median Filter Image--------------中值滤波后的图像\n");
	printf("窗口标题：Average Filter Image-------------均值滤波后的图像\n");
	printf("窗口标题：Weighted(Gaussian) Filter Image--加权(高斯)滤波后的图像\n");
	PrintStar();
	p=template_size;//模版的行
	q=template_size;//模版的列
	logo=0;
	return 0;
}

//--------------------------------------------------------------------------------------------------------------------------------

IplImage*  removeNoise(IplImage* pImg,int mode)//----------------------------------------------------------------------------主函数-----------------------------------
{
	; //声明IplImage指针
    //pImg = cvLoadImage("E:\\30%.bmp", 0);//载入图像

	IplImage* Img_Median;
	IplImage* Img_Average;
	IplImage* Img_Weighted;

	Img_Median=cvCloneImage(pImg);
	Img_Average=cvCloneImage(pImg);
	Img_Weighted=cvCloneImage(pImg);

	int height;//（这里容易出错）若直接写int height=pImg->height;会有问题，所以定义和赋值分开写
	int width;

	height=pImg->height;    //height可以在前面单独定义
	width=pImg->width;
	Input();
	m=height;
	n=width;
	int i,j,g,h,*z;
	for (i=0;i<m-p+1;i++)                   //模版左上角在图像像素区域中逐像素移动
	{
		for (j=0;j<n-q+1;j++)
		{
			z=&a[0];                        //指针从新指到一维数组首地址，以待下一次存数排序（这里容易出错）
			for (g=i;g<i+p;g++)
			{
				for (h=j;h<j+q;h++)
				{
					*z=((uchar *)(pImg->imageData+g*pImg->widthStep))[h];//把模版中的像素值存到一个一维数组里（这里容易出错）
					z++;                    //修改指针地址
				}
			}
			Judge_Method();
			((uchar *)(Img_Median->imageData+(i+(p-1)/2)*Img_Median->widthStep))[j+(q-1)/2]=result_median; //把模版中间位置元素替换为结果
			((uchar *)(Img_Average->imageData+(i+(p-1)/2)*Img_Average->widthStep))[j+(q-1)/2]=result_average; //把模版中间位置元素替换为结果
			((uchar *)(Img_Weighted->imageData+(i+(p-1)/2)*Img_Weighted->widthStep))[j+(q-1)/2]=result_weighted; //把模版中间位置元素替换为结果
		}
	}
	printf("加权(高斯)滤波的模版为：\n");
	for (i=0;i<p;i++)
	{
		for (j=0;j<q;j++)
		{
			printf("c[%d][%d]=%2d ",i,j,c[i][j]);
		}
		printf("\n");
	}

	switch(mode){
	case 1: //中值滤波
		{
			cvReleaseImage( &Img_Median ); //释放图像
			return Img_Median; 
		}
	case 2: //均值滤波
		{
			cvReleaseImage( &Img_Median ); //释放图像
			cvReleaseImage( &Img_Weighted ); //释放图像
			return Img_Average;
		}
	case 3: //加权(高斯)滤波
		{
			cvReleaseImage( &Img_Median ); //释放图像
			cvReleaseImage( &Img_Average ); //释放图像
			return Img_Weighted;
		}
	}

	cvReleaseImage( &Img_Median ); //释放图像
	cvReleaseImage( &Img_Average ); //释放图像
	cvReleaseImage( &Img_Weighted ); //释放图像
}
//--------------------------------------------------------------------------------------------------------------------------------

};






#endif