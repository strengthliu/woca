#include "appFace.h"
#include "xmlParser.h"
#include <iostream>
#include <stdio.h>  
#include <math.h>
#include <vector>
#define M_PI       3.14159265358979323846
using namespace std;

appFace::appFace(Mat _imageOrigine, Mat _imageContourSM){
	_imageOrigine.copyTo(imageOrigine);
	_imageContourSM.copyTo(imageContourSM);
	_imageContourSM.copyTo(maskContourHuman);
	
	if ( loadCascadeFile() < 0 ) 
		cout << "load Cascade Files error!" << endl;
}

int appFace::loadCascadeFile(){
	String face_cascade_name = "haarcascade_frontalface_alt.xml";

	String eyes_cascade_name = "haarcascade_eye.xml";

	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";

	String nose_cascade_name = "haarcascade_mcs_nose.xml";

	if( !face_cascade.load( face_cascade_name ) ){ return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ return -1; };
	if( !nose_cascade.load( nose_cascade_name ) ){ return -1; };
	if( !mouth_cascade.load( mouth_cascade_name ) ){ return -1; };

	return 1;
}

int appFace::getLeftFaceWidth(){
	int _fl = calFirstColOfContour(imageFaceContourSM);
	int _flr = calFirstColOfContour_Row(imageFaceContourSM);
	//边缘小于全脸的1/8，就算是接着的。
	int w = (calLastColOfContour(imageFaceContourSM)-calFirstColOfContour(imageFaceContourSM))/8;
	int wt = 0;
	//uchar* rowData = imageContourSM.ptr<uchar>(_flr);

	for(int i=_fl; i>0; i--){
		int index = _flr*imageContourSM.cols + i;
		if(imageContourSM.data[index]>5)
			wt++;
		else
			break;
	}
	
	cout << "距离左mask:" << wt << "  M的1/15= "<< w<<";  最左列(y,x)："<<_fl<<","<<_flr<<"   左脸宽度："<<faceMiddleRect.x - _fl <<"   MASK宽度："<<imageFaceContourSM.cols<< endl;

	if(wt<w && wt>0)
		return faceMiddleRect.x - _fl;
	else
		return -1;
}

int appFace::getRightFaceWidth(){
	int _fl = calLastColOfContour(imageFaceContourSM);
	int _flr = calLastColOfContour_Row(imageFaceContourSM);
	//边缘小于全脸的1/8，就算是接着的。
	int w = (calLastColOfContour(imageFaceContourSM)-calFirstColOfContour(imageFaceContourSM))/8;
	int wt = 0;
	//uchar* rowData = imageContourSM.ptr<uchar>(_flr);
	for(int i=_fl; i<imageContourSM.cols; i++){
		//TODO 没有脸到边的情况
		int index = _flr*imageContourSM.cols + i;
		if(imageContourSM.data[index]>5)
			wt++;
		else
			break;
	}
	cout << "距离右mask:" <<wt << "  M的1/15= "<< w << ";  最右列(y,x)："<<_fl<<","<<_flr<<"   右脸宽度："<< _fl - faceMiddleRect.x  <<"   MASK宽度："<<imageFaceContourSM.cols<< endl;

	if(wt<w && wt>0){
		return _fl - this->faceMiddleRect.x;
	}
	else
		return -1;
}


void appFace::autoChangeSize(Mat &image, Mat &trimap, int longer)
{
	int m = image.rows, n = image.cols;
	if (m < longer) return;
	double rate = 1.0 * longer / m;

	resize(image, image, Size(0, 0), rate, rate);
	resize(trimap, trimap, Size(0, 0), rate, rate);
}

bool appFace::aroundEye(int _row, int _col){
	int a = _col - eyeCircles[0].c.x, b = _row - eyeCircles[0].c.y;
	int d1 = a*a + b*b;
	int c = eyeCircles[0].radius;
	if ( d1 < c*c)
		return true;
	a = _col - eyeCircles[1].c.x, b = _row - eyeCircles[1].c.y;
	int d2 = a*a + b*b;
	c = eyeCircles[1].radius;
	if (d2 < c*c)
		return true;
	return false;
}

bool appFace::approximateFaceColor(int _row, int _col){
	Vec3b yCrCb = YCrCbImage.at<Vec3b>(_row, _col);
	int cr = yCrCb[1] - CRMEAN, cb = yCrCb[2] - CBMEAN;
	double mi = -0.5 * (S11*cr*cr + 2*S12*cb*cr + S22*cb*cb);
	double propa = exp(mi);
	return (propa > 0.5);
}

int appFace::calFirstRowOfContourHuman(){//改写为调用新方法
	return calFirstRowOfContour(imageContourSM);
}

int appFace::calFirstRowOfContour(Mat countourSM){
	for (int _row = 0; _row < countourSM.rows ; _row++)
	{
		uchar* rowData = countourSM.ptr<uchar>(_row);
		for (int _col = 0 ; _col < countourSM.cols ;_col++)
		{
			if (rowData[_col] > 5)
				return _row;
		}
	}
	return -1;
}

int appFace::calLastRowOfContour(Mat countourSM){
	for (int _row = countourSM.rows; _row >0 ; _row--)
	{
		uchar* rowData = countourSM.ptr<uchar>(_row);
		for (int _col = countourSM.cols ; _col >0 ;_col--)
		{
			if (rowData[_col] == 255){
				cout <<  _row << endl;
				return _row;
			}
		}
	}

	return -1;
}

int appFace::calFirstColOfContour(Mat countourSM){
	for (int _col = 10 ; _col < countourSM.cols ;_col++)
	{
		//uchar* rowData = countourSM.ptr<uchar>(_col);
		for (int _row = 0; _row < countourSM.rows ; _row++)
		{
			//int index = _row*countourSM.cols+_col;
			int index = _row*countourSM.cols+_col;
			if (countourSM.data[index] > 5){
//				cout <<  _row << "  -  " << _col << endl;
				return _col;
			}
		}
	}
	return -1;
}

int appFace::calLastColOfContour(Mat countourSM){
	for (int _col = countourSM.cols ; _col >0 ;_col--)
	{
		for (int _row = countourSM.rows; _row >0 ; _row--)
		{
			int index = _row*countourSM.cols+_col;
			if (countourSM.data[index] > 5){
//				cout <<  _row << "  -  " << _col << endl;
				return _col;
			}
		}
	}
	return -1;
}

int appFace::calFirstColOfContour_Row(Mat countourSM){
	for (int _col = 10 ; _col < countourSM.cols ;_col++)
	{
		//uchar* rowData = countourSM.ptr<uchar>(_col);
		for (int _row = 0; _row < countourSM.rows ; _row++)
		{
			//int index = _row*countourSM.cols+_col;
			int index = _row*countourSM.cols+_col;
			if (countourSM.data[index] > 5){
//				cout <<  _row << "  -  " << _col << endl;
				return _row;
			}
		}
	}
	return -1;
}

int appFace::calLastColOfContour_Row(Mat countourSM){
	for (int _col = countourSM.cols ; _col >0 ;_col--)
	{
		uchar* rowData = countourSM.ptr<uchar>(_col);
		for (int _row = countourSM.rows; _row >0 ; _row--)
		{
			int index = _row*countourSM.cols+_col;
			if (countourSM.data[index] > 5){
//				cout <<  _row << "  -  " << _col << endl;
				return _row;
			}
		}
	}
	return -1;
}

vector<Rect> appFace::detectEyes(Mat _face){//在脸中检测眼睛
		//开始检测眼睛，
	Mat faceROI = _face;//frame_gray( faceDetectedRect );
	std::vector<Rect> eyes;
	int minSize = faceROI.rows / 5;
	eyes_cascade.detectMultiScale( faceROI, eyes, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));
		
	if(eyes.size()>2){	
		eyes.erase(eyes.begin()+2,eyes.end());//删除区间[2,结尾];区间从0开始
	}
	else if(eyes.size()>0)
		return eyes;
	else
		cout << " 没有检测到眼睛。 " << endl;
	return eyes;
}

Mat appFace::createROI(Mat m,string name){
		//抠出目标
		Mat _gray;
		Mat mROI;
		//leftEyeROI.convertTo(leftEyeROI, leftEyeROI.type(), 1, 0); // ROI子图像之间的复制
		
		//先调整原图平均亮度
		IplImage* _imageBgr = &IplImage(m);
		Mat _tar ;
		m.copyTo(_tar);
		IplImage* _imageTar = &IplImage(_tar);
		set_avg_gray(_imageTar,_imageTar,(double)128.0);//亮度平衡处理
		imwrite("MedianData//"+name+"_128Light.jpg", _tar);
		
		//cvSmooth(_imageBgr,_imageTar);
		cvSmooth(_imageTar,_imageTar);

		imwrite("MedianData//"+name+"_le0.jpg", _tar);
		cvtColor( _tar, _gray, CV_BGR2GRAY );
		Mat mt2 = Mat(_gray);
		//高斯去噪
		//eye_gray_r = removeNoise(eye_gray_r,3);
		IplImage* eye_gray_r = &IplImage(_gray);
		imwrite("MedianData//"+name+"_le1.jpg", mt2);

		//直方图后，效果不好。
		equalizeHist( _gray, _gray );
		//imwrite("MedianData//le2.jpg", mt2);
		Mat edge;
		Canny(_gray, edge, 50, 150);
		imwrite("MedianData//"+name+"_leedge.jpg", edge);

		IplImage *graySource = &IplImage(_gray);
		CvSize imageSize1 = cvSize(_gray.cols, _gray.rows);
		IplImage *grayDst = cvCreateImage(imageSize1, IPL_DEPTH_8U, 1);
		int offSet = 0 ;
		if(m.cols > m.rows)
			offSet = ((float)(m.rows));
		else
			offSet = ((float)(m.cols));
		cout << "m.rows:" << m.rows << " - m.cols:" << m.cols <<  " . offSet: " << offSet << endl;
		//generateThresholdImage(graySource,grayDst,offSet,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);
		//70是个测试的值，不知道为什么，暂时使用着。
		cvThreshold(graySource, grayDst, offSet, 255,CV_THRESH_BINARY);
		//cvAdaptiveThreshold(graySource, grayDst, 5, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);
		//cvShowImage("adaptiveThresh", Iat);


		Mat mt = Mat(grayDst);
		imwrite("MedianData//"+name+"_le2.jpg", mt);
		imwrite("MedianData//"+name+"_le.jpg", mt2);
		return mt;
}

void appFace::setEyesParameters(vector<Rect> __eyes){
	eyeNumber = __eyes.size();
	int leftN,rightN;
	if(eyeNumber == 2){
		if(__eyes[0].x+__eyes[0].width > __eyes[1].x+__eyes[1].width/2){
			leftN = 1;rightN = 0;
		}else {
			leftN = 0;rightN = 1;
		}
		eyeDetectedRects[0] = Rect(faceDetectedRect.x+__eyes[leftN].x,faceDetectedRect.y+__eyes[leftN].y,__eyes[leftN].width,__eyes[leftN].height);
		eyeDetectedRects[1] = Rect(faceDetectedRect.x+__eyes[rightN].x,faceDetectedRect.y+__eyes[rightN].y,__eyes[rightN].width,__eyes[rightN].height);

		Mat mask7,mask1,mask6;
		Mat roi ;
		imageOrigine(eyeDetectedRects[0]).copyTo(roi);
		//=======================================================================
		mask1 = createROI(roi,"eyeDetectedRects[0]1",1,2,3); // 中值算法 1/3半径
		mask6 = createROI(roi,"eyeDetectedRects[0]6",0,2,3); // 中值算法 1/3半径
		mask7 = createROI(roi,"eyeDetectedRects[0]7",0,2,roi.rows/5); // 中值算法 5半径
		//=======================================================================
		Rect rBrow = Rect(0,0,mask1.cols,mask1.rows/4);
		imwrite("MedianData//eyeDetectedRects[0]10.png",mask1);
		removeBrow(mask1,rBrow);//MASK去掉眉毛部分
		filterBlock(mask7,mask1,true); // 过滤细线MASK图
		filterBlock(mask6,mask1,true); // 过滤粗线MASK图
		imwrite("MedianData//eyeDetectedRects[0]71.png",mask7);

		Mat mask71,mask11,mask61;
		imageOrigine(eyeDetectedRects[1]).copyTo(roi);
		//=======================================================================
		mask11 = createROI(roi,"eyeDetectedRects[1]1",1,2,3); // 中值算法 1/3半径
		mask61 = createROI(roi,"eyeDetectedRects[1]6",0,2,3); // 中值算法 1/3半径
		mask71 = createROI(roi,"eyeDetectedRects[1]7",0,2,roi.rows/5); // 中值算法 5半径
		//=======================================================================
		rBrow = Rect(0,0,mask11.cols,mask11.rows/4);
		imwrite("MedianData//eyeDetectedRects[1]10.png",mask1);
		removeBrow(mask11,rBrow);//MASK去掉眉毛部分
		filterBlock(mask71,mask11,true); // 过滤细线MASK图
		filterBlock(mask61,mask11,true); // 过滤粗线MASK图
		imwrite("MedianData//eyeDetectedRects[1]71.png",mask71);

		eyesPoint = getEyePoint(mask6,mask7,mask61,mask71); // 根据左右眼的ROI，计算点。

	} else 
	{
		for(int i=0;i<eyeNumber;i++){
			eyeDetectedRects[i] = Rect(faceDetectedRect.x+__eyes[i].x,faceDetectedRect.y+__eyes[i].y,__eyes[i].width,__eyes[i].height);
		}
	}
	cout << "" << endl;
	//Mat roi = debugFrame(eyeDetectedRects[0]);
	//createROI(roi,"eyes");

}
void appFace::setMouthsParameter(Vector<Rect> mouths){

	//******************************* 验证嘴是否正确 ******************************************************
	int x1 = int(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2); //middlePointX(eyeDetectedRects[0]); // 双眼中心XY坐标
	int y1 = int(eyeDetectedRects[0].y+eyeDetectedRects[0].height/2);//middlePointY(eyeDetectedRects[0]);
	int x2 = int(eyeDetectedRects[1].x+eyeDetectedRects[1].width/2);//middlePointX(eyeDetectedRects[1]);
	int y2 = int(eyeDetectedRects[1].y+eyeDetectedRects[1].height/2);//middlePointY(eyeDetectedRects[1]);
	int _y = 0;int _ym=-1;
	for( size_t mi = 0; mi < mouths.size(); mi++ )
	{
		//如果嘴不与其它部分重合，且在两眼之间，就算是嘴。
		int mx = int(faceDetectedRect.x+mouths[mi].x+mouths[mi].width/2);
		int my = int(faceDetectedRect.y+mouths[mi].y+mouths[mi].height/2);
		cout << mx << ": " << x1 << ": " << x2<< "|| my :"<<my<<"  "<<y1 << "  "<< y2 << endl;
		bool _betwin = false;
		if((mx>=x1 && mx<=x2) || mx<= x1 && mx >=x2) _betwin = true;
		if(_betwin && my>y1 && my>y2) 
		{
			//最下面的那一个，是嘴。
			if(mouths[mi].y>_y){
				_y = mouths[mi].y;
				_ym = mi;
			}
		}
	}
	//****************************************************************************************************************
	//找到了一个嘴
	if(_ym>=0){
		this->mouthDetectedRect = Rect(this->faceDetectedRect.x+mouths[_ym].x,this->faceDetectedRect.y+mouths[_ym].y,mouths[_ym].width,mouths[_ym].height);
	} else {
		cout << " 没有找到正确的嘴 " << endl;
	}
}
void appFace::setMouthsParameter(Vector<Rect> mouths,Rect mouthRegion){
	//因为是在指定区域查找的，不需要根据眼睛判断了，只取最下面的一个、最靠眼中心的一个。
	int buttomNum;
	int middle = mouthRegion.width/2;
	if(mouths.size()>0){
		int buttom = 0;
		for(int i=0;i<mouths.size();i++){
			int b = mouths[i].y+mouths[i].height/2;
			if(buttom<b){
				buttom = b;
				buttomNum = i;
			}
		}

		this->mouthDetectedRect = Rect(
			mouthRegion.x+mouths[buttomNum].x,
			mouthRegion.y+mouths[buttomNum].y,
			mouths[buttomNum].width,
			mouths[buttomNum].height);
	}
}

void appFace::setNoseParameter(Vector<Rect> noses){

	//****************************  检查鼻子是否正确  *************************************************************
	int x1 = int(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2); //middlePointX(eyeDetectedRects[0]); // 双眼中心XY坐标
	int y1 = int(eyeDetectedRects[0].y+eyeDetectedRects[0].height/2);//middlePointY(eyeDetectedRects[0]);
	int x2 = int(eyeDetectedRects[1].x+eyeDetectedRects[1].width/2);//middlePointX(eyeDetectedRects[1]);
	int y2 = int(eyeDetectedRects[1].y+eyeDetectedRects[1].height/2);//middlePointY(eyeDetectedRects[1]);
	int _y = 0;int _ym=-1;
	int x3 = int(mouthDetectedRect.x+mouthDetectedRect.width/2);//middlePointX(mouthDetectedRect); // 嘴中心X坐标
	int y3 = int(mouthDetectedRect.y+mouthDetectedRect.height/2);//middlePointY(mouthDetectedRect); // 嘴中心Y坐标

	_y = 0; int _yn = -1;
	for( size_t ni = 0; ni < noses.size(); ni++ )//检测到一个鼻子，但是被下面的条件过滤掉了。
	{
		int nx = int(faceDetectedRect.x+noses[ni].x+noses[ni].width/2); // 鼻子中心X坐标
		int ny = int(faceDetectedRect.y+noses[ni].y+noses[ni].height/2); // 鼻子中心Y坐标
		cout << nx << ": " << x1 << ": " << x2<< " " << x3 << "|| ny :"<<ny<<"  "<<y1 << "  "<< y2 << " " << y3 << endl;
		bool _betwin = false;
		if((nx>=x1 && nx<=x2) || (nx<= x1 && nx >=x2)) _betwin = true;

		if(_betwin && ny>y1 && ny>y2 && ny<y3) 
		{
			//在嘴上面，在眼睛下面，在两眼之间，就算是鼻子。

			//取最下面的那个
			if(noses[ni].y>_y){
				_y = noses[ni].y;
				_yn = ni;
			}
		}
	} 
	//****************************************************************************************************************
	if(_yn>=0){
		this->noseDetectedRect = Rect(this->faceDetectedRect.x+noses[_yn].x,this->faceDetectedRect.y+noses[_yn].y,noses[_yn].width,noses[_yn].height);
	} else {
		cout << " 没有找到鼻子。 " << endl;
	}
}
void appFace::setNoseParameter(Vector<Rect> noses,Rect noseRegion){
	//因为是在指定区域查找的，不需要根据眼睛判断了，只取最靠眼中心的一个,最下面的一个。
	int middle = noseRegion.width/2;
	int middleNum = -1;
	int distance = noseRegion.width;
	for(int i=0; i<noses.size(); i++){
		int d = abs(noses[i].x+(float)(noses[i].width)/2 - middle);
		if(distance > d){
			distance = d;
			middleNum = i;
		}
	}
	if(middleNum>=0)
		this->noseDetectedRect = Rect(
		noseRegion.x+noses[middleNum].x,
		noseRegion.y+noses[middleNum].y,
		noses[middleNum].width,
		noses[middleNum].height);
	else
		cout << " 没有找到鼻子。 " << endl;
}

void appFace::setEyesParameters(vector<Rect> __eyes, Rect faces){
	setEyesParameters(__eyes);
	/*
	this->eyeNumber = __eyes.size();
	for(int i=0;i<__eyes.size();i++){
		this->eyeDetectedRects[i] = __eyes[i];
	}
	*/
}

void appFace::setEyesParameters(vector<Rect> __eyes, Rect faces,Mat frame){
}

vector<Rect> appFace::detectEyes(Mat _face,Mat frame){//在脸中检测眼睛
		//开始检测眼睛，
	Mat faceROI = _face;//frame_gray( faceDetectedRect );
	std::vector<Rect> eyes;
	int minSize = faceROI.rows / 5;
	eyes_cascade.detectMultiScale( faceROI, eyes, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));
		
	//-- Draw eyes
	for( int j = 0; j < eyes.size(); j++ )
	{

		Rect _rect_el = Rect(faces[0].x + eyes[j].x, faces[0].y + eyes[j].y, eyes[j].width, eyes[j].height);
		rectangle(frame, _rect_el, Scalar(123,123,255));
		Point center( faces[0].x + eyes[j].x + eyes[j].width*0.5, faces[0].y + eyes[j].y + eyes[j].height*0.5 );
		int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
		circle( frame, center, 8, Scalar( 255, 0, 0 ), 4, 8, 0 );
		circle( frame, center, 8, Scalar( 255, 0, 0 ), 4, 8, 0 );

	}

	if(eyes.size()>0)
		return eyes;
	else
		cout << " 没有检测到眼睛。 " << endl;
	return eyes;
}

vector<Rect> appFace::detectMouth(Mat faceROI){//在脸中检测嘴
		//在脸里检测嘴
		//Mat faceROI = _face;//frame_gray( faceDetectedRect );
		std::vector<Rect> mouths;
		int minSize1 = faceROI.rows*1/6;
		mouth_cascade.detectMultiScale( faceROI, mouths, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize1*3, minSize1));

		return mouths;
}

vector<Rect> appFace::detectNose(Mat _face){//在脸中检测鼻子

		//在脸里检测鼻子
		Mat faceROI = _face;//frame_gray( faceDetectedRect );
		std::vector<Rect> noses;
		//这个参数值影响巨大
		int minSize2 = faceROI.rows / 8;
		nose_cascade.detectMultiScale( faceROI, noses, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize2, minSize2));
		return noses;
}

int appFace::leftFace(Mat img){//判断图中的脸向哪个方向偏。1向左，2向右，0没有检测到脸，-1只有一只眼
	// 判断向哪侧偏
	Mat frame_gray;
	vector<Rect> _faces;
	cvtColor( img, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	face_cascade.detectMultiScale( frame_gray, _faces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	Rect maskFaceRect;

	if(_faces.size()>0){
		for(int i_face = 0;i_face<_faces.size();i_face++){
			Mat _faceROI;
			_faceROI = frame_gray(_faces[i_face]);
			std::vector<Rect> _eyes = detectEyes(_faceROI);

			//-- Get eyes' center and radius
			if (_eyes.size() == 2){  //如果检测到了2只眼睛
				//先判断左右眼
				int leftEyeMiddleX = _faces[i_face].x + _eyes[0].x + (float)(_eyes[0].width)/2;
				int rightEyeMiddleX = _faces[i_face].x + _eyes[1].x + (float)(_eyes[1].width)/2;
				int leftN,rightN;					
				if(leftEyeMiddleX<rightEyeMiddleX){
					leftN = 0;rightN=1;
				} else {
					leftN = 1;rightN=0;
				}
				int leftEyeMiddleY = _faces[i_face].y + _eyes[leftN].y + (float)(_eyes[leftN].height)/2;
				int rightEyeMiddleY = _faces[i_face].y + _eyes[rightN].y + (float)(_eyes[rightN].height)/2;
				if(leftEyeMiddleY > rightEyeMiddleY)
					return 1;
				else
					return 2;
			}else{
				return -1;
			}
		}
	}else{
		return 0;
	}
}


int appFace::rotateDetectFaces() { //旋转检测脸。如果没有返回0。如果有，设置脸数。
	Mat frame;
	imageOrigine.copyTo(frame);
	std::vector<Rect> _faces;
    double angle = 1;  // 每步旋转角度  
	double _angle,_lastTestAngle;
	int _eyes_y_dif = 0;//两只眼睛的Y坐标差
	int _faceDirectTemp = leftFace(frame); // 判断脸向哪个方向偏。


	int direct = 2;//脸的旋转方向，默认为向右转。
	if(_faceDirectTemp > 0 )
		direct = _faceDirectTemp ;
	else
		return -1; // TODO:  没有处理一只眼的情况
	int angleDegree = (360)/angle;
	for(int i_angle=0;i_angle<angleDegree;i_angle++){
		Mat rotateImg;  
		imwrite("MedianData//simpleFaceDetectionb.png",frame);
		
		//if(direct == 2)
			rotateImg = rotate(frame,i_angle*angle);
		//if(direct == 1)
		//	rotateImg = rotate(frame,(360-i_angle*angle));
		
		Mat frame_gray;
		cvtColor( rotateImg, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );

		//-- Detect faces
		cout << " 旋转后重新检测脸一次。。。" << endl;
		face_cascade.detectMultiScale( frame_gray, _faces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		Rect maskFaceRect;
		if(_faces.size() <1){
			cout << " 没有检测到脸。 " << i_angle * angle << endl;
			imwrite("MedianData//simpleFaceDetectionc.png",frame_gray);
			/*
			//用抠图。在下面实现。
			if(imageFaceContourSM.cols){ // 已经做了脸的抠图
				Mat roatedimageFaceContourSM = rotate(imageFaceContourSM,i_angle * angle);
				maskFaceRect = Rect(
					this->calFirstColOfContour(roatedimageFaceContourSM),
					this->calFirstRowOfContour(roatedimageFaceContourSM),
					this->calLastColOfContour(roatedimageFaceContourSM)-calFirstColOfContour(roatedimageFaceContourSM),
					this->calLastRowOfContour(roatedimageFaceContourSM)-calFirstRowOfContour(roatedimageFaceContourSM));

				_faces.push_back(maskFaceRect);// = vector<Rect>(maskFaceRect));
			}
			*/
		}

		for(int i_face = 0;i_face<_faces.size();i_face++){
			Mat _faceROI;
			_faceROI = frame_gray(_faces[i_face]);
			std::vector<Rect> _eyes = detectEyes(_faceROI);
			//-- Get eyes' center and radius
			if (_eyes.size() == 2){  //如果检测到了2只眼睛
				//先判断左右眼
				int leftEyeMiddleX = _faces[i_face].x + _eyes[0].x + (float)(_eyes[0].width)/2;
				int rightEyeMiddleX = _faces[i_face].x + _eyes[1].x + (float)(_eyes[1].width)/2;
				int leftN,rightN;					
				if(leftEyeMiddleX<rightEyeMiddleX){
					leftN = 0;rightN=1;
				} else {
					leftN = 1;rightN=0;
				}

				int leftEyeMiddleY = _faces[i_face].y + _eyes[leftN].y + (float)(_eyes[leftN].height)/2;
				int rightEyeMiddleY = _faces[i_face].y + _eyes[rightN].y + (float)(_eyes[rightN].height)/2;
				if(leftEyeMiddleY > rightEyeMiddleY && angle > 0 ) angle = -1 * angle;
				//如果两只眼睛是平的：中心点Y坐标相等，或再旋转一次，过头了，如果比上度次的小，就用这次的。
				int _eyes_y_dif1 = leftEyeMiddleY - rightEyeMiddleY;
					
				cout<< " 检测到2只眼睛  中心点Y坐标不相等 " << leftEyeMiddleY << " " << rightEyeMiddleY 
					<< ", 坐标差： " << _eyes_y_dif1 << ", 最小坐标差：" << _eyes_y_dif 
					<< ", 眼0高度/10:" << _eyes[0].height/10 << ",  眼1高度/10:" << _eyes[1].height/10 << endl;
				string filename = "MedianData//simpleFaceDetection1";// ".png";
				std::ostringstream oss;
				oss << filename << i_face << ".png";
				imwrite(oss.str(),rotateImg);

				if(_eyes_y_dif1 != 0){  //中心点Y坐标不相等
					if(_eyes_y_dif == 0)
						_eyes_y_dif = _eyes_y_dif1;
					else {
							cout << " 进入 " << endl;
						if(abs(_eyes_y_dif) > abs(_eyes_y_dif1)) // 如果旋转后，距离小了
							_eyes_y_dif = _eyes_y_dif1; //记录小的距离
						else if(abs(_eyes_y_dif1)<_eyes[0].height/10 && abs(_eyes_y_dif1)<_eyes[1].height/10)
						{ //否则就是大了或相等，向回旋转
							rotateAngle = (i_angle-1)*angle; // 记录最佳旋转角度
							//设置脸相关全局参数
							this->faces = _faces;

							cout<< "检测到2只眼睛  中心点Y坐标:" << leftEyeMiddleY << " " << rightEyeMiddleY << " " << _eyes_y_dif1 << endl;
							return _faces.size();
						}
					}
				} else {
					rotateAngle = (i_angle-1)*angle; // 记录最佳旋转角度
					//设置脸相关全局参数
					this->faces = _faces;

					cout<< "检测到2只眼睛  中心点Y坐标:" << leftEyeMiddleY << " " << rightEyeMiddleY << " " << _eyes_y_dif1 << endl;
					return _faces.size();
				}
			} else if(_eyes.size() == 1){  //如果检测到了1只眼睛，可能是侧脸。待处理。
				//如果只检测到一只眼睛，要根据眼睛中心离脸边缘最大处，为双眼平行角度。离哪侧近，就是哪侧眼睛。根据中线，对称过程，构成另一只眼睛。

				cout<< "检测到一只眼睛" <<  endl;
				//break;

			} else {  //如果没有检测到眼睛
				cout<< "没有检测到眼睛" << endl;
				//break;
			}
		}
	}

	if(_faces.size() <1){
		cout << " 没有检测到脸。" << endl;
		//用抠图。

	}

	//设置脸相关全局参数
	this->faces = _faces;

	return 0;
}

Mat appFace::rotate(Mat srcm,double angle){

	IplImage* src =  &IplImage(srcm);
	
	int angle1 = angle;
	angle = abs((int)angle % 180); 
	if (angle > 90) { 
		angle = 90 - ((int)angle % 90); 
	} 
	

	if(angle<0) angle = 360+angle;

	IplImage* dst = NULL; 
	int width = 
	(double)(src->height * sin(angle * CV_PI / 180.0)) + 
	(double)(src->width * cos(angle * CV_PI / 180.0 )) + 1; 
	int height = 
	(double)(src->height * cos(angle * CV_PI / 180.0)) + 
	(double)(src->width * sin(angle * CV_PI / 180.0 )) + 1; 
	int tempLength = sqrt((double)src->width * src->width + src->height * src->height) + 10; 
	int tempX = (tempLength + 1) / 2 - src->width / 2; 
	int tempY = (tempLength + 1) / 2 - src->height / 2; 
	int flag = -1; 

	angle = angle1;

	dst = cvCreateImage(cvSize(width, height), src->depth, src->nChannels); 
	cvZero(dst); 
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels); 
	cvZero(temp); 

	cvSetImageROI(temp, cvRect(tempX, tempY, src->width, src->height)); 
	cvCopy(src, temp, NULL); 
	cvResetImageROI(temp); 


	float m[6]; 
	int w = temp->width; 
	int h = temp->height; 
	m[0] = (float) cos(flag * angle * CV_PI / 180.); 
	m[1] = (float) sin(flag * angle * CV_PI / 180.); 
	m[3] = -m[1]; 
	m[4] = m[0]; 
	// 将旋转中心移至图像中间 
	m[2] = w * 0.5f; 
	m[5] = h * 0.5f; 
	// 
	CvMat M = cvMat(2, 3, CV_32F, m); 
	cvGetQuadrangleSubPix(temp, dst, &M); 
	cvReleaseImage(&temp); 
	Mat ret = Mat(dst);
	return ret; 
}

void appFace::rotate(){
	imageOrigine = rotate(imageOrigine,this->rotateAngle);
	//imwrite("MedianData//imageOrigine1.png",imageOrigine);
	imageContourSM = rotate(imageContourSM,this->rotateAngle);
	//imwrite("MedianData//imageContourSM.png",imageContourSM);
	imageFaceContourSM = rotate(imageFaceContourSM,this->rotateAngle);
	//imwrite("MedianData//imageFaceContourSM.png",imageFaceContourSM);
//	imageHairContourSM = rotate(imageHairContourSM,this->rotateAngle); 
//	imageRealHairContourSM = rotate(imageRealHairContourSM,this->rotateAngle);
//	if(imageRealFaceContourSM.cols)
	imageRealFaceContourSM = rotate(imageRealFaceContourSM,this->rotateAngle);
	//imwrite("MedianData//imageRealFaceContourSM.png",imageRealFaceContourSM);
	imageRealContourSM = rotate(imageRealContourSM,this->rotateAngle);
	//imwrite("MedianData//imageRealContourSM.png",imageRealContourSM);

	cvtColor(imageOrigine, this->YCrCbImage, CV_BGR2YCrCb);
	firstRowOfContourHuman = calFirstRowOfContourHuman();
	if (firstRowOfContourHuman < 0)
		cout << "error firstRowOfContourHuman" << endl;

}

void appFace::setFaceParameters(Mat frame){
	int i = 0;//设置第一张脸

		//-- Amply face rectangle
	int rowBegin = firstRowOfContourHuman;
	int colBegin = faces[i].x;
	if (colBegin - 10 > 0)
		colBegin = colBegin - 10;
	int rowEnd = faces[i].y + faces[i].height;
	if (rowEnd + 5 < frame.rows )
		rowEnd = rowEnd + 5;
	int colEnd = faces[i].x + faces[i].width;
	if (colEnd + 5 < frame.cols)
		colEnd = colEnd + 5;

	int e = calLastRowOfContour(this->imageFaceContourSM);
	if(e>0)
		rowEnd = e;
//		cout <<  e << endl;

	this->faceChangeRect = Rect(colBegin, rowBegin, colEnd - colBegin, rowEnd - rowBegin);

	//-- Draw rectangles
	//rectangle(frame, faces[i], Scalar(255,0,0));
	//rectangle(frame, this->faceChangeRect, Scalar(0,255,0));
		
	Rect _rect_f = Rect(faces[i].x, faces[i].y, faces[i].width/2, faces[i].height);
	faceDetectedRect = faces[i];
	//rectangle(frame, this->faceDetectedRect, Scalar(0,0,255));
	//rectangle(frame, _rect_f, Scalar(0,0,255));

}

//
void appFace::simpleFaceDetection1(){
	Mat frame;
	imageOrigine.copyTo(frame);

	//std::vector<Rect> faces; //做为全局变量了.
	Mat frame_gray;
	
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- 旋转并检测脸，取出双眼平行的图像。
	int faceCount  = rotateDetectFaces();
	cout << faceCount << endl;
	if(faceCount > 0 ){

		rotate();//旋转所有图像
		frame = rotate(frame,this->rotateAngle); // 旋转frame，准备在上面标记五官。
		frame.copyTo(debugFrame);//初始化debugFrame.

		setFaceParameters(frame);//设置脸的参数
		debugFace();

		cvtColor( frame, frame_gray, CV_BGR2GRAY );//因为旋转了，所以需要重新处理一次源图片。
		equalizeHist( frame_gray, frame_gray );
		Mat faceROI = frame_gray( faceDetectedRect );

		for( size_t i = 0; i < faces.size(); i++ )
		{
			//-- In each face, detect eyes
			std::vector<Rect> __eyes;
			__eyes = detectEyes(faceROI);//检测眼睛
			if(__eyes.size()>0){
				setEyesParameters(__eyes,faces[i]);
				debugEyes();
			}
			else{
				cout << " 没有检测到眼睛，退出程序。" << endl;
				return;
			}
			
			//在脸里检测嘴
			//Mat faceROI = frame_gray( faceDetectedRect );
			Rect mouthRegion = Rect(
				eyeDetectedRects[0].x,
				eyeDetectedRects[0].y+eyeDetectedRects[0].height,
				eyeDetectedRects[1].x + eyeDetectedRects[1].height - eyeDetectedRects[0].x,
				faceDetectedRect.y+faceDetectedRect.height - (eyeDetectedRects[0].y+eyeDetectedRects[0].height)
				);
			Mat mouthROI = frame_gray(mouthRegion);
			std::vector<Rect> mouths;
			mouths = this->detectMouth(mouthROI);
			if(mouths.size()>0){
				setMouthsParameter(mouths,mouthRegion);
				debugMouth();
			}else{
				cout << " 没有检测到嘴。 " << endl;
			}

			//在脸里检测鼻子
			Rect nr = Rect(
				eyeDetectedRects[0].x+eyeDetectedRects[0].width/2, // 左眼中心X
				eyeDetectedRects[0].y+eyeDetectedRects[0].height/2, //左眼中心Y
				eyeDetectedRects[1].x+eyeDetectedRects[1].width/2 -(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2), // 右眼中心到左眼中心的宽度
				mouthDetectedRect.y+mouthDetectedRect.height/2-(eyeDetectedRects[0].y+eyeDetectedRects[0].height/2)); // 嘴中心到左眼中心的高度
			Mat noseROI = frame_gray( nr );
			rectangle(debugFrame, nr, Scalar(0,0,0));
			std::vector<Rect> noses;
			noses = this->detectNose(noseROI);

			if(noses.size() > 0 ){
				setNoseParameter(noses,nr);
				Mat roi = frame(noseDetectedRect);
				createROI(roi,"nose");

				debugNose();
			} else {
				cout << " 没有检测到鼻子。 " << endl;
			}


			//确定脸的中线faceMiddleRect。如果存在鼻子，就按鼻子；否则如果存在嘴，就按嘴；否则，就按脸。
			if(noseDetectedRect.x > 0)
				faceMiddleRect = Rect(noseDetectedRect.x + noseDetectedRect.width/2, faces[i].y, 2, faces[i].height);
			else if(mouthDetectedRect.x>0)
				faceMiddleRect = Rect(mouthDetectedRect.x+mouthDetectedRect.width/2, faces[i].y, 2, faces[i].height);
			else
				faceMiddleRect = Rect(faceDetectedRect.x+faceDetectedRect.width/2, faces[i].y, 2, faces[i].height);

		}
	}
	else{
		cout << " 没有检测到脸，什么也不做。" << endl;
		return;
	}
	imwrite("MedianData//simpleFaceDetection.png",frame);
}

//
void appFace::simpleFaceDetection(){
	Mat frame;
	imageOrigine.copyTo(frame);

	//std::vector<Rect> faces; //做为全局变量了.
	Mat frame_gray;
	
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	if(faces.size() <1)
		cout << " 没有检测到脸。" << endl;
	for( size_t i = 0; i < faces.size(); i++ )
	{
		//-- Amply face rectangle
		int rowBegin = firstRowOfContourHuman;
		int colBegin = faces[i].x;
		if (colBegin - 10 > 0)
			colBegin = colBegin - 10;
		int rowEnd = faces[i].y + faces[i].height;
		if (rowEnd + 5 < frame_gray.rows )
			rowEnd = rowEnd + 5;
		int colEnd = faces[i].x + faces[i].width;
		if (colEnd + 5 < frame_gray.cols)
			colEnd = colEnd + 5;

		int e = calLastRowOfContour(this->imageFaceContourSM);
		if(e>0)
			rowEnd = e;
//		cout <<  e << endl;

		this->faceChangeRect = Rect(colBegin, rowBegin, colEnd - colBegin, rowEnd - rowBegin);

		//-- Draw rectangles
		rectangle(frame, faces[i], Scalar(255,0,0));
		rectangle(frame, this->faceChangeRect, Scalar(0,255,0));
		
		Rect _rect_f = Rect(faces[i].x, faces[i].y, faces[i].width/2, faces[i].height);
		faceDetectedRect = faces[i];
		rectangle(frame, this->faceDetectedRect, Scalar(0,0,255));
		rectangle(frame, _rect_f, Scalar(0,0,255));


		//-- In each face, detect eyes
		Mat faceROI = frame_gray( faceDetectedRect );
		std::vector<Rect> eyes;
		int minSize = faceROI.rows / 5;
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));
		
		//-- Draw eyes
		for( size_t j = 0; j < eyes.size(); j++ )
		{
			if (eyes[j].y > faceDetectedRect.height*0.5)
			{
				if (j == eyes.size()-1)
				{
					eyes.pop_back();
					continue;
				}
				vector<Rect>::iterator itr = eyes.begin();
				while (itr != eyes.end())
				{

					if (*itr == eyes[j]) eyes.erase(itr);//删除值为3的元素
					++itr;
				}
			}

			Rect _rect_el = Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);
			rectangle(frame, _rect_el, Scalar(123,123,255));
			Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
			int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
			circle( frame, center, 8, Scalar( 255, 0, 0 ), 4, 8, 0 );
			circle( frame, center, 8, Scalar( 255, 0, 0 ), 4, 8, 0 );
		}
		this->eyeNumber = eyes.size();
		//-- Get eyes' center and radius
		if (eyes.size() == 2){
			eyeDetectedRects[0] = Rect(faces[i].x + eyes[0].x, faces[i].y + eyes[0].y, eyes[0].width, eyes[0].height);
			eyeDetectedRects[1] = Rect(faces[i].x + eyes[1].x, faces[i].y + eyes[1].y, eyes[1].width, eyes[1].height);
			Point2i c1(eyeDetectedRects[0].x + eyeDetectedRects[0].width*0.5, eyeDetectedRects[0].y + eyeDetectedRects[0].height*0.5);
			Point2i c2(eyeDetectedRects[1].x + eyeDetectedRects[1].width*0.5, eyeDetectedRects[1].y + eyeDetectedRects[1].height*0.5);
			eyeCircles[0].c = c1;
			eyeCircles[1].c = c2;//(eyeRects[0].width + eyeRects[0].height)*0.25
			eyeCircles[0].radius = (eyeDetectedRects[0].width +eyeDetectedRects[0].height)*0.125
				+(eyeDetectedRects[1].width +eyeDetectedRects[1].height)*0.125;
			eyeCircles[1].radius = (eyeDetectedRects[0].width +eyeDetectedRects[0].height)*0.125
				+(eyeDetectedRects[1].width +eyeDetectedRects[1].height)*0.125;
		}
		else
			cout << "eyes detection error! eyes' number : " << eyes.size() << endl;

		//在脸里检测嘴
		//Mat faceROI = frame_gray( faceDetectedRect );
		std::vector<Rect> mouths;
		int minSize1 = faceROI.rows / 6;
		mouth_cascade.detectMultiScale( faceROI, mouths, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize1*2, minSize1));

		int x1 = int(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2); //middlePointX(eyeDetectedRects[0]);
		int y1 = int(eyeDetectedRects[0].y+eyeDetectedRects[0].height/2);//middlePointY(eyeDetectedRects[0]);
		int x2 = int(eyeDetectedRects[1].x+eyeDetectedRects[1].width/2);//middlePointX(eyeDetectedRects[1]);
		int y2 = int(eyeDetectedRects[1].y+eyeDetectedRects[1].height/2);//middlePointY(eyeDetectedRects[1]);

		int _y = 0;int _ym=-1;
		for( size_t mi = 0; mi < mouths.size(); mi++ )
		{
			//如果嘴不与其它部分重合，且在两眼之间，就算是嘴。
			int mx = int(faces[i].x+mouths[mi].x+mouths[mi].width/2);
			int my = int(faces[i].y+mouths[mi].y+mouths[mi].height/2);
			//cout << mx << ": " << x1 << ": " << x2<< "|| my :"<<my<<"  "<<y1 << "  "<< y2 << endl;
			bool _betwin = false;
			if((mx>=x1 && mx<=x2) || mx<= x1 && mx >=x2) _betwin = true;
			if(_betwin && my>y1 && my>y2) 
			{
				//最下面的那一个，是嘴。
				if(mouths[mi].y>_y){
					_y = mouths[mi].y;
					_ym = mi;
				}
			}
		}
		//找到了一个嘴
		if(_ym>=0){
			this->mouthDetectedRect = Rect(this->faceDetectedRect.x+mouths[_ym].x,this->faceDetectedRect.y+mouths[_ym].y,mouths[_ym].width,mouths[_ym].height);
			//Rect _rect_ml = Rect(faces[i].x + mouthDetectedRect.x, faces[i].y + mouthDetectedRect.y, mouthDetectedRect.width, mouthDetectedRect.height);
			rectangle(frame, mouthDetectedRect, Scalar(100,100,100));
			Rect _rect_mlm = Rect(mouthDetectedRect.x+mouthDetectedRect.width/2, faces[i].y, 2, faces[i].height);
			rectangle(frame, _rect_mlm, Scalar(100,100,100));
		}


				//在脸里检测鼻子
		//Mat faceROI = frame_gray( faceDetectedRect );
		std::vector<Rect> noses;
		//这个参数值影响巨大
		int minSize2 = faceROI.rows / 4;
		nose_cascade.detectMultiScale( faceROI, noses, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize2, minSize2));

		int x3 = int(mouthDetectedRect.x+mouthDetectedRect.width/2);//middlePointX(mouthDetectedRect);
		int y3 = int(mouthDetectedRect.y+mouthDetectedRect.height/2);//middlePointY(mouthDetectedRect);

		_y = 0; int _yn = -1;
		for( size_t ni = 0; ni < noses.size(); ni++ )
		{
			int nx = int(faces[i].x+noses[ni].x+noses[ni].width/2);
			int ny = int(faces[i].y+noses[ni].y+noses[ni].height/2);
			//cout << mx << ": " << x1 << ": " << x2<< "|| my :"<<my<<"  "<<y1 << "  "<< y2 << endl;
			bool _betwin = false;
			if((nx>=x1 && nx<=x2) || nx<= x1 && nx >=x2) _betwin = true;

			if(_betwin && ny>y1 && ny>y2 && ny<y3) 
			{
				//在嘴上面，在眼睛下面，在两眼之间，就算是鼻子。

				//取最下面的那个
				if(noses[ni].y>_y){
					_y = noses[ni].y;
					_yn = ni;
				}
			}
		} 
		if(_yn>=0){
			this->noseDetectedRect = Rect(this->faceDetectedRect.x+noses[_yn].x,this->faceDetectedRect.y+noses[_yn].y,noses[_yn].width,noses[_yn].height);
			rectangle(frame, noseDetectedRect, Scalar(0,0,0));
			Rect _rect_nlm = Rect(noseDetectedRect.x + noseDetectedRect.width/2, faces[i].y, 2, faces[i].height);
			rectangle(frame, _rect_nlm, Scalar(0,0,0));
		}


		//确定脸的中线faceMiddleRect。如果存在鼻子，就按鼻子；否则如果存在嘴，就按嘴；否则，就按脸。
		if(_yn>0)
			faceMiddleRect = Rect(noseDetectedRect.x + noseDetectedRect.width/2, faces[i].y, 2, faces[i].height);
		else if(_ym>0)
			faceMiddleRect = Rect(mouthDetectedRect.x+mouthDetectedRect.width/2, faces[i].y, 2, faces[i].height);
		else
			faceMiddleRect = Rect(faceDetectedRect.x+faceDetectedRect.width/2, faces[i].y, 2, faces[i].height);

	}

	imwrite("MedianData//simpleFaceDetection.png",frame);
}



void appFace::colorBasedFaceDetection(){
	maskFace = Mat(imageOrigine.rows, imageOrigine.cols, CV_8UC1, Scalar::all(0));
	Mat frame;
	imageOrigine.copyTo(frame);

	uchar *mask_face_data = maskFace.ptr<uchar>(0);
	for (int _row = faceChangeRect.y; _row < faceChangeRect.y + faceChangeRect.height ; _row++)
	{
		for (int _col = faceChangeRect.x ; _col < faceChangeRect.x + faceChangeRect.width ; _col++)
		{
			int index = _row*frame.cols + _col;
			//-- Face detection in level of pixel
			//-- Use color space of YCbCr
			if ( (approximateFaceColor(_row, _col)) || aroundEye(_row, _col))
				mask_face_data[index] = 255;
		}
	}

	medianBlur(maskFace,maskFace, 5);
	//-- Complete components inside face, like nose, mouth etc.
	insideComponent(maskFace);
	insideComponent(maskFace);

	faceChangeRect = calFaceChangeRect(maskFace);

	imwrite("MedianData//colorBasedFaceDetection.png",maskFace);
}

void appFace::FaceDetection(){
	simpleFaceDetection1();
	//simpleFaceDetection();
	colorBasedFaceDetection();
	;
}

void appFace::insideComponent(Mat &maskImage, int min_dif)
{
	Mat label_image;
	maskImage.convertTo(label_image, CV_32SC1);

	int label_count = 254;

	//-- make inside components (nose, mouth) become white (face)
	for(int y=0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		Rect rect;
		for(int x=0; x < label_image.cols; x++) {
			if(row[x] != 0) {
				continue;
			}
			floodFill(label_image, cv::Point(x,y), label_count, &rect, min_dif, min_dif, 4);
			if (rect.area() < faceChangeRect.area())
				break;

			label_count--;
		}

		for(int i=rect.y; i < (rect.y+rect.height); i++) {
			int *row2 = (int*)label_image.ptr(i);
			for(int j=rect.x; j < (rect.x+rect.width); j++) {
				if(row2[j] != label_count) {
					continue;
				}
				maskImage.at<uchar>(i,j) = 255;
			}
		}
	}

	//-- 
	label_count = 256;
	for(int y=0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		Rect rect;
		for(int x=0; x < label_image.cols; x++) {
			if(row[x] != 255) {
				continue;
			}
			floodFill(label_image, cv::Point(x,y), label_count, &rect, min_dif, min_dif, 4);
			if (rect.area() < 0.3*faceChangeRect.area())
				break;

			label_count++;
		}
		for(int i=rect.y; i < (rect.y+rect.height); i++) {
			int *row2 = (int*)label_image.ptr(i);
			for(int j=rect.x; j < (rect.x+rect.width); j++) {
				if(row2[j] != label_count) {
					continue;
				}
				maskImage.at<uchar>(i,j) = 0;
			}
		}
	}
}

void appFace::insideComponent2(Mat &maskImage, int min_dif)
{
	Mat label_image;
	maskImage.convertTo(label_image, CV_32SC1);

	int label_count = 256;

	//-- make inside components (nose, mouth) become white (face)
	for(int y=0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		Rect rect;
		for(int x=0; x < label_image.cols; x++) {
			if(row[x] > 64 && row[x] < 256) {
				continue;
			}
			floodFill(label_image, cv::Point(x,y), label_count, &rect, min_dif, min_dif, 4);
			if (rect.area() < faceChangeRect.area())
				break;

			label_count--;
		}

		for(int i=rect.y; i < (rect.y+rect.height); i++) {
			int *row2 = (int*)label_image.ptr(i);
			for(int j=rect.x; j < (rect.x+rect.width); j++) {
				if(row2[j] != label_count) {
					continue;
				}
				maskImage.at<uchar>(i,j) = 255;
			}
		}
	}
}

Rect appFace::calFaceChangeRect(Mat _maskFace){
	int leftCol = -1, rightCol = -1;
	uchar* maskColData = _maskFace.ptr<uchar>(0);
	for (int _col = faceChangeRect.x ; _col < faceChangeRect.x + faceChangeRect.width ;_col++ )
	{
		for (int _row = faceChangeRect.y ; _row < faceChangeRect.y + faceChangeRect.height ;_row++)
		{
			int index = _row*_maskFace.cols + _col;
			if (maskColData[index] > 0)
			{
				leftCol = _col - 5;
				break;
			}
		}
		if (leftCol >= 0)
			break;
	}

	for (int _col = faceChangeRect.x + faceChangeRect.width - 1; _col >= faceChangeRect.x ;_col-- )
	{
		for (int _row = faceChangeRect.y ; _row < faceChangeRect.y + faceChangeRect.height ;_row++)
		{
			int index = _row*_maskFace.cols + _col;
			if (maskColData[index] > 0)
			{
				rightCol = _col + 5;
				break;
			}
		}
		if (rightCol >= 0)
			break;
	}


	Rect rect(leftCol, faceChangeRect.y, rightCol-leftCol, faceChangeRect.height);
	return rect;
}

void appFace::FaceChange(Mat faceModel, Mat leftEyeModel, Mat rightEyeModel,Mat leftEyeWithBrowModel,Mat rightEyeWithBrowModel,Mat leftEyePupilModel,Mat rightEyePupilModel,Mat mouthModel,Mat noseModel,int mode){
	Mat resultImage;
	imageOrigine.copyTo(resultImage);
	//insideComponent2(imageRealContourSM, 30);
	filterMirrorBackground(resultImage);
	hsvCompromise(faceModel);
	replaceFaceAndEyes(faceModel, leftEyeModel, rightEyeModel,leftEyeWithBrowModel,rightEyeWithBrowModel, resultImage,leftEyePupilModel,rightEyePupilModel,mouthModel,noseModel,mode);
	//saveImages(resultImage);
}
void appFace::FaceChange(int isHair){
	Mat resultImage;
	imageOrigine.copyTo(resultImage);
	//insideComponent2(imageRealContourSM, 30);
	filterMirrorBackground(resultImage);
	Mat faceModel = chm.currentHead.faceModel;
	hsvCompromise(faceModel);
	replaceFaceAndEyes( resultImage,isHair);
	//saveImages(resultImage);
}

void appFace::filterMirrorBackground(Mat &resultImage){
	//------------------------
	//Add filter mirror here, this->imageContourSM is the contour of human, resultImage is the original image
	//滤镜加在这
	this->imageContourSM;

	//------------------------
}

void appFace::brightnessContrast(Mat &resultImage, double alpha, double beta){
	//alpha brightness, beta contrast
	double k = tan( (45 + 44 * beta) / 180 * M_PI );
	Vec3b* result_data = resultImage.ptr<Vec3b>(0);
	int totalResult = resultImage.rows * resultImage.cols;
	for (int i = 0 ; i < totalResult; i++)
	{
		for (int _c = 0 ; _c < 3; _c++ )
		{
			int x = result_data[i][_c];
			result_data[i][_c] = saturate_cast<uchar>((x - 127.5 * (1 - alpha)) * k + 127.5 * (1 + alpha));
		}
	}
	
}

void appFace::hsvCompromise(Mat &faceModel){
	Mat bgrImage,hsvImage;
	cvtColor(imageOrigine, bgrImage, CV_BGRA2BGR);
	cvtColor(bgrImage, hsvImage, CV_BGR2HSV);

	//-- Get face's color mean But won't use it any more.
	Mat faceColorROI = hsvImage( faceDetectedRect );
	Mat mask_face( faceColorROI.rows, faceColorROI.cols, CV_8UC1, Scalar::all(0) );
	for ( int _row = faceColorROI.rows * 0.5 ; _row < faceColorROI.rows * 0.7 ; _row++ )
	{
		uchar* mask_face_row_data = mask_face.ptr<uchar>(_row);
		for ( int _col = faceColorROI.cols * 0.2 ; _col < faceColorROI.cols * 0.3 ; _col++)
		{
			mask_face_row_data[_col] = 255;
		}
		for ( int _col = faceColorROI.cols * 0.7 ; _col < faceColorROI.cols * 0.8 ; _col++)
		{
			mask_face_row_data[_col] = 255;
		}
	}
	faceMeanColorHSV = mean(faceColorROI,mask_face);

	Mat bgrFaceModel, hsvFaceModel;
	cvtColor(faceModel, bgrFaceModel, CV_BGRA2BGR);
	cvtColor(bgrFaceModel, hsvFaceModel, CV_BGR2HSV);

	Scalar modelMeanColorHSV = mean(hsvFaceModel);
	int totalModel = bgrFaceModel.rows * bgrFaceModel.cols;
	Vec4b *face_model_data = faceModel.ptr<Vec4b>(0);
	Vec3b* hsv_model_data = hsvFaceModel.ptr<Vec3b>(0);
	for (int i = 0; i < totalModel; i++)
	{
		//if (bgr_model_data[i][0] > 250 && bgr_model_data[i][1] > 250 && bgr_model_data[i][2] > 250)
		if (face_model_data[i][3] == 0)
			continue;
		for (int _c = 0; _c < 3; _c++)
		{
			hsv_model_data[i][_c] = 0.6*hsv_model_data[i][_c] + 0.4*faceMeanColorHSV[_c];
			//hsv_model_data[i][_c] = 0.6*hsv_model_data[i][_c] + faceMeanColorHSV[_c];
		}
	}
	cvtColor(hsvFaceModel, bgrFaceModel, CV_HSV2BGR);
	Vec3b* bgr_model_data = bgrFaceModel.ptr<Vec3b>(0);
	for (int i = 0; i < totalModel; i++)
	{
		face_model_data[i] = Vec4b(bgr_model_data[i][0],bgr_model_data[i][1],bgr_model_data[i][2],face_model_data[i][3]);
	}
	return;
}

void appFace::bgrCompromise(Mat &faceModel){
	Mat frame;
	cvtColor(imageOrigine, frame, CV_BGR2HSV);

	//-- Get face's color mean But won't use it any more.
	Mat faceColorROI = frame( faceDetectedRect );
	Mat mask_face( faceColorROI.rows, faceColorROI.cols, CV_8UC1, Scalar::all(0) );
	for ( int _row = faceColorROI.rows * 0.5 ; _row < faceColorROI.rows * 0.7 ; _row++ )
	{
		uchar* mask_face_row_data = mask_face.ptr<uchar>(_row);
		for ( int _col = faceColorROI.cols * 0.2 ; _col < faceColorROI.cols * 0.3 ; _col++)
		{
			mask_face_row_data[_col] = 255;
		}
		for ( int _col = faceColorROI.cols * 0.7 ; _col < faceColorROI.cols * 0.8 ; _col++)
		{
			mask_face_row_data[_col] = 255;
		}
	}
	Scalar faceMeanColorBGR = mean(faceColorROI,mask_face);

	Scalar modelMeanColorBGR = mean(faceModel);
	int totalModel = faceModel.rows * faceModel.cols;
	Vec3b* bgr_model_data = faceModel.ptr<Vec3b>(0);
	for (int i = 0; i < totalModel; i++)
	{
		if (bgr_model_data[i][0] > 250 && bgr_model_data[i][1] > 250 && bgr_model_data[i][2] > 250)
			continue;

		for (int _c = 0; _c < 3; _c++)
		{
			bgr_model_data[i][_c] = bgr_model_data[i][_c]/2 + faceMeanColorBGR[_c]/2;
		}
	}

	return;
}
void appFace::initTempMat(){
	//Mat _bgraFrameLight;
	//Mat _bgraFrame;
	//Mat faceSample;//脸模型的临时文件
	//Mat frame;
	faceModel = chm.currentHead.faceModel;
	leftEyeModel = chm.currentExpression.leftEye.model;
	rightEyeModel = chm.currentExpression.rightEye.model;
	leftEyeWithBrowModel = chm.currentExpression.leftEyeBrow.model;
	rightEyeWithBrowModel = chm.currentExpression.rightEyeBrow.model;
	leftEyePupilModel = chm.currentExpression.leftPupilModel;
	rightEyePupilModel = chm.currentExpression.rightPupilModel;
	mouthModel = chm.currentExpression.mouth.model;
	noseModel = chm.currentHead.nose.model;
	fWidth = 0.0;
	fHeight = 0.0;
}
void appFace::initCounter(){
	if (imageFaceContourSM.cols)
	{
		imageFaceContourSM.copyTo(maskFace);
		insideComponent(maskFace);
		imwrite("MedianData//maskFace.png", maskFace);
	}

	if (imageHairContourSM.cols)
	{
		imageHairContourSM.copyTo(maskHair);
		imageRealHairContourSM.copyTo(maskRealHair);
		insideComponent(maskHair);
		//insideComponent(maskRealHair);
		imwrite("MedianData//maskHair.png", maskHair);
		imwrite("MedianData//maskReal.png",maskRealHair);
	}

	if (imageRealFaceContourSM.cols)
	{
		imageRealFaceContourSM.copyTo(maskRealFace);
	}

	maskFace.copyTo(maskFaceReplace);
	maskHair.copyTo(maskHairReplace);

}

void appFace::calFaceParameters(){}

//根据不同模式，处理脸模板。
void appFace::resizeFaceModel(int mode){
	Mat faceModel = chm.currentHead.faceModel;
	//如果按比例缩放后，模板脸比要换的区域小，就把模板拉宽到与替换区域等宽。
	//if(_fcols<faceChangeRect.width) _fcols=faceChangeRect.width;
	//TODO 这里是近似值。因为正确的应该是根据人脸中线判断脸大小。
	int faceWidth = -1;
	//判断左脸宽
	int leftFaceWidth = getLeftFaceWidth();
	//判断右脸宽
	int rightFaceWidth = getRightFaceWidth();
	cout << leftFaceWidth << "  " << rightFaceWidth << endl;
	//如果左脸宽>0，以左脸宽度计算
	if(leftFaceWidth > 0 && leftFaceWidth>rightFaceWidth) faceWidth = leftFaceWidth*2;
	//如果右脸宽>0，以右脸宽度计算
	if(rightFaceWidth > 0 && rightFaceWidth>leftFaceWidth) faceWidth = rightFaceWidth*2;
	//faceWidth = 0;
	//如果左右都没有，按上下调整的等比缩放

	//因为长头发时脸不够宽，所以调上下高度,宽度等比收缩
	if(faceWidth > 0){
		//怕脸边缘差异过大，先把脸加宽10%，这样会保持本人脸形。做完捏脸后，要去掉下面这行。
		faceWidth = faceWidth*1;
		fWidth = (double)  faceWidth / faceModel.cols;
		fHeight = (double) faceChangeRect.height / faceModel.rows;
		//如果计算出了脸的宽度，就修改faceChangeRect
		faceChangeRect = Rect(faceChangeRect.x+(faceChangeRect.width-faceWidth)/2,faceChangeRect.y,faceChangeRect.width,faceChangeRect.height);
	}
	else{ // TODO: 如果两侧挡脸，按黄金分割设置脸宽。
		faceWidth = faceModel.cols *  ((double)faceChangeRect.height / (double)faceModel.rows);
		fWidth = (double)  faceWidth / faceModel.cols;
		fHeight = (double) faceChangeRect.height / faceModel.rows;
		faceChangeRect = Rect(faceChangeRect.x+(faceChangeRect.width-faceWidth)/2,faceChangeRect.y,faceChangeRect.width,faceChangeRect.height);
	}

	//按小比值缩放，因为按大的可能会眼睛过界。
	double _fcols = 0.0;
	//如果faceWidth>0，
	if(faceWidth>0){
		if(fWidth < fHeight)
			_fcols = fWidth*faceModel.cols;
		else
			_fcols = fHeight*faceModel.cols;

		//resize(faceModel, faceSample, Size(faceChangeRect.width, faceChangeRect.height));
		resize(faceModel, faceSample, Size( _fcols,faceChangeRect.height));
		//cout << _fcols << "  " << faceChangeRect.height << endl;
	}
	faceSampleRect = Rect((faceMiddleRect.x-faceWidth/2),(faceChangeRect.y+faceChangeRect.height-faceSample.rows),(faceSample.cols),(faceSample.rows));
}

//在resultImage图中，按mode模式，将脸换成faceModel。
void appFace::replaceFace(Mat faceModel,Mat &resultImage,int mode){

	Mat frame;resultImage.copyTo(frame);
	//Mat faceSample;
	//-- Change face
	Mat faceSampleBGR;
	cvtColor(faceSample, faceSampleBGR, CV_BGRA2BGR);

	Mat _bgraFrame,_bgraFrameLight,_bgrFrameSkin,_skinMask;
	//这里需要处理
	//_bgraFrameLight = _bgraFrame.copyTo(
	cvtColor(frame, _bgraFrame, CV_BGR2BGRA);

	cvtColor(frame, _bgraFrameLight, CV_BGR2BGRA);
	frame.copyTo(_bgrFrameSkin);

	//cvtColor(frame, _skinMask, CV_BGR2BGRA);
	
	Vec4b *bgra_frame_data = _bgraFrame.ptr<Vec4b>(0);
	Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);
	
	uchar *maskData = maskFace.ptr<uchar>(0);
	uchar *maskHairData = maskHair.ptr<uchar>(0);
	uchar *maskRealHairData = maskRealHair.ptr<uchar>(0);

	//换脸上的显示数据
	uchar* mask_face_replace_data = maskFaceReplace.ptr<uchar>(0);
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);

	//因脸模板加宽了，不能从左到右匹配，而应该从中间对齐。需要将faceChangeRect向右移到中线。
	int middle = -1*(this->faceMiddleRect.x - faceChangeRect.x-faceChangeRect.width/2) +  faceSample.cols/2 - faceChangeRect.width/2;

	
	//亮度平衡处理
	IplImage* imageFace = &IplImage(faceSampleBGR);
	IplImage* imageBgr = &IplImage(_bgraFrameLight);
	IplImage* imageBgrSkin = &IplImage(_bgrFrameSkin);
	double gFace = get_avg_gray(imageFace);
	double gBgr = get_avg_gray(imageBgr);
	//为了调试捏脸，先注释掉。
	set_avg_gray(imageBgr,imageBgr,gFace*0.9);

	//肤色处理。
	CvSize imageSize = cvSize(imageBgrSkin->width, imageBgrSkin->height);
	IplImage *imageSkin = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
	
	cvSkinSegment(imageBgrSkin,imageSkin);
	//cvSkinYUV(imageBgrSkin,imageSkin);
	//cvSkinHSV(imageBgrSkin,imageSkin);
	Mat skinMat= Mat(imageSkin);

	imwrite("MedianData//skinTemp.png", skinMat);
	imwrite("MedianData//faceTemp.png", faceSampleBGR);
	//写调整亮度后的文件
	imwrite("MedianData//bgrLight.png", _bgraFrameLight);

	//捏脸瘦脸
	imwrite("MedianData//bgrLightBeforeChangeFaceNoTransparent.png", _bgraFrameLight);
	//changeFace(_bgraFrameLight,mask_face_replace_data,faceSample);
	imwrite("MedianData//bgrLightAfterChangeFaceNoTransparent.png", _bgraFrameLight);

	for (int _row = 0; _row < faceSample.rows ; _row++)
	{
		Vec3b *colData = faceSampleBGR.ptr<Vec3b>(_row);
		Vec4b *colDataBGRA = faceSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < faceSample.cols ; _col++){
			int r = faceChangeRect.y + _row; 
			//因脸模板加宽了，不能从左到右匹配，而应该从中间对齐。需要将faceChangeRect向右移到中线。
			int c = faceChangeRect.x - middle + _col;
			int index = r*frame.cols + c;
			//-- Get valid area of face model
			if (colDataBGRA[_col][3] == 0){
				mask_face_replace_data[index] = 0;
				continue;
			}
			
			if(mode >= 0)
			{
				if (mask_face_replace_data[index] == 255)
				{
					//变化明暗度
					Vec3b vf_hsv,vf_rgb,vb_hsv,vb_rgb,cartoon_vb_rgb,cartoon_vb_hsv;
					vf_rgb = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
					//vb_rgb = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
					vb_rgb = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
					//cartoon_vb_rgb = Vec3b(cartoonBgra_frame_data[index][0],cartoonBgra_frame_data[index][1],cartoonBgra_frame_data[index][2]);
					/*
					if(vb_rgb[0]<200) vb_rgb[0]=200;
					if(vb_rgb[1]<200) vb_rgb[1]=200;
					if(vb_rgb[2]<200) vb_rgb[2]=200;
					*/
					vf_hsv = kcvRGB2HSV(vf_rgb);
					vb_hsv = kcvRGB2HSV(vb_rgb);
					//cartoon_vb_hsv = kcvRGB2HSV(cartoon_vb_rgb);
					//去掉过于明显的数据
					//if(vb_hsv[2]<200)
					{
						vf_hsv[0] = vb_hsv[0];
						//加上下面的，脸上就显得乱了，很脏。
						//vf_hsv[1] = vf_hsv[1]+vb_hsv[1]*0.3;N

						//vf_hsv[2] = vf_hsv[2]+vb_hsv[2]*0.3;
					}
					vf_rgb = kcvHSV2RGB(vf_hsv);
					//如果脸模板相同位置不透明，说明是有内容的
					if(colDataBGRA[_col][3]>5){
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
						bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
						bgra_frame_light_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
					} else {
						//如果脸模板相同位置透明，说明是捏脸的部分，要被去掉

					}
					//*/
					//先改成不透明
					//bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
					continue;
				}
				if(mask_real_face_data[index] < 32){
					continue;
				}
				//如果脸部范围为白色，直接取脸色值。
				if (mask_real_face_data[index] > 223){
					bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
					bgra_frame_light_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
				}
				//否则，要做透明化处理
				else {
					//bgra_frame_data[index][3] = 255;
					//double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
					//bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2], 255 - mask_real_face_data[index]);
					//变化明暗度
					Vec3b vf_rgb = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
					Vec3b vb_rgb = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
					Vec3b vf_hsv = kcvRGB2HSV(vf_rgb);
					Vec3b vb_hsv = kcvRGB2HSV(vb_rgb);
					
					vf_hsv[0] = vb_hsv[0];
					//vf_hsv[1] = vb_hsv[1];
					//vf_hsv[2] = vb_hsv[2];
					
					vf_rgb = kcvHSV2RGB(vf_hsv);
					double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
					//bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], 255);
					bgra_frame_light_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], 255);
					bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
					//bgra_frame_light_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
					//*/
					//bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2], (1-rate)*mask_real_face_data[index]);
					
					/*//根据透明度与模板数据
					bgra_frame_data[index][2] = saturate_cast<uchar>((1-rate)*colData[_col][2] + rate * bgra_frame_data[index][2]);
					bgra_frame_data[index][1] = saturate_cast<uchar>((1-rate)*colData[_col][1] + rate * bgra_frame_data[index][1]);
					bgra_frame_data[index][0] = saturate_cast<uchar>((1-rate)*colData[_col][0] + rate * bgra_frame_data[index][0]);
					//*/

				}				
			}
			else{
				if(maskData[index] < 5)
					continue;
				bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
				bgra_frame_light_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
			}
		}
	}

	//rectangle(_bgraFrame, this->faceDetectedRect, Scalar(0,0,255));
	rectangle(_bgraFrame, this->faceMiddleRect, Scalar(0,0,255));

	imwrite("MedianData//bgrLightTemp.png", _bgraFrameLight);
	imwrite("MedianData//bgrTemp.png", _bgraFrame);
}

void appFace::resizeNoseModel(int mode){

	Mat noseModel = chm.currentHead.nose.model;

	//换完，先换鼻子。设定鼻子的位置：居中，靠上。
	Mat noseSample;
	Rect noseRect;
	if(this->noseDetectedRect.x>0){
		float noseHeight,noseWidth;

		if(mode == REALMODEPLUS){
			noseHeight = (((float)(noseDetectedRect.y)+((float)(noseDetectedRect.height))/2) - ((float)(eyeDetectedRects[0].y)+((float)(eyeDetectedRects[0].height))/2))*1.1;
			noseWidth = ((float)noseDetectedRect.width) *3/4;
		}

		if(mode == REALMODE){
			noseHeight = (((float)(noseDetectedRect.y)+((float)(noseDetectedRect.height))/2) - ((float)(eyeDetectedRects[0].y)+((float)(eyeDetectedRects[0].height))/2))*1.1;
			noseWidth = ((float)noseDetectedRect.width) *3/4;
		}

		if(mode == QFITMODE){
			noseHeight = ((float)(noseDetectedRect.y)+((float)(noseDetectedRect.height)/2)) - ((float)(eyeDetectedRects[0].y)+((float)(eyeDetectedRects[0].height))/2);
			noseWidth = noseModel.cols*((float)noseHeight/(float)noseModel.rows);
		}
		//cout << (int)noseHeight << " " << (int)noseWidth << endl;
		//设定鼻子的区域。
		noseRect = Rect(
			//将模板移到中线为中心的位置，用noseDectctedRect的y坐标
			this->faceMiddleRect.x - noseWidth/2,
			//鼻子上边要到眼前中心，下边在鼻子框的中心,再向下移1/8.
			eyeDetectedRects[0].y+eyeDetectedRects[0].height/2+noseHeight/10,
			noseWidth,
			noseHeight);
		//按比例放大缩小鼻子宽度
		//取均值的缩放比例，缩放模板	
		//resize(noseModel, noseSample, Size(0, 0), (fWidth+fHeight)/2, (fWidth+fHeight)/2);
		resize(noseModel, noseSample, Size(noseWidth, noseHeight));
		replaceNose(_bgraFrameLight, noseSample,noseRect,maskRealFace);
		replaceNose(_bgraFrame, noseSample,noseRect,maskRealFace);
	}

	//rectangle(_bgraFrame, noseRect, Scalar(0,0,0));
	//rectangle(_bgraFrame, this->noseDetectedRect, Scalar(0,0,0));

	imwrite("MedianData//bgrLightTempWithNose.png", _bgraFrameLight);
	imwrite("MedianData//bgrTempWithNose.png", _bgraFrame);


}

void appFace::resizeEyes(int mode){
	//-- Resize eyeModels to the eyes' sizes

	int mean_width = eyeDetectedRects[0].width/2 + eyeDetectedRects[1].width/2;
	int mean_height = eyeDetectedRects[0].height/2 + eyeDetectedRects[1].height/2;
	eyeDetectedRects[0] = Rect(eyeDetectedRects[0].x + eyeDetectedRects[0].width/2 - mean_width/2, 
		eyeDetectedRects[0].y + eyeDetectedRects[0].height/2 - mean_height/2,
		mean_width,mean_height);
	eyeDetectedRects[1] = Rect(eyeDetectedRects[1].x + eyeDetectedRects[1].width/2 - mean_width/2, 
		eyeDetectedRects[1].y + eyeDetectedRects[1].height/2 - mean_height/2,
		mean_width,mean_height);

	if (this->eyeNumber == 2){

		//按缩放比小的值，等比放缩眼睛
		double eHeight=0.0;
		double eWeith=0.0;
		resizeEyeRate = 0.0;

		//如果是增强写实风格8，按检测眼睛大小缩放
		if(mode == REALMODEPLUS){		//REALMODE，写实版。按实际五官大小、位置，乘以接近1的系数，放五官。
			//先取左眼模板3个点
			Point ps[2][3];//取眼睛模板的左、右点。lr:0取左眼白点，1取右眼白点，2取左眼边缘，3取右眼边缘，4取眼白上边缘，5取眼上边缘。
			ps[0][0] = this->getEyeModelPoint(leftEyeModel,0);
			ps[0][1] = this->getEyeModelPoint(leftEyeModel,3);
			ps[0][2] = this->getEyeModelPoint(leftEyeModel,5);
			ps[0][0] = this->getEyeModelPoint(rightEyeModel,1);
			ps[0][1] = this->getEyeModelPoint(rightEyeModel,2);
			ps[0][2] = this->getEyeModelPoint(rightEyeModel,5);

			resizeModel(leftEyeModel,ps[0][0],ps[0][1],ps[0][2],eyesPoint[2],eyesPoint[3],eyesPoint[0]);
			resizeModel(rightEyeModel,ps[1][0],ps[1][1],ps[1][2],eyesPoint[6],eyesPoint[7],eyesPoint[4]);

		}



		//如果是写实风格1，按检测眼睛大小缩放
		if(mode == REALMODE){		//REALMODE，写实版。按实际五官大小、位置，乘以接近1的系数，放五官。
			eHeight = (double)((double)eyeDetectedRects[0].height / (double)leftEyeModel.rows);
			eWeith = (double)((double)eyeDetectedRects[0].width / (double)leftEyeModel.cols);
			if(eWeith>eHeight)
				resizeEyeRate = eHeight;
			else
				resizeEyeRate = eWeith;

			//现在是等比缩小了0.9，真实情况应该根据眼在框里的实际比例。
			resize(leftEyeModel, leftEyeSample, Size(leftEyeModel.cols*resizeEyeRate*0.9, leftEyeModel.rows*resizeEyeRate*0.9));
			resize(rightEyeModel, rightEyeSample, Size(rightEyeModel.cols*resizeEyeRate*0.9, rightEyeModel.rows*resizeEyeRate*0.9));
			resize(leftEyeWithBrowModel, leftEyeWithBrowSample, Size(leftEyeWithBrowModel.cols*resizeEyeRate*0.9, leftEyeWithBrowModel.rows*resizeEyeRate*0.9));
			resize(rightEyeWithBrowModel, rightEyeWithBrowSample, Size(rightEyeWithBrowModel.cols*resizeEyeRate*0.9, rightEyeWithBrowModel.rows*resizeEyeRate*0.9));
			resize(leftEyePupilModel,leftEyePupilSample,Size(leftEyePupilModel.cols*resizeEyeRate*0.9, leftEyePupilModel.rows*resizeEyeRate*0.9));
			resize(rightEyePupilModel,rightEyePupilSample,Size(rightEyePupilModel.cols*resizeEyeRate*0.9, rightEyePupilModel.rows*resizeEyeRate*0.9));
		}
		//如果是Q萌风格2，按脸缩放比例缩放眼睛
		if(mode == QMODE){
			resize(leftEyeModel, leftEyeSample, Size(0, 0), fWidth, fHeight);
			resize(rightEyeModel, rightEyeSample, Size(0, 0), fWidth, fHeight);
			resize(leftEyePupilModel,leftEyePupilSample,Size(0,0),fWidth, fHeight);
			resize(rightEyePupilModel,rightEyePupilSample,Size(0,0),fWidth, fHeight);
		}

		//如果是卡通风格3，按小的等比缩放
		if(mode == QFITMODE){
			if (fWidth < fHeight)
				resizeEyeRate = fWidth;
			else
				resizeEyeRate = fHeight;

			resize(leftEyeModel, leftEyeSample, Size(0, 0), resizeEyeRate, resizeEyeRate);
			resize(rightEyeModel, rightEyeSample, Size(0, 0), resizeEyeRate, resizeEyeRate);
			resize(leftEyeWithBrowModel, leftEyeWithBrowSample, Size(0, 0), resizeEyeRate, resizeEyeRate);
			resize(rightEyeWithBrowModel, rightEyeWithBrowSample, Size(0, 0), resizeEyeRate, resizeEyeRate);
			resize(leftEyePupilModel,leftEyePupilSample,Size(0,0),resizeEyeRate, resizeEyeRate);
			resize(rightEyePupilModel,rightEyePupilSample,Size(0,0),resizeEyeRate, resizeEyeRate);
		}


	//replaceEyes( _bgraFrame,leftEyeSample,  rightEyeSample, leftEyeWithBrowSample, rightEyeWithBrowSample, leftEyePupilSample, rightEyePupilSample, maskRealFace);
	//replaceEyes(mode);
	}
		// TODO 
	//如果只检测到一只眼睛，根据对称，推导出另一只眼睛。
	else if (this->eyeNumber == 1)
	{
		if (this->eyeDetectedRects[0].x < this->faceChangeRect.x + 0.5 * this->faceChangeRect.width)
		{
			leftEyeRect = this->eyeDetectedRects[0];
			rightEyeRect = Rect(this->eyeDetectedRects[0].x + 0.5 * this->faceChangeRect.width, this->faceChangeRect.y, 
				this->eyeDetectedRects[0].width, this->eyeDetectedRects[0].height);
			rightEyePupilRect = eyeDetectedRects[0];
		}
		else
		{
			rightEyeRect = this->eyeDetectedRects[0];
			leftEyeRect = Rect(this->eyeDetectedRects[0].x - 0.5 * this->faceChangeRect.width, this->faceChangeRect.y, 
				this->eyeDetectedRects[0].width, this->eyeDetectedRects[0].height);
			leftEyePupilRect = eyeDetectedRects[0];
		}
	}

}

void appFace::replaceEyes(int mode){
	//移动眼睛
	//因为检测的是眼球，所以要根据中线，移动眼睛到相应位置。
	//如果眼睛eyeDetectedRects[0]在左半边脸
	//if (this->eyeDetectedRects[0].x < this->faceDetectedRect.x + 0.5 * this->faceDetectedRect.width)
	int leftEyeNum = 0,rightEyeNum = 1;
	if (this->eyeDetectedRects[0].x > faceMiddleRect.x){
		leftEyeNum = 1;
		rightEyeNum = 0;
	}

	//改为从中心线对齐。_dif真正眼睛的偏移量，也就是瞳孔在眼中心的偏移量，也是眼睛移动量。
	//***************** 保持两眼间距 ****************
	//眼睛间距
	int _tj = (eyeDetectedRects[rightEyeNum].x - (eyeDetectedRects[leftEyeNum].x+eyeDetectedRects[leftEyeNum].width))/2;
	//向右偏为正，向左偏为负
	int _py = ((eyeDetectedRects[rightEyeNum].x-_tj) - faceMiddleRect.x);
			
	//眼睛，以瞳孔y轴中心为中心，上下居中；以中线为中心，增加眼间距。
	leftEyeRect = Rect(
		//左眼X = 中线，向左眼睛模型距离，再移瞳孔
		faceMiddleRect.x - leftEyeSample.cols - _tj,
		eyeDetectedRects[leftEyeNum].y + 0.5*eyeDetectedRects[leftEyeNum].height-0.5*leftEyeSample.rows, 
		leftEyeSample.cols, 
		leftEyeSample.rows );
	rightEyeRect = Rect(
		//右眼X= 中线，向右移瞳孔
		faceMiddleRect.x + _tj,
		eyeDetectedRects[rightEyeNum].y + 0.5*eyeDetectedRects[rightEyeNum].height-0.5*leftEyeSample.rows, 
		rightEyeSample.cols, 
		rightEyeSample.rows );
	/*//保持眼中心模式。
	leftEyeWithBrowRect = Rect(
		leftEyeRect.x + leftEyeRect.width/2 - leftEyeWithBrowSample.cols/2,
		leftEyeRect.y - (leftEyeWithBrowSample.rows - leftEyeSample.rows),
		leftEyeWithBrowSample.cols,
		leftEyeWithBrowSample.rows);
	rightEyeWithBrowRect = Rect(
		rightEyeRect.x +rightEyeRect.width/2 - rightEyeWithBrowSample.cols/2,
		rightEyeRect.y -(rightEyeWithBrowSample.rows - rightEyeSample.rows),
		rightEyeWithBrowSample.cols,
		rightEyeWithBrowSample.rows);
	//*/
	// 保持眼间距离模式。以中线为基准，所以当眉毛两侧突出于眼时，就会偏。Y轴以眼底为平，向上扩展。
	leftEyeWithBrowRect = Rect(
		faceMiddleRect.x - leftEyeWithBrowSample.cols - _tj,
		leftEyeRect.y - (leftEyeWithBrowSample.rows - leftEyeSample.rows),
		leftEyeWithBrowSample.cols,
		leftEyeWithBrowSample.rows);
	rightEyeWithBrowRect = Rect(faceMiddleRect.x + _tj,
		rightEyeRect.y -(rightEyeWithBrowSample.rows - rightEyeSample.rows),
		rightEyeWithBrowSample.cols,
		rightEyeWithBrowSample.rows);
	//*/
	leftEyePupilRect = Rect(
		//左瞳孔X = 左眼中线 + 按缩放倍数偏移
		leftEyeRect.x + leftEyeRect.width/2 - leftEyePupilSample.cols/2 + _py*resizeEyeRate,
		eyeDetectedRects[leftEyeNum].y+ (0.5*eyeDetectedRects[leftEyeNum].height  - 0.5*leftEyePupilSample.rows),
		leftEyePupilSample.cols, 
		leftEyePupilSample.rows );
	rightEyePupilRect = Rect(
		//右瞳孔X = 右眼中线 + 按缩放倍数偏移
		rightEyeRect.x + rightEyeRect.width/2 - rightEyePupilSample.cols/2 + _py*resizeEyeRate,
		eyeDetectedRects[rightEyeNum].y+ 0.5*eyeDetectedRects[leftEyeNum].height  - 0.5*rightEyePupilSample.rows,
		rightEyePupilSample.cols, 
		rightEyePupilSample.rows );
	//要从眼的中心定位，而不是从左端，否则会造成左眼过中线的问题。

	Vec4b *bgra_frame_data = _bgraFrame.ptr<Vec4b>(0);
	Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);
	uchar *maskData = maskFace.ptr<uchar>(0);

	//-- Change left eye
	for (int _row = 0; _row < leftEyeWithBrowSample.rows ; _row++)
	{
		Vec4b *colData = leftEyeWithBrowSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < leftEyeWithBrowSample.cols ; _col++){
			//-- Get valid area of face model
			if (colData[_col][3] == 0)
			{
				continue;
			}
			
			//if (colData[_col][0] > 250 && colData[_col][1] > 250 && colData[_col][2] > 250)
			//	continue;
			int r = leftEyeWithBrowRect.y + _row; 
			int c = leftEyeWithBrowRect.x + _col;
			int index = r*frame.cols + c;

			//-- Override face where mask > 0
			if (maskData[index] == 0)
				continue;
			//frameData[index] = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
			bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
			bgra_frame_light_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
		}
	}

	//-- Change right eye
	for (int _row = 0; _row < rightEyeWithBrowSample.rows ; _row++)
	{
		Vec4b *colData = rightEyeWithBrowSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < rightEyeWithBrowSample.cols ; _col++){
			//-- Get valid area of face model
			if (colData[_col][3] == 0)
			{
				continue;
			}
			int r = rightEyeWithBrowRect.y + _row; 
			int c = rightEyeWithBrowRect.x + _col;
			int index = r*frame.cols + c;
			
			//-- Override face where mask > 0
			if (maskData[index] == 0)
				continue;
			//frameData[index] = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
			bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
			bgra_frame_light_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
		}
	}

	//-- Change left eye pupil
	for (int _row = 0; _row < leftEyePupilSample.rows ; _row++)
	{
		//Vec4b *colDataEye = leftEyeSample.ptr<Vec4b>(_row);
		Vec4b *colData = leftEyePupilSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < leftEyePupilSample.cols ; _col++){
			//-- Get valid area of face model
			if (colData[_col][3] == 0)
			{
				continue;
			}
			int r = leftEyePupilRect.y + _row; 
			int c = leftEyePupilRect.x + _col;
			int index = r*frame.cols + c;

			//-- Override face where mask > 0
			if (maskData[index] == 0)
				continue;
			//frameData[index] = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
			//找到这个位置点，眼睛模板中的数据，因为画眼睛时已经变化为眼睛值，所以取背景数据就行了。
			//用公式计算出两色混合后的颜色
			//上面错了。应该是背景有透明值，就取眼珠的值。
			/*
			int _r = (int)(bgra_frame_data[index][0]*bgra_frame_data[index][3]*(1-colData[_col][3])+colData[_col][0]*colData[_col][3])/(bgra_frame_data[index][3]+colData[_col][3]+bgra_frame_data[index][3]*colData[_col][3]);
			int _g = (int)(bgra_frame_data[index][1]*bgra_frame_data[index][3]*(1-colData[_col][3])+colData[_col][1]*colData[_col][3])/(bgra_frame_data[index][3]+colData[_col][3]+bgra_frame_data[index][3]*colData[_col][3]);
			int _b = (int)(bgra_frame_data[index][2]*bgra_frame_data[index][3]*(1-colData[_col][3])+colData[_col][2]*colData[_col][3])/(bgra_frame_data[index][3]+colData[_col][3]+bgra_frame_data[index][3]*colData[_col][3]);
			*/
			//赋值，透明度255
			//现在没有眼睛模板，先注释掉用着。
			if(bgra_frame_data[index][3]<160 && bgra_frame_data[index][3]>100){
				bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
				bgra_frame_light_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
			}
			else{
				bgra_frame_data[index] = Vec4b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2],255);
				bgra_frame_light_data[index] = Vec4b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2],255);
			}

		}
	}

	//-- Change right eye pupil
	for (int _row = 0; _row < rightEyePupilSample.rows ; _row++)
	{
		Vec4b *colData = rightEyePupilSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < rightEyePupilSample.cols ; _col++){
			//-- Get valid area of face model
			if (colData[_col][3] == 0)
			{
				continue;
			}
			int r = rightEyePupilRect.y + _row; 
			int c = rightEyePupilRect.x + _col;
			int index = r*frame.cols + c;

			//-- Override face where mask > 0
			if (maskData[index] == 0)
				continue;

			if(bgra_frame_data[index][3]<160 && bgra_frame_data[index][3]>100){
				bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
				bgra_frame_light_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
			}
			else{
				bgra_frame_data[index] = Vec4b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2],255);
				bgra_frame_light_data[index] = Vec4b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2],255);
			}
		}
	}
	rectangle(_bgraFrame, rightEyeRect, Scalar(0,0,0));
	rectangle(_bgraFrame, leftEyeRect, Scalar(0,0,0));
	rectangle(_bgraFrame, rightEyeWithBrowRect, Scalar(255,255,255));
	rectangle(_bgraFrame, leftEyeWithBrowRect, Scalar(255,255,255));
	
	//rectangle(_bgraFrame, eyeDetectedRects[0], Scalar(255,255,255));
	//rectangle(_bgraFrame, eyeDetectedRects[1], Scalar(255,255,255));
	rectangle(_bgraFrame, rightEyePupilRect, Scalar(255,255,255));
	rectangle(_bgraFrame, leftEyePupilRect, Scalar(255,255,255));


}

void appFace::resizeMouth(int mode){
	//最后换嘴。设定嘴的位置：居中，靠上。
	Mat mouthSample;
	Rect mouthRect;
	if(this->mouthDetectedRect.x>0){
		double mouthResize = 0.0;
		int mouthWidth ;
		int mouthHeight ;

		//如果是写实风格
		if(mode == REALMODE){

			//按等比例缩放，省事，没有风格。
			mouthResize = (fWidth+fHeight)/2;

			//如果嘴宽过检测框，说明脸计算宽了。
			if(mouthModel.cols*mouthResize > this->mouthDetectedRect.width){
				//嘴宽度取检测宽度的4/5。
				mouthWidth = this->mouthDetectedRect.width;
				mouthHeight = mouthModel.rows*((double)mouthWidth/(double)mouthModel.cols);
				//按比例放大缩小嘴
				//取均值的缩放比例，缩放模板	
				resize(mouthModel, mouthSample, Size(mouthWidth, mouthHeight));
				//设定嘴的区域。将模板移到中线为中心的位置，用mouthDectctedRect的y坐标
			} else {
				//resize(mouthModel, mouthSample, Size(mouthWidth, mouthHeight));
				resize(mouthModel, mouthSample, Size(0, 0),mouthResize,mouthResize);
			}
		}

		if(mode == QFITMODE){
			//按等比例缩放，省事，没有风格。
			mouthResize = (fWidth+fHeight)/2;
			resize(mouthModel, mouthSample, Size(0, 0),mouthResize,mouthResize);
		}

		if(mode == QMODE){
			//按等比例缩放，省事，没有风格。
			mouthResize = (fWidth+fHeight)/2;
			resize(mouthModel, mouthSample, Size(0, 0),mouthResize,mouthResize);
		}


		//嘴的上沿是鼻子的下沿与检测到的嘴的上沿的中间，或嘴的1/5
		//if(this->noseDetectedRect.x>0){
		//	mouthRect = Rect(this->faceMiddleRect.x - mouthSample.cols/2,this->mouthDetectedRect.y+(noseRect.y+noseRect.height-this->mouthDetectedRect.y)/2,mouthSample.cols,mouthSample.rows);
		//} else
		if(mode == REALMODE){
			mouthRect = Rect(this->faceMiddleRect.x - mouthSample.cols/2,
				mouthDetectedRect.y+mouthDetectedRect.height*1/10,
				mouthSample.cols,
				mouthSample.rows);
		}

		if(mode == QFITMODE || mode == QMODE){
			mouthRect = Rect(this->faceMiddleRect.x - mouthSample.cols/2,
				mouthDetectedRect.y+mouthDetectedRect.height/3,
				mouthSample.cols,
				mouthSample.rows);
		}

		replaceMouth(_bgraFrameLight, mouthSample,mouthRect,maskRealFace);
		replaceMouth(_bgraFrame, mouthSample,mouthRect,maskRealFace);
	}

	//rectangle(_bgraFrame, mouthDetectedRect, Scalar(0,0,255));
	//rectangle(_bgraFrame, mouthRect, Scalar(0,0,0));

}

void appFace::replaceFaceAndEyes(Mat &resultImage,int mode){
	initCounter(); // 初始化
	initTempMat();

	// ================= 换脸 ========================
	resizeFaceModel(mode);
	Mat faceModel = chm.currentHead.faceModel;
	replaceFace(faceModel,resultImage,mode);

	// ================= 换鼻子 ========================
	resizeNoseModel(mode);

	// ================= 换眼睛 ========================
	resizeEyes(mode);
	replaceEyes(mode);

	// ================= 换嘴 ========================
	resizeMouth(mode);




	saveImages(_bgraFrameLight,"lightBeforeChangeFace.png");
	saveImages(_bgraFrame,"beforeChangeFace.png");
	//捏脸瘦脸
	imwrite("MedianData//bgrLightBeforeChangeFace.png", _bgraFrameLight);
	//changeFace(_bgraFrameLight,mask_face_replace_data,faceSample);
	imwrite("MedianData//bgrLightAfterChangeFace.png", _bgraFrameLight);

	//changeFace(_bgraFrame,mask_face_replace_data,faceSample);
	imwrite("MedianData//bgrAfterChangeFace.png", _bgraFrame);
	//String fileName;
	saveImages(_bgraFrame,"afterChangeFace.png");
	saveImages(_bgraFrameLight,"lightAfterChangeFace.png");

	//Blur and sharpen
	//postProcessus(frame);
	_bgraFrame.copyTo(resultImage);
	imwrite("ResultData//2.png", _bgraFrame);
	imwrite("ResultData//3.png", _bgraFrameLight);

}


//frame是原始图像
void appFace::replaceFaceAndEyes(Mat faceModel, Mat leftEyeModel, Mat rightEyeModel,Mat leftEyeWithBrowModel,Mat rightEyeWithBrowModel, Mat &frame, Mat leftEyePupilModel,Mat rightEyePupilModel,Mat mouthModel,Mat noseModel,int mode){
	initCounter();
}


void appFace::postProcessus(Mat &image)
{
	Mat blurImage;
	GaussianBlur(image, blurImage, Size(3,3), 0.7, 0.7);

	Vec3b *image_data = image.ptr<Vec3b>(0);
	Vec3b *blur_image_data = blurImage.ptr<Vec3b>(0);
	uchar *contour_data = imageContourSM.ptr<uchar>(0);
	int total = image.cols * image.rows;

	for (int i = 0 ; i < total ; i++){
		if (contour_data[i] == 255)
			image_data[i] = blur_image_data[i];
	}
}

Mat appFace::generateTrimapOfFace(int dSize){
	Mat edge;
	Canny(maskFace, edge, 1, 3);
	Mat trimapOfFace;
	maskFace.copyTo(trimapOfFace);
	int totalEdge = edge.rows * edge.cols;
	uchar *trimapData = trimapOfFace.ptr<uchar>(0);
	uchar *colData = edge.ptr<uchar>(0);
	for (int _row = 0; _row < edge.rows ; _row++)
	{
		for (int _col = 0 ; _col < edge.cols ; _col++)
		{
			int index = _row * edge.cols + _col;
			if (colData[index] == 0)
				continue;

			for(int _c = -dSize; _c < dSize; _c++)
			{
				for (int _r = -dSize; _r < dSize; _r++)
				{
					int changeIndex = index + _c + _r*edge.cols;
					if (changeIndex >=0 && changeIndex < totalEdge)
					{
						trimapData[changeIndex] = 132;
					}
				}
			}
		}
	}
	for (int _i = 0 ; _i < totalEdge ;_i++)
	{
		if (trimapData[_i] == 0)
		{
			trimapData[_i] = 4;
		}
		else if (trimapData[_i] == 255)
		{
			trimapData[_i] = 252;
		}
		else
			trimapData[_i] = 132;
	}

	cvtColor(trimapOfFace,trimapOfFace,CV_GRAY2BGR);
	imwrite("MedianData//trimapFace.jpg", trimapOfFace);

	return trimapOfFace;
}


void appFace::saveImages(Mat resultImage,String fileName){
	Mat _face;
	//cvtColor(resultImage, _face, CV_BGR2BGRA);
	resultImage.copyTo(_face);

	Mat _contour;
	//cvtColor(resultImage, _contour, CV_BGR2BGRA);
	resultImage.copyTo(_contour);

	int totalPixels = _contour.rows * _contour.cols;
	Vec4b* face_data = _face.ptr<Vec4b>(0);

	//切整齐的人脸
	uchar* mask_face_data = maskFaceReplace.ptr<uchar>(0);
	//背景数据
	Vec4b* contour_data = _contour.ptr<Vec4b>(0);
	//真人外边缘
	uchar* mask_real_contour_data = imageRealContourSM.ptr<uchar>(0);
	//切整齐的真人外边缘
	uchar* mask_contour_data = imageContourSM.ptr<uchar>(0);

	//	changeFace(_contour,mask_face_data,_face);

	
	for(int i = 0 ; i < totalPixels ; i++)
	{
		if (mask_face_data[i] == 0)
		{
			face_data[i] = Vec4b(0,0,0,0);
		}
		else
		{
			//face_data[i] = Vec4b(face_data[i][0], face_data[i][1], face_data[i][2],mask_real_contour_data[i]);
			contour_data[i] = Vec4b(0,0,0,0);
			continue;
		}
		
		if (mask_contour_data[i] == 0)
		{
			contour_data[i] = Vec4b(0,0,0,0);
		}
	}

		imwrite("MedianData//face"+fileName, _face);
		imwrite("MedianData//contour"+fileName, _contour);
	
	/*
	//亮度平衡处理
	IplImage* imageFace = &IplImage(_face);
	IplImage* imageBgr = &IplImage(_contour);
	IplImage* imageBgrSkin = &IplImage(_contour);
	double gFace = get_avg_gray(imageFace);
	double gBgr = get_avg_gray(imageBgr);
	set_avg_gray(imageBgr,imageBgr,gFace*0.7);
	*/


}

void appFace::saveImages1(Mat resultImage){
	Mat _face;
	//cvtColor(resultImage, _face, CV_BGR2BGRA);
	resultImage.copyTo(_face);

	Mat _contour;
	//cvtColor(resultImage, _contour, CV_BGR2BGRA);
	resultImage.copyTo(_contour);

	int totalPixels = _contour.rows * _contour.cols;
	Vec4b* face_data = _face.ptr<Vec4b>(0);

	//切整齐的人脸
	uchar* mask_face_data = maskFaceReplace.ptr<uchar>(0);
	//背景数据
	Vec4b* contour_data = _contour.ptr<Vec4b>(0);
	//真人外边缘
	uchar* mask_real_contour_data = imageRealContourSM.ptr<uchar>(0);
	//切整齐的真人外边缘
	uchar* mask_contour_data = imageContourSM.ptr<uchar>(0);

	//	changeFace(_contour,mask_face_data,_face);

	
	for(int i = 0 ; i < totalPixels ; i++)
	{
		if (mask_face_data[i] == 0)
		{
			face_data[i] = Vec4b(0,0,0,0);
		}
		else
		{
			//face_data[i] = Vec4b(face_data[i][0], face_data[i][1], face_data[i][2],mask_real_contour_data[i]);
			contour_data[i] = Vec4b(0,0,0,0);
			continue;
		}
		
		if (mask_contour_data[i] == 0)
		{
			contour_data[i] = Vec4b(0,0,0,0);
		}
	}


	imwrite("MedianData//face_only1.png", _face);
	imwrite("MedianData//contour_only1.png", _contour);
//	imwrite("MedianData//cartoonFilter//test.jpg", _contour);
	
	/*
	//亮度平衡处理
	IplImage* imageFace = &IplImage(_face);
	IplImage* imageBgr = &IplImage(_contour);
	IplImage* imageBgrSkin = &IplImage(_contour);
	double gFace = get_avg_gray(imageFace);
	double gBgr = get_avg_gray(imageBgr);
	set_avg_gray(imageBgr,imageBgr,gFace*0.7);
	*/


}


Vec3b appFace::kcvRGB2HSV(Vec3b img)  
{  
	Vec3b ret = Vec3b();
	//顺序可能错了
	int b=img[2];  
	int g=img[1];  
	int r=img[0];  
	int maxval=max(b,max(g,r));  
	int minval=min(b,min(g,r));  
	int v=maxval;  
	double diff=maxval-minval;  
	int s=diff*255/(v+DBL_EPSILON);  

	double h=0;  
	diff=60/(diff+DBL_EPSILON);  
	if(v==r)  
	{  
		h=(g-b)*diff;  
	}  
	else if(v==g)  
	{  
		h=(b-r)*diff+120.f;  
	}  
	else  
	{  
		h=(r-g)*diff+240.f;  
	}  
	if( h<0)  
	{  
		h+=360.f;  
	}  
	ret[0]=h/2;  
	ret[1]=s;  
	ret[2]=v;  

	return ret;  
}  

Vec3b appFace::kcvHSV2RGB(Vec3b img)  
{  
	Vec3b ret = Vec3b();
	int h=img[0];  
	int s=img[1];  
	int v=img[2];  
	int c=(double)v*s/255;  
	double hh=(double)h*2/60;  
	double x=c*(1-abs(fmod(hh,2)-1));  
	int r,g,b;  
	if(0<=hh&&hh<1)  
	{  
		r=c;g=x;b=0;  
	}  
	else if(1<=hh&hh<2)  
	{  
		r=x;g=c;b=0;  
	}  
	else if(2<=hh&&hh<3)  
	{  
		r=0;g=c;b=x;  
	}  
	else if(3<=hh&&hh<4)  
	{  
		r=0;g=x;b=c;  
	}  
	else if(4<=hh&&hh<5)  
	{  
		r=x;g=0;b=c;  
	}  
	else  
	{  
		r=c;g=0;b=x;  
	}  
	int m=v-c;  
	ret[2]=b+m;  
	ret[1]=g+m;  
	ret[0]=r+m;  

	return ret;  
}  

Vec3b appFace::kcvRGB2HSL(Vec3b img) {

		Vec3b ret = Vec3b();

	int R=img[0];  
	int G=img[1];  
	int B=img[2];  
	
	int H,S,L;
	
    double Max,Min,del_R,del_G,del_B,del_Max;

    Min = min(R, min(G, B));    //Min. value of RGB
    Max = max(R, max(G, B));    //Max. value of RGB
    del_Max = Max - Min;        //Delta RGB value

    L = (Max + Min) / 2.0;

    if (del_Max == 0)           //This is a gray, no chroma...
    {
        //H = 2.0/3.0;          //Windows下S值为0时，H值始终为160（2/3*240）
        H = 0;                  //HSL results = 0 ÷ 1
        S = 0;
    }
    else                        //Chromatic data...
    {
        if (L < 0.5) S = del_Max / (Max + Min);
        else         S = del_Max / (2 - Max - Min);

        del_R = (((Max - R) / 6.0) + (del_Max / 2.0)) / del_Max;
        del_G = (((Max - G) / 6.0) + (del_Max / 2.0)) / del_Max;
        del_B = (((Max - B) / 6.0) + (del_Max / 2.0)) / del_Max;

        if      (R == Max) H = del_B - del_G;
        else if (G == Max) H = (1.0 / 3.0) + del_R - del_B;
        else if (B == Max) H = (2.0 / 3.0) + del_G - del_R;

        if (H < 0)  H += 1;
        if (H > 1)  H -= 1;
    }	
	ret[0]=(int)H;  
	ret[1]=(int)S;  
	ret[2]=(int)L;  
	
	return ret;  
}

Vec3b appFace::kcvHSL2RGB(Vec3b img){

    double R,G,B;
	double H,S,L;
	H = img[0];
	S = img[1];
	L = img[2];

    double var_1, var_2;
    if (S == 0)                       //HSL values = 0 ÷ 1
    {
        R = L * 255.0;                   //RGB results = 0 ÷ 255
        G = L * 255.0;
        B = L * 255.0;
    }
    else
    {
        if (L < 0.5) var_2 = L * (1 + S);
        else         var_2 = (L + S) - (S * L);

        var_1 = 2.0 * L - var_2;

        R = 255.0 * Hue2RGB(var_1, var_2, H + (1.0 / 3.0));
        G = 255.0 * Hue2RGB(var_1, var_2, H);
        B = 255.0 * Hue2RGB(var_1, var_2, H - (1.0 / 3.0));
    }

	Vec3d ret = Vec3d(R,G,B);
	return ret;
}

double appFace::Hue2RGB(double v1, double v2, double vH)
{
    if (vH < 0) vH += 1;
    if (vH > 1) vH -= 1;
    if (6.0 * vH < 1) return v1 + (v2 - v1) * 6.0 * vH;
    if (2.0 * vH < 1) return v2;
    if (3.0 * vH < 2) return v1 + (v2 - v1) * ((2.0 / 3.0) - vH) * 6.0;
    return (v1);
}

int appFace::averageLight(Mat faceSampleBGR){
	
	double count = 1.0;
	double lightV = 0.0;
	for(int i=0;i<faceSampleBGR.rows;i++){
		Vec4b *colDataBGRA = faceSampleBGR.ptr<Vec4b>(i);
		Vec3b *colDataBGR = faceSampleBGR.ptr<Vec3b>(i);
		for(int j=0;j<faceSampleBGR.cols;j++){
			int a = colDataBGRA[j][3];
			cout << i << "  " << j << "  " << a << endl;
			if (a > 10)
			{
				//转换成亮度
				Vec3d p = kcvRGB2HSL(colDataBGR[j]);
				//累加
				count++;
				lightV = lightV + p[2];
			}
		}
	}
	int ret = (int)lightV / count;
	return ret;
}

void appFace::adjustLight(Mat faceSampleBGR,float _v){
	Vec3b *bgra_frame_data = faceSampleBGR.ptr<Vec3b>(0);
	for(int i=0;i<faceSampleBGR.rows;i++){
		Vec4b *colDataBGRA = faceSampleBGR.ptr<Vec4b>(i);
		Vec3b *colDataBGR = faceSampleBGR.ptr<Vec3b>(i);
		for(int j=0;j<faceSampleBGR.cols;j++){
			int a = colDataBGRA[j][3];
			if (a > 10)
			{
				//转换成亮度
				Vec3d p = kcvRGB2HSL(colDataBGR[j]);
				p[2] = (int)p[2]*_v;
				p = kcvHSL2RGB(p);
				bgra_frame_data[i*faceSampleBGR.rows+j] = p;
			}
		}
	}

}

double appFace::get_avg_gray(IplImage *img)
{
    IplImage *gray = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
    cvCvtColor(img,gray,CV_RGB2GRAY);
    CvScalar scalar = cvAvg(gray);
    cvReleaseImage(&gray);
    return scalar.val[0];
}

void appFace::set_avg_gray(IplImage *img,IplImage *out,double avg_gray)
{
	double prev_avg_gray = get_avg_gray(img);
	if(prev_avg_gray>0 && avg_gray > 0){
		cout << (double)(avg_gray/prev_avg_gray) << endl;
		cvConvertScale(img,out,(double)(avg_gray/prev_avg_gray));
	}
	else
		cout << "Div by zero!" << endl;
	int point;
}

void appFace::cvSkinHSV(IplImage* src,IplImage* dst)    
{    

	//cvZero(dst);
	Mat ms = Mat(src);
	Mat mt = Mat(dst);

    IplImage* hsv=cvCreateImage(cvGetSize(src),8,3);    
    //IplImage* cr=cvCreateImage(cvGetSize(src),8,1);     
    //IplImage* cb=cvCreateImage(cvGetSize(src),8,1);     
    cvCvtColor(src,hsv,CV_BGR2HSV);    
    //cvSplit(ycrcb,0,cr,cb,0);     
    
    static const int V=2;    
    static const int S=1;    
    static const int H=0;    
    
    //dst=cvCreateImage(cvGetSize(src),8,3);     
    cvZero(dst);    
    
    for (int h=0;h<src->height;h++) {    
		unsigned char* phsv=(unsigned char*)hsv->imageData+h*hsv->widthStep;    
        unsigned char* psrc=(unsigned char*)src->imageData+h*src->widthStep;    
        unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;    
        for (int w=0;w<src->width;w++) {    
			//cout << h << "  " << w << endl;
			Vec3b *colData = mt.ptr<Vec3b>(w);
            if (phsv[H]>=7&&phsv[H]<=29)    
            {    
                memcpy(pdst,psrc,3);    
				//colData[w][0] = 255;
				//colData[w][1] = 255;
				//colData[w][2] = 255;
            }

            phsv+=3;    
            psrc+=3;    
            pdst+=3;    
        }    
    }    
    //cvCopyImage(dst,_dst);     
    //cvReleaseImage(&dst);     
	
}    

void appFace::cvSkinSegment(IplImage* img, IplImage* mask){
	CvSize imageSize = cvSize(img->width, img->height);
	IplImage *imgY = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
	IplImage *imgCr = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
	IplImage *imgCb = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
	

	IplImage *imgYCrCb = cvCreateImage(imageSize, img->depth, img->nChannels);
	cvCvtColor(img,imgYCrCb,CV_BGR2YCrCb);
	cvSplit(imgYCrCb, imgY, imgCr, imgCb, 0);
	int y, cr, cb, l, x1, y1, value;
	unsigned char *pY, *pCr, *pCb, *pMask;
	
	pY = (unsigned char *)imgY->imageData;
	pCr = (unsigned char *)imgCr->imageData;
	pCb = (unsigned char *)imgCb->imageData;
	pMask = (unsigned char *)mask->imageData;
	cvSetZero(mask);
	l = img->height * img->width;
	for (int i = 0; i < l; i++){
		y  = *pY;
		cr = *pCr;
		cb = *pCb;
		cb -= 109;
		cr -= 152
			;
		x1 = (819*cr-614*cb)/32 + 51;
		y1 = (819*cr+614*cb)/32 + 77;
		x1 = x1*41/1024;
		y1 = y1*73/1024;
		value = x1*x1+y1*y1;
		if(y<100)	(*pMask)=(value<700) ? 255:0;
		else		(*pMask)=(value<850)? 255:0;
		pY++;
		pCr++;
		pCb++;
		pMask++;
	}
	cvReleaseImage(&imgY);
	cvReleaseImage(&imgCr);
	cvReleaseImage(&imgCb);
	cvReleaseImage(&imgYCrCb);
}

void appFace::cvSkinYUV(IplImage* src,IplImage* dst)    
{    
    IplImage* ycrcb=cvCreateImage(cvGetSize(src),8,3);    
    //IplImage* cr=cvCreateImage(cvGetSize(src),8,1);     
    //IplImage* cb=cvCreateImage(cvGetSize(src),8,1);     
    cvCvtColor(src,ycrcb,CV_BGR2YCrCb);    
    //cvSplit(ycrcb,0,cr,cb,0);     
    //namedWindow("input image");
    static const int Cb=2;    
    static const int Cr=1;    
    static const int Y=0;    
    
    //IplImage* dst=cvCreateImage(cvGetSize(src),8,3);     
    cvZero(dst);    
    
    for (int h=0;h<src->height;h++) {    
        unsigned char* pycrcb=(unsigned char*)ycrcb->imageData+h*ycrcb->widthStep;    
        unsigned char* psrc=(unsigned char*)src->imageData+h*src->widthStep;    
        unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;    
        for (int w=0;w<src->width;w++) {    
            if (pycrcb[Cr]>=133&&pycrcb[Cr]<=173&&pycrcb[Cb]>=77&&pycrcb[Cb]<=127)    
            {    
                    memcpy(pdst,psrc,3);    
            }    
            pycrcb+=3;    
            psrc+=3;    
            pdst+=3;    
        }    
    }    
    //cvCopyImage(dst,_dst);     
    //cvReleaseImage(&dst);     
}    

//捏脸。
//背景数据，要替换部分MASK，脸模板。简单的将脸下面移动上去，做好连接。
void appFace::changeFace(Mat _bgraFrameLight,uchar *mask_face_replace_data,Mat faceSample){
	//Vec4b *bgra_frame_data = _bgraFrame.ptr<Vec4b>(0);
	Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);

	int leftCol = this->calFirstColOfContour(faceSample);
	int rightCol = this->calLastColOfContour(faceSample);
	int topRow = this->calFirstRowOfContour(faceSample);
	int buttomRow = this->calLastRowOfContour(faceSample);

	//cout << "leftCol:" << leftCol <<" rightCol:"<< rightCol <<" topRow:"<< topRow <<" buttomRow:"<< buttomRow << endl;
	//从列开始数
	for(int _col = faceSampleRect.x;_col<faceSampleRect.x+faceSampleRect.width;_col++){
		int firstRow = 0,lastRow = 0;
		//从最后一行开始数
		for(int _row=faceSampleRect.y+faceSampleRect.height;_row>faceSampleRect.y;_row--){
			int index = _row*_col+_col;
			//如果当前点在mask_face_replace_data里
			if(mask_face_replace_data[index]>0){
				if(firstRow == 0) firstRow = _row;

				//如果当前点在faceSample里
				//这里还没有对faceSample定位
				//如果 点在脸矩形内
				//if(faceSampleRect.x < _col && faceSampleRect.x+faceSampleRect.width > _col && faceSampleRect.y < _row && faceSampleRect.y+faceSampleRect.height > _row )
				{
					//换算模板的点坐标值。
					int _row_ = _row - faceSampleRect.y;
					int _col_ = _col - faceSampleRect.x;

					Vec4b *colDataBGRA = faceSample.ptr<Vec4b>(_row_);

					//如果脸模板上这个点是不透明的，说明有值。
					if(colDataBGRA[_col_][3]>0){
						if(lastRow == 0) lastRow = _row;
						if(firstRow != lastRow) {

							//应该用拉申的办法。先把边缘处理干净。
							//这部分是多出来的，不要的。
							CvSize imageSize = cvSize(1, firstRow - lastRow);
							IplImage *imageSkin = cvCreateImage(imageSize, IPL_DEPTH_8U, 4);
							Mat mt = Mat(imageSkin);
							Vec4b *_d = mt.ptr<Vec4b>(0);
							for(int i=0;i<firstRow - lastRow;i++){
								_d[i] = bgra_frame_light_data[(firstRow+i)*_bgraFrameLight.cols+_col];
							}
							//cout << "第一个 mt.rows = " << mt.rows << " " << endl;
							resize(mt, mt, Size(1, 2*(firstRow - lastRow)));
							Vec4b *_d_ = mt.ptr<Vec4b>(0);
						
							//cout << "第二个 mt.rows = " << mt.rows << " " << endl;

							for(int j=0;j<2*(firstRow - lastRow);j++){
							//cout << "j = " << j << " " << endl;
								//bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col] = mt.data[j];
							
								if((_row+j)*_bgraFrameLight.cols+_col < _row_){
									cout << " 写到脸里面来了" << endl;
								}else {
									bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col][0] = _d_[j][0];
									bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col][1] = _d_[j][1];
									bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col][2] = _d_[j][2];
									bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col][3] = _d_[j][3];
								}
							}
						//扫描下一列
						//cout << "_col:" << _col << " _row:" << _row << "; lastRow:" << lastRow << "; firstRow:" << firstRow << endl;
						_row = 0;
						}
						
					}
				}
			}
		}
	}
		imwrite("MedianData//bgrTemp3.png", _bgraFrameLight);
		imwrite("MedianData//cartoonFilter//test.jpg", _bgraFrameLight);

}

/**
* frame 原图
* mouthModel 鼻子的模板图
* realSource
*/
void appFace::replaceNose(Mat _bgraFrameLight,Mat noseSample,Rect noseRect,Mat maskRealFace)
{
			
	//设定嘴的位置：居中，靠上。
	//Mat mouthSample;
//	mouthSample = mouthModel.
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *bgra_frame_data = _bgraFrameLight.ptr<Vec4b>(0);
	//Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);

	if(this->noseDetectedRect.x>0){

		//设定鼻子的区域。将模板移到中线为中心的位置，用mouthDectctedRect的y坐标
		//noseRect = Rect(this->faceMiddleRect.x - noseSample.cols/2,noseDetectedRect.y+noseDetectedRect.height-noseSample.rows,noseSample.cols,noseSample.rows);
		//加上嘴
		if(this->noseDetectedRect.x>0){
			for (int _row = 0; _row < noseSample.rows ; _row++)
			{
				Vec4b *colData = noseSample.ptr<Vec4b>(_row);

				for (int _col = 0 ; _col < noseSample.cols ; _col++){
					int r = noseRect.y + _row; 
					int c = noseRect.x + _col;
					int index = r*_bgraFrameLight.cols + c;//frame


					//-- Get valid area of nose model
					//为了解决白边问题，设定透明度为<250 的区域。鼻子不行，上边缘及两侧，需要渐变，与脸融合。
					if (colData[_col][3] == 0)
					{
						continue;
					}

					//鼻子，只取色相是不对了，还是黄。
					
					colData[_col][0] = superimposingTransparent(colData[_col][0],bgra_frame_data[index][0],colData[_col][3],255);
					colData[_col][1] = superimposingTransparent(colData[_col][1],bgra_frame_data[index][1],colData[_col][3],255);
					colData[_col][2] = superimposingTransparent(colData[_col][2],bgra_frame_data[index][2],colData[_col][3],255);
					//colData[_col][3] = 255;
					
					//-- Override face where mask > 0
					if(true){
						//变化明暗度
						Vec3b vf_hsv,vf_rgb,vb_hsv,vb_rgb;
						vf_rgb = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
						vb_rgb = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
						/*
						if(vb_rgb[0]<200) vb_rgb[0]=200;
						if(vb_rgb[1]<200) vb_rgb[1]=200;
						if(vb_rgb[2]<200) vb_rgb[2]=200;
						*/
						vf_hsv = kcvRGB2HSV(vf_rgb);
						vb_hsv = kcvRGB2HSV(vb_rgb);
						//去掉过于明显的数据
						//if(vb_hsv[2]<200)
						{
							vf_hsv[0] = vb_hsv[0];
							//vf_hsv[1] = vb_hsv[1];
							//vf_hsv[2] = vb_hsv[2];
						}
						vf_rgb = kcvHSV2RGB(vf_hsv);
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255; // mask_real_face_data
						//先改成不透明
						bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
						//bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2],bgra_frame_data[index][3]);
						//bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], 255);
						//bgra_frame_light_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
						//*/
						//bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
						//continue;
					}
				}
			}
		}
	}
			imwrite("MedianData//nose.png", noseSample);

}

/**
* frame 原图
* mouthModel 嘴的模板图
* realSource
*/
void appFace::replaceMouth(Mat _bgraFrameLight,Mat mouthSample,Rect mouthRect,Mat maskRealFace)
{
	
	//设定嘴的位置：居中，靠上。
	//Mat mouthSample;
//	mouthSample = mouthModel.
	//Rect mouthRect;
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *bgra_frame_data = _bgraFrameLight.ptr<Vec4b>(0);
	//Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);

	if(this->mouthDetectedRect.x>0){

		//加上嘴
		if(this->mouthDetectedRect.x>0){
			for (int _row = 0; _row < mouthSample.rows ; _row++)
			{
				Vec4b *colData = mouthSample.ptr<Vec4b>(_row);
				for (int _col = 0 ; _col < mouthSample.cols ; _col++){
					//-- Get valid area of face model
					if (colData[_col][3] < 250)
					{
						continue;
					}
					int r = mouthRect.y + _row; 
					int c = mouthRect.x + _col;
					int index = r*_bgraFrameLight.cols + c;//frame

					//-- Override face where mask > 0
					if(true){
						//变化明暗度
						Vec3b vf_hsv,vf_rgb,vb_hsv,vb_rgb;
						vf_rgb = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
						vb_rgb = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
						/*
						if(vb_rgb[0]<200) vb_rgb[0]=200;
						if(vb_rgb[1]<200) vb_rgb[1]=200;
						if(vb_rgb[2]<200) vb_rgb[2]=200;
						*/
						vf_hsv = kcvRGB2HSV(vf_rgb);
						vb_hsv = kcvRGB2HSV(vb_rgb);
						//去掉过于明显的数据
						//if(vb_hsv[2]<200)
						{
							vf_hsv[0] = vb_hsv[0];
							//vf_hsv[1] = vb_hsv[1];
							//vf_hsv[2] = vb_hsv[2];
						}
						vf_rgb = kcvHSV2RGB(vf_hsv);
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255; // mask_real_face_data
						//先改成不透明
						bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2],(1-rate)*mask_real_face_data[index]);
						//bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2],255);
						//bgra_frame_light_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
						//*/
						//bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
						//continue;
					}
				}
			}
		}
	}
}

//void appFace::replaceEyes(Mat _bgraFrameLight,Mat leftEye, Mat rightEye,Mat leftEyeWithBrowModel,Mat rightEyeWithBrowModel, Mat &resultImage,Mat leftEyePupilModel,Mat rightEyePupilModel,Mat grayData){}

int appFace::superimposingTransparent(int c1,int c2,int a1,int a2){//Vec4b colData,Vec4b bgra_frame_data,int _transparent){
	double c12;	
//int c1,c2,a1,a2;
	//a1 = _transparent;
	//a2 = bgra_frame_data[3];
	
	//for(int i = 0;i<3;i++){
	//	c1 = colData[i];
	//	c2 = bgra_frame_data[i];
	//	c12 = (c1*a1*(1-a2/255)+c2*a2)/(a1+a2-a1*a2);
	c12 = (double)c2*(1-((double)a1/255))+(double)c1*((double)a1/255);
	if( c12>255){
		cout << c1 <<" " << a1 << " " << c2 << " " << a2 << " " << c12 << " " << endl;
		c12 = (double)c2*(1-(double)a1/255)+(double)c1*(double)a1/255;
	}
	//	colData[i] = c12;
	//}
	
	//colData[3] = 2;
	//colData[3] = _transparent;
		return c12;
}

void appFace::generateThresholdImage(IplImage *graySource,IplImage *grayDst,double max,int method, int type, double* parameters){

}



Mat appFace::createROI(Mat m,string name,int pre,int mode,int range){ // m源图。pre，预处理：0，无；1，直方图。mode：1，cvThreshold；2，cvAdaptiveThreshold；3，Canny边缘检测。
		//抠出目标
		Mat _gray;
		Mat mROI;
		//先调整原图平均亮度
		IplImage* _imageBgr = &IplImage(m);
		Mat _tar ;
		m.copyTo(_tar);
		IplImage* _imageTar = &IplImage(_tar);
		set_avg_gray(_imageTar,_imageTar,(double)128.0);//亮度平衡处理
		imwrite("MedianData//"+name+"_128Light.jpg", _tar);
		
		//cvSmooth(_imageBgr,_imageTar);
		cvSmooth(_imageTar,_imageTar);
		cvtColor( _tar, _gray, CV_BGR2GRAY );
		Mat mt2 = Mat(_gray);
		//eye_gray_r = removeNoise(eye_gray_r,3);//高斯去噪
		IplImage* eye_gray_r = &IplImage(_gray);
		//imwrite("MedianData//"+name+"_le1.jpg", mt2);

		int offSet = 0 ;
		if(m.cols > m.rows)
			offSet = ((float)(m.rows));
		else
			offSet = ((float)(m.cols));
		//cout << "m.rows:" << m.rows << " - m.cols:" << m.cols <<  " . offSet: " << offSet << endl;

		if(pre == 1){
			//眼睛直方图后，效果不好。因为眼睛的黑白已经很明显了。鼻子和嘴就需要.
			equalizeHist( _gray, _gray );
		//imwrite("MedianData//le2.jpg", mt2);
		}

		if(mode == 3){
			//****************************************************************************
			//这个值是懵上的。可能原因是，亮度调到128，对眼睛来说，眼线更为突出了。
			Mat edge;
			Canny(_gray, edge, 50, 150);
			//imwrite("MedianData//"+name+"_leedge.jpg", edge);
			//****************************************************************************
			reverseROI(_gray); // 转成黑底白字
		}

		IplImage *graySource = &IplImage(_gray);
		CvSize imageSize1 = cvSize(_gray.cols, _gray.rows);
		IplImage *grayDst = cvCreateImage(imageSize1, IPL_DEPTH_8U, 1);

		if(mode == 1){ // 确定阈值
			cvThreshold(graySource, grayDst, range, 255,CV_THRESH_BINARY);
			reverseROI(grayDst); // 转成黑底白字
		}

		if(mode == 2 || mode == 3){ // 自适应阈值
			range =  m.rows/range; // 范围为宽的1/range
			if(range % 2 == 1 && range > 1){}else range++;
			if(mode == 2) // 中值算法
				cvAdaptiveThreshold(graySource, grayDst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, range, range);

			if(mode == 3) // 高斯算法
				cvAdaptiveThreshold(graySource, grayDst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, range, range);
			
			reverseROI(grayDst); // 转成黑底白字
			filterBlock(grayDst,5,5,255); // 过滤大小需要重新设计
		}

		Mat mt = Mat(grayDst);
		imwrite("MedianData//"+name+"_le2.jpg", mt);
		//imwrite("MedianData//"+name+"_le.jpg", mt2);
		return mt;
}

void appFace::filterBlock(Mat frame,int w,int h,int v){ // 过滤掉小于(w,h)的独立块。stack overflow!!
	//这段下面都写错了。应该用生长法。
	//找到一个点，然后四面生长，扩大矩形，找出连通块，如果这个块小于指定矩形，删除；否则记录这个块，清除，寻找下一个，最后再把这个块复制回去。
	//定义两个一样大的Mat，一个是返回值，另一个做临时变量。
	//cout << " w: " << w << " " << h << endl;
	Mat src,tar;
	int b=0,bt=50,f=255,ft=250;
	//imwrite( "MedianData//frameClear.png",frame);
	frame.copyTo(src);
	//imwrite( "MedianData//srcClear.png",src);
	tar = frame.clone();frame.setTo(Scalar(b));
	//从原图中查找，如果找到有值点，就生长，直到所有连通点都找到，将这个块放到临时块里，清除原图里的这个块。判断这个块大小，如果超出指定范围，复制到返回值Mat里。
	for(int _row=0;_row<src.rows;_row++){
		for(int _col=0;_col<src.cols;_col++){
			int index = _row * src.cols + _col;
			if(src.data[index] ) { //该点有值。
				//cout << " col: " << _col << " " << _row << endl;
				tar.setTo(Scalar(b)); // 设成背景色。
				nineBox(src,tar,_col,_row,true);//生长法：nineBox
				//Mat reverse; tar.copyTo(reverse);reverse.setTo(Scalar(b));
				imwrite("MedianData//filterBlockReverse.png",tar);
				Rect r = getROIRect(tar);
				//reverseROI(tar);
				//旋转SRC，使它最贴合宽高最大比例
				int width = r.width;//this->calLastColOfContour(reverse) - this->calFirstColOfContour(reverse);
				int height = r.height;//this->calLastRowOfContour(reverse) - this->calFirstRowOfContour(reverse);
				//cout <<"width:"<< width << " height:" << height << " x:" << r.x << " y:" << r.y << " w:" << w << " h:" << h << endl;
				if(width >= w && height >= h) {
					copyROI(tar,frame); // 如果连接的一个块大于指定宽高，就复制回去。
					imwrite("MedianData//frameBlock.png",frame);
				}
			}
		}
	}
	//reverseROI(frame);
	return;
}

//根据MASK，做联通过滤。如果frame中的一个点在MASK内，那么它在MASK之外的联通域也保留；(在MASK内的，做联通处理)。否则删除。
void appFace::filterBlock(Mat frame,Mat mask,bool blackBackground){
	Mat ff,t,ret;
	if(!blackBackground){reverseROI(frame);reverseROI(mask);	}
	int b=0,f=255;
	frame.copyTo(ff);frame.copyTo(t);frame.copyTo(ret);ret.setTo(Scalar(b));// 这里全部都复制出来，避免操作对原图影响。
	for(int _row = 0;_row < ff.rows;_row++){
		for(int _col = 0;_col<ff.cols;_col++){
			if(ff.at<uchar>(_row,_col)){ //f.data[_row*f.cols+_col]<50){ // 如果这个点有值
				t.setTo(Scalar(b));
				nineBox(ff,t,_col,_row,blackBackground);

				if(joinMask(t,mask,true)) {//因为copyROI要求是黑底白字，所以需要反转一下。设为白背景
					copyROI(t,ret);
					imwrite("MedianData//return.png",ret);
				}
				else{
					cout << " remove " << endl;
				}
			}
		}
	}
	if(!blackBackground){reverseROI(ret);reverseROI(mask);	}
	ret.copyTo(frame);
	//frame = ret;
}

// 判断t中的点是否有在mask范围内的。
bool appFace::joinMask(Mat t,Mat mask,bool blackBackground){ 
	int f=255,ft=200,b=0,bt=50;
	if(!blackBackground){ // 先转成白底黑字
		reverseROI(t);reverseROI(mask);
	}
	Rect r = copyROI(t,t);
	Mat _tt = t(r);
	Mat _maskt = mask(r);
	for(int _row=0;_row<r.height;_row++){
		for(int _col=0;_col<r.width;_col++){
			int index = _row*r.width+_col;
			if(_tt.at<uchar>(_row,_col)){//_tt.data[index] < 50 ) {
				//cout << _tt.at<uchar>(_row,_col) << " , " << _row << " , " << _col << " , " << _maskt.at<uchar>(_row,_col) << endl;
				if( _maskt.at<uchar>(_row,_col)){//.data[index] < 50){
					//cout << _tt.at<uchar>(_row,_col) << " . " << _row << " . " << _col << " . " << _maskt.at<uchar>(_row,_col) << endl;
					//rectangle(t, r, Scalar(255,0,0));
					//imwrite("MedianData//block.png",_tt);
					//imwrite("MedianData//block1.png",_maskt);
					
					if(!blackBackground){ // 先转成白底黑字
						reverseROI(t);reverseROI(mask);
					}
					
					return true;
				}
			}
		}
	}
	if(!blackBackground){ // 转成白底黑字了，再给转回去。
		reverseROI(t);reverseROI(mask);
	}	
	return false;
}

int appFace::getROIWidth(Mat _roi,int mode){ // mode:0，是黑底，1是白底

}
int appFace::getROIHeight(Mat _roi,int mode){
}

Mat appFace::nineBox(Mat m,Mat t,int x,int y,bool blackBackground){
	int b=0,f=255;

	int left,right,top,buttom;
	if(x-1>=0) left = x-1;else left = x;
	if(x+1<m.cols) right = x+1;else right = x;
	if(y-1>=0) top = y-1;else top = y;
	if(y+1<m.rows) buttom = y+1;else buttom = y;
	Rect r=Rect(left,top,right-left+1,buttom-top+1);

	Mat submat = m(r);Mat _submat;submat.copyTo(_submat);
	if(!t.cols){	m.copyTo(t);	t.setTo(Scalar(b));	}
	Mat submatt = t(r);
	submatt = submatt + _submat;//写目标
	submat.setTo(Scalar(b));//清空源

	for(int _r=0;_r<r.height;_r++){//判断递归
		for(int _c=0;_c<r.width;_c++){
			if(_submat.at<uchar>(_r,_c)){
				int xt = left + _c;
				int yt = top + _r;
				nineBox(m,t,xt,yt);
			}
		}
	}
	return m;
}

//取SRC中指定点（startX,startY）的联通MASK，并将其移出至TAR中。要求，起始时，指定点有前景值；图像为白底黑字。
Mat appFace::nineBox(Mat frameSrc,Mat frameTar,int startX,int startY){
	return nineBox(frameSrc,frameTar,startX,startY,true);
}

Rect appFace::getROIRect(Mat src){
	int left=src.cols,top=src.rows,right=0,buttom=0;
	// TODO: 改用矩阵加减法
	for(int _row=0;_row<src.rows;_row++){
		for(int _col=0;_col<src.cols;_col++){
			int index = _row * src.cols + _col;
			if(src.data[index] ) {
				if(left > _col) left = _col;
				if(right < _col) right = _col;
				if(top > _row) top = _row;
				if(buttom < _row) buttom = _row;
			}
		}
	}
	Rect ret = Rect(left,top,right-left,buttom-top);
	return ret;
}

Rect appFace::copyROI(Mat src,Mat tar){//将src里大于0的点，复制到tar里.
	if(!tar.cols){
		src.copyTo(tar);
	}
	int left=src.cols,top=src.rows,right=0,buttom=0;
	// TODO: 改用矩阵加减法
	for(int _row=0;_row<src.rows;_row++){
		for(int _col=0;_col<src.cols;_col++){
			int index = _row * src.cols + _col;
			if(src.data[index] ) {
				if(left > _col) left = _col;
				if(right < _col) right = _col;
				if(top > _row) top = _row;
				if(buttom < _row) buttom = _row;
				//tar.data[index] = 0;
			}
		}
	}
	tar = tar + src;
	Rect ret = Rect(left,top,right-left,buttom-top);
	return ret;
}
void appFace::absROI(Mat roi){
	for(int _row=0;_row<roi.rows;_row++){
		for(int _col=0;_col<roi.cols;_col++){
			int index = _row * roi.cols + _col;
			if(roi.data[index] < 50) 
				roi.data[index] = 0;
			else
				roi.data[index] = 255;
		}
	}

}
void appFace::reverseROI(Mat roi){
	for(int _row=0;_row<roi.rows;_row++){
		for(int _col=0;_col<roi.cols;_col++){
			int index = _row * roi.cols + _col;
			if(roi.data[index] < 50) 
				roi.data[index] = 255;
			else
				roi.data[index] = 0;
		}
	}
}

//  返回两个四个点的数组，每组上下左右，第一组左眼，第二组右眼。不能确定边缘的点，为（0，0）.
Point* appFace::getEyePoint(Mat leftEyeROI_lt,Mat leftEyeROI_rt,Mat rightEyeROI_rt,Mat rightEyeROI_lt){ 
	Point eyes[8] ;
	int llx,lly,lrx,lry,ltx,lty,lbx,lby,rlx,rly,rrx,rry,rtx,rty,rbx,rby;
	Rect rectl = Rect(0,leftEyeROI_lt.rows/4,leftEyeROI_lt.cols,leftEyeROI_lt.rows/2);
	Rect rectr = Rect(0,leftEyeROI_rt.rows/4,leftEyeROI_rt.cols,leftEyeROI_rt.rows/2);

	Mat matLl = leftEyeROI_lt(rectl);
	Mat matLr = leftEyeROI_rt(rectl);
	Mat matRl = rightEyeROI_lt(rectr);
	Mat matRr = rightEyeROI_rt(rectr);

	Mat leftEyeROI_l;
	leftEyeROI_lt.copyTo(leftEyeROI_l);
	removeBrow(leftEyeROI_lt,rectl);
	leftEyeROI_l = leftEyeROI_l - leftEyeROI_lt;
	//imwrite("MedianData//leftEyeROI_l.png",leftEyeROI_l);

	Mat leftEyeROI_r;
	leftEyeROI_rt.copyTo(leftEyeROI_r);
	removeBrow(leftEyeROI_rt,rectl);
	leftEyeROI_r = leftEyeROI_r - leftEyeROI_rt;

	Mat rightEyeROI_r;
	rightEyeROI_lt.copyTo(rightEyeROI_r);
	removeBrow(rightEyeROI_lt,rectr);
	rightEyeROI_r = rightEyeROI_r - rightEyeROI_lt;

	Mat rightEyeROI_l;
	rightEyeROI_rt.copyTo(rightEyeROI_l);
	removeBrow(rightEyeROI_rt,rectr);
	rightEyeROI_l = rightEyeROI_l - rightEyeROI_rt;

	llx = this->calFirstColOfContour(matLl);
	lly = this->calFirstColOfContour_Row(matLl);
	Point llp = Point(llx,lly);eyes[2] = llp;
	lrx = this->calLastColOfContour(matLr);
	lry = this->calLastColOfContour_Row(matLr);
	Point lrp = Point(lrx,lry);eyes[3] = lrp;
	ltx = this->calFirstRowOfContour(matLr);
	lty = this->calFirstColOfContour_Row(matLr);
	Point ltp = Point(ltx,lty);eyes[0] = ltp;

	rlx = this->calFirstColOfContour(matRl);
	rly = this->calFirstColOfContour_Row(matRl);
	Point rlp = Point(rlx,rly);eyes[6] = rlp;
	rrx = this->calLastColOfContour(matRr);
	rry = this->calLastColOfContour_Row(matRr);
	Point rrp = Point(rrx,rry);eyes[7] = rrp;
	rtx = this->calFirstRowOfContour(matRl);
	rty = this->calLastColOfContour_Row(matRl);
	Point rtp = Point(rtx,rty);eyes[4] = rtp;

	//逻辑判断：如果

	return eyes;
}

Point* appFace::getNosePoint(Mat noseROI){ //  返回两个点的数组，左右。不能确定边缘的点，为（0，0）.
	Point  nose[2];
	return nose;
}

Point* appFace::getMouthPoint(Mat mouthROI){ //  返回四个点的数组，上下左右。不能确定边缘的点，为（0，0）.
	Point  mouth[4];
	return mouth;
}

void appFace::removeBrow(Mat eyeMask,Rect r){ // 设定眼睛ROI的上1/6,一定会包括眉毛或不包括眼睛。
	// 清除上1/6区域中的一切联通块
	int b=0,f=255;
	Mat tar;
	eyeMask.copyTo(tar);

	//Mat rt; eyeMask(r).copyTo(rt);Mat rtt;rt.copyTo(rtt);rtt.setTo(Scalar(b));
	Mat rt = eyeMask(r);//Mat rtt;rt.copyTo(rtt);rtt.setTo(Scalar(b));
	imwrite("MedianData//Bfremoved.png",rt);
	//imwrite("MedianData//Bfremoved1.png",rtt);
	for(int _row=r.y;_row<r.y+r.height;_row++){
		for(int _col=r.x;_col<r.x+r.width;_col++){
			uchar ut = rt.at<uchar>(_row,_col);
			if(rt.at<uchar>(_row,_col)){
				//cout << _col << " " << _row << endl;
				tar.setTo(Scalar(b));
				nineBox(eyeMask,tar,_col,_row,true);
				imwrite("MedianData//removed1.png",tar);
			}
		}
	}
	imwrite("MedianData//removed.png",eyeMask);

}


void appFace::debugFace(){
	//rectangle(debugFrame, faceDetectedRect, Scalar(0,0,255));
		//-- Draw rectangles
	rectangle(debugFrame, faceDetectedRect, Scalar(255,0,0));
	rectangle(debugFrame, faceDetectedRect, Scalar(0,255,0));
		
	Rect _rect_f = Rect(faceDetectedRect.x, faceDetectedRect.y, faceDetectedRect.width/2, faceDetectedRect.height);
	rectangle(debugFrame, faceDetectedRect, Scalar(0,0,255));
	rectangle(debugFrame, _rect_f, Scalar(0,0,255));

	imwrite("MedianData//debug.png",debugFrame);
}

//取眼睛模板的左、右点。lr:0取左眼白点，1取右眼白点，2取左眼边缘，3取右眼边缘，4取眼白上边缘，5取眼上边缘。
Point appFace::getEyeModelPoint(Mat model,int lr){
	if(lr==0)
	for(int _col=0;_col<model.cols;_col++){
		for(int _row=0;_row<model.rows;_row++){
			Vec4b pd = model.at<Vec4b>(_row,_col);
			if(pd[3]>100 && pd[3]<160){
				Point ret = Point(_col,_row);
				return ret;
			}
		}
	}
	if(lr==1)
	for(int _col=model.cols;_col>0;_col--){
		for(int _row=0;_row<model.rows;_row++){
			Vec4b pd = model.at<Vec4b>(_row,_col);
			if(pd[3]>100 && pd[3]<160){
				Point ret = Point(_col,_row);
				return ret;
			}
		}
	}
	if(lr==2)
	for(int _col=0;_col<model.cols;_col++){
		for(int _row=0;_row<model.rows;_row++){
			Vec4b pd = model.at<Vec4b>(_row,_col);
			if(pd[3]>250){
				Point ret = Point(_col,_row);
				return ret;
			}
		}
	}
	if(lr==3)
	for(int _col=model.cols;_col>0;_col--){
		for(int _row=0;_row<model.rows;_row++){
			Vec4b pd = model.at<Vec4b>(_row,_col);
			if(pd[3]>250){
				Point ret = Point(_col,_row);
				return ret;
			}
		}
	}
	if(lr==4)
	for(int _row=0;_row<model.rows;_row++){
		for(int _col=model.cols;_col>0;_col--){
			Vec4b pd = model.at<Vec4b>(_row,_col);
			if(pd[3]>100 && pd[3]<160){
				Point ret = Point(_col,_row);
				return ret;
			}
		}
	}
	if(lr==5)
	for(int _row=0;_row<model.rows;_row++){
		for(int _col=0;_col<model.cols;_col++){
			Vec4b pd = model.at<Vec4b>(_row,_col);
			if(pd[3]>250){
				Point ret = Point(_col,_row);
				return ret;
			}
		}
	}
	Point ret = Point(0,0);
	return ret;
}


//将模板按指定三点进行旋转和缩放。
void appFace::resizeModel(Mat modelImg,Point ml,Point mr,Point mt,Point el,Point er,Point et){
	double arcS;
	int xmlength = mr.x - ml.x; // TODL: mr.x值不对
	int ymlength = mr.y - ml.y; // 模型左右点Y值差
	int xelength = er.x - el.x;
	int yelength = er.y - el.y;

	//* 缩放处理方案一：没有考虑眼睛顶点，直接左右上下缩放，眼睛会被上下放大。
	bool yReverse = false;
	if((ymlength > 0 && yelength < 0) || (ymlength < 0 && yelength > 0)) yReverse = true;
	cout << xelength << " " << xmlength << " " << yelength << " " << ymlength << endl;
	resize(modelImg,modelImg,Size(0,0),xelength/xmlength,yelength/ymlength);
	if(yReverse) flip(modelImg,modelImg,0);
	//*/

	/*//*缩放处理方案二：旋转，再缩放。
	// TODO: 先旋模板转到水平，再缩放，再旋转到眼睛角度，再上下缩放，到模板顶点与眼睛顶点相同高度，这时的眼睛模板上下高度是一个近似值。
	arcS = atan2((double)modelImg.rows,(double)modelImg.cols);
	arcS = arcS*180/CV_PI; // 计算模型对角线原始角度
	
	//旋转到水平

	//横向缩放，到眼睛的左右点长度。

	//*/
	cout << arcS << endl;
	cout << "" << endl;
}

void appFace::setHeadModel(string headName,string expressionName){

}

void appFace::debugEyes(){
	for(int i=0;i<this ->eyeNumber;i++){
		rectangle(debugFrame, eyeDetectedRects[i], Scalar(0,255,0));
	}
	imwrite("MedianData//debug.png",debugFrame);

}
void appFace::debugNose(){
	rectangle(debugFrame, this->noseDetectedRect, Scalar(0,0,0));
	imwrite("MedianData//debug.png",debugFrame);
}
void appFace::debugMouth(){
	rectangle(debugFrame, this->mouthDetectedRect, Scalar(255,0,255));
	imwrite("MedianData//debug.png",debugFrame);
}

