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
	//��ԵС��ȫ����1/8�������ǽ��ŵġ�
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
	
	cout << "������mask:" << wt << "  M��1/15= "<< w<<";  ������(y,x)��"<<_fl<<","<<_flr<<"   ������ȣ�"<<faceMiddleRect.x - _fl <<"   MASK��ȣ�"<<imageFaceContourSM.cols<< endl;

	if(wt<w && wt>0)
		return faceMiddleRect.x - _fl;
	else
		return -1;
}

int appFace::getRightFaceWidth(){
	int _fl = calLastColOfContour(imageFaceContourSM);
	int _flr = calLastColOfContour_Row(imageFaceContourSM);
	//��ԵС��ȫ����1/8�������ǽ��ŵġ�
	int w = (calLastColOfContour(imageFaceContourSM)-calFirstColOfContour(imageFaceContourSM))/8;
	int wt = 0;
	//uchar* rowData = imageContourSM.ptr<uchar>(_flr);
	for(int i=_fl; i<imageContourSM.cols; i++){
		//TODO û�������ߵ����
		int index = _flr*imageContourSM.cols + i;
		if(imageContourSM.data[index]>5)
			wt++;
		else
			break;
	}
	cout << "������mask:" <<wt << "  M��1/15= "<< w << ";  ������(y,x)��"<<_fl<<","<<_flr<<"   ������ȣ�"<< _fl - faceMiddleRect.x  <<"   MASK��ȣ�"<<imageFaceContourSM.cols<< endl;

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

int appFace::calFirstRowOfContourHuman(){//��дΪ�����·���
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

vector<Rect> appFace::detectEyes(Mat _face){//�����м���۾�
		//��ʼ����۾���
	Mat faceROI = _face;//frame_gray( faceDetectedRect );
	std::vector<Rect> eyes;
	int minSize = faceROI.rows / 5;
	eyes_cascade.detectMultiScale( faceROI, eyes, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));
		
	if(eyes.size()>2){	
		eyes.erase(eyes.begin()+2,eyes.end());//ɾ������[2,��β];�����0��ʼ
	}
	else if(eyes.size()>0)
		return eyes;
	else
		cout << " û�м�⵽�۾��� " << endl;
	return eyes;
}

Mat appFace::createROI(Mat m,string name){
		//�ٳ�Ŀ��
		Mat _gray;
		Mat mROI;
		//leftEyeROI.convertTo(leftEyeROI, leftEyeROI.type(), 1, 0); // ROI��ͼ��֮��ĸ���
		
		//�ȵ���ԭͼƽ������
		IplImage* _imageBgr = &IplImage(m);
		Mat _tar ;
		m.copyTo(_tar);
		IplImage* _imageTar = &IplImage(_tar);
		set_avg_gray(_imageTar,_imageTar,(double)128.0);//����ƽ�⴦��
		imwrite("MedianData//"+name+"_128Light.jpg", _tar);
		
		//cvSmooth(_imageBgr,_imageTar);
		cvSmooth(_imageTar,_imageTar);

		imwrite("MedianData//"+name+"_le0.jpg", _tar);
		cvtColor( _tar, _gray, CV_BGR2GRAY );
		Mat mt2 = Mat(_gray);
		//��˹ȥ��
		//eye_gray_r = removeNoise(eye_gray_r,3);
		IplImage* eye_gray_r = &IplImage(_gray);
		imwrite("MedianData//"+name+"_le1.jpg", mt2);

		//ֱ��ͼ��Ч�����á�
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
		//70�Ǹ����Ե�ֵ����֪��Ϊʲô����ʱʹ���š�
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
		mask1 = createROI(roi,"eyeDetectedRects[0]1",1,2,3); // ��ֵ�㷨 1/3�뾶
		mask6 = createROI(roi,"eyeDetectedRects[0]6",0,2,3); // ��ֵ�㷨 1/3�뾶
		mask7 = createROI(roi,"eyeDetectedRects[0]7",0,2,roi.rows/5); // ��ֵ�㷨 5�뾶
		//=======================================================================
		Rect rBrow = Rect(0,0,mask1.cols,mask1.rows/4);
		imwrite("MedianData//eyeDetectedRects[0]10.png",mask1);
		removeBrow(mask1,rBrow);//MASKȥ��üë����
		filterBlock(mask7,mask1,true); // ����ϸ��MASKͼ
		filterBlock(mask6,mask1,true); // ���˴���MASKͼ
		imwrite("MedianData//eyeDetectedRects[0]71.png",mask7);

		Mat mask71,mask11,mask61;
		imageOrigine(eyeDetectedRects[1]).copyTo(roi);
		//=======================================================================
		mask11 = createROI(roi,"eyeDetectedRects[1]1",1,2,3); // ��ֵ�㷨 1/3�뾶
		mask61 = createROI(roi,"eyeDetectedRects[1]6",0,2,3); // ��ֵ�㷨 1/3�뾶
		mask71 = createROI(roi,"eyeDetectedRects[1]7",0,2,roi.rows/5); // ��ֵ�㷨 5�뾶
		//=======================================================================
		rBrow = Rect(0,0,mask11.cols,mask11.rows/4);
		imwrite("MedianData//eyeDetectedRects[1]10.png",mask1);
		removeBrow(mask11,rBrow);//MASKȥ��üë����
		filterBlock(mask71,mask11,true); // ����ϸ��MASKͼ
		filterBlock(mask61,mask11,true); // ���˴���MASKͼ
		imwrite("MedianData//eyeDetectedRects[1]71.png",mask71);

		eyesPoint = getEyePoint(mask6,mask7,mask61,mask71); // ���������۵�ROI������㡣

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

	//******************************* ��֤���Ƿ���ȷ ******************************************************
	int x1 = int(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2); //middlePointX(eyeDetectedRects[0]); // ˫������XY����
	int y1 = int(eyeDetectedRects[0].y+eyeDetectedRects[0].height/2);//middlePointY(eyeDetectedRects[0]);
	int x2 = int(eyeDetectedRects[1].x+eyeDetectedRects[1].width/2);//middlePointX(eyeDetectedRects[1]);
	int y2 = int(eyeDetectedRects[1].y+eyeDetectedRects[1].height/2);//middlePointY(eyeDetectedRects[1]);
	int _y = 0;int _ym=-1;
	for( size_t mi = 0; mi < mouths.size(); mi++ )
	{
		//����첻�����������غϣ���������֮�䣬�������졣
		int mx = int(faceDetectedRect.x+mouths[mi].x+mouths[mi].width/2);
		int my = int(faceDetectedRect.y+mouths[mi].y+mouths[mi].height/2);
		cout << mx << ": " << x1 << ": " << x2<< "|| my :"<<my<<"  "<<y1 << "  "<< y2 << endl;
		bool _betwin = false;
		if((mx>=x1 && mx<=x2) || mx<= x1 && mx >=x2) _betwin = true;
		if(_betwin && my>y1 && my>y2) 
		{
			//���������һ�������졣
			if(mouths[mi].y>_y){
				_y = mouths[mi].y;
				_ym = mi;
			}
		}
	}
	//****************************************************************************************************************
	//�ҵ���һ����
	if(_ym>=0){
		this->mouthDetectedRect = Rect(this->faceDetectedRect.x+mouths[_ym].x,this->faceDetectedRect.y+mouths[_ym].y,mouths[_ym].width,mouths[_ym].height);
	} else {
		cout << " û���ҵ���ȷ���� " << endl;
	}
}
void appFace::setMouthsParameter(Vector<Rect> mouths,Rect mouthRegion){
	//��Ϊ����ָ��������ҵģ�����Ҫ�����۾��ж��ˣ�ֻȡ�������һ����������ĵ�һ����
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

	//****************************  �������Ƿ���ȷ  *************************************************************
	int x1 = int(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2); //middlePointX(eyeDetectedRects[0]); // ˫������XY����
	int y1 = int(eyeDetectedRects[0].y+eyeDetectedRects[0].height/2);//middlePointY(eyeDetectedRects[0]);
	int x2 = int(eyeDetectedRects[1].x+eyeDetectedRects[1].width/2);//middlePointX(eyeDetectedRects[1]);
	int y2 = int(eyeDetectedRects[1].y+eyeDetectedRects[1].height/2);//middlePointY(eyeDetectedRects[1]);
	int _y = 0;int _ym=-1;
	int x3 = int(mouthDetectedRect.x+mouthDetectedRect.width/2);//middlePointX(mouthDetectedRect); // ������X����
	int y3 = int(mouthDetectedRect.y+mouthDetectedRect.height/2);//middlePointY(mouthDetectedRect); // ������Y����

	_y = 0; int _yn = -1;
	for( size_t ni = 0; ni < noses.size(); ni++ )//��⵽һ�����ӣ����Ǳ�������������˵��ˡ�
	{
		int nx = int(faceDetectedRect.x+noses[ni].x+noses[ni].width/2); // ��������X����
		int ny = int(faceDetectedRect.y+noses[ni].y+noses[ni].height/2); // ��������Y����
		cout << nx << ": " << x1 << ": " << x2<< " " << x3 << "|| ny :"<<ny<<"  "<<y1 << "  "<< y2 << " " << y3 << endl;
		bool _betwin = false;
		if((nx>=x1 && nx<=x2) || (nx<= x1 && nx >=x2)) _betwin = true;

		if(_betwin && ny>y1 && ny>y2 && ny<y3) 
		{
			//�������棬���۾����棬������֮�䣬�����Ǳ��ӡ�

			//ȡ��������Ǹ�
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
		cout << " û���ҵ����ӡ� " << endl;
	}
}
void appFace::setNoseParameter(Vector<Rect> noses,Rect noseRegion){
	//��Ϊ����ָ��������ҵģ�����Ҫ�����۾��ж��ˣ�ֻȡ������ĵ�һ��,�������һ����
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
		cout << " û���ҵ����ӡ� " << endl;
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

vector<Rect> appFace::detectEyes(Mat _face,Mat frame){//�����м���۾�
		//��ʼ����۾���
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
		cout << " û�м�⵽�۾��� " << endl;
	return eyes;
}

vector<Rect> appFace::detectMouth(Mat faceROI){//�����м����
		//����������
		//Mat faceROI = _face;//frame_gray( faceDetectedRect );
		std::vector<Rect> mouths;
		int minSize1 = faceROI.rows*1/6;
		mouth_cascade.detectMultiScale( faceROI, mouths, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize1*3, minSize1));

		return mouths;
}

vector<Rect> appFace::detectNose(Mat _face){//�����м�����

		//�����������
		Mat faceROI = _face;//frame_gray( faceDetectedRect );
		std::vector<Rect> noses;
		//�������ֵӰ��޴�
		int minSize2 = faceROI.rows / 8;
		nose_cascade.detectMultiScale( faceROI, noses, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize2, minSize2));
		return noses;
}

int appFace::leftFace(Mat img){//�ж�ͼ�е������ĸ�����ƫ��1����2���ң�0û�м�⵽����-1ֻ��һֻ��
	// �ж����Ĳ�ƫ
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
			if (_eyes.size() == 2){  //�����⵽��2ֻ�۾�
				//���ж�������
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


int appFace::rotateDetectFaces() { //��ת����������û�з���0������У�����������
	Mat frame;
	imageOrigine.copyTo(frame);
	std::vector<Rect> _faces;
    double angle = 1;  // ÿ����ת�Ƕ�  
	double _angle,_lastTestAngle;
	int _eyes_y_dif = 0;//��ֻ�۾���Y�����
	int _faceDirectTemp = leftFace(frame); // �ж������ĸ�����ƫ��


	int direct = 2;//������ת����Ĭ��Ϊ����ת��
	if(_faceDirectTemp > 0 )
		direct = _faceDirectTemp ;
	else
		return -1; // TODO:  û�д���һֻ�۵����
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
		cout << " ��ת�����¼����һ�Ρ�����" << endl;
		face_cascade.detectMultiScale( frame_gray, _faces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		Rect maskFaceRect;
		if(_faces.size() <1){
			cout << " û�м�⵽���� " << i_angle * angle << endl;
			imwrite("MedianData//simpleFaceDetectionc.png",frame_gray);
			/*
			//�ÿ�ͼ��������ʵ�֡�
			if(imageFaceContourSM.cols){ // �Ѿ��������Ŀ�ͼ
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
			if (_eyes.size() == 2){  //�����⵽��2ֻ�۾�
				//���ж�������
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
				//�����ֻ�۾���ƽ�ģ����ĵ�Y������ȣ�������תһ�Σ���ͷ�ˣ�������϶ȴε�С��������εġ�
				int _eyes_y_dif1 = leftEyeMiddleY - rightEyeMiddleY;
					
				cout<< " ��⵽2ֻ�۾�  ���ĵ�Y���겻��� " << leftEyeMiddleY << " " << rightEyeMiddleY 
					<< ", ���� " << _eyes_y_dif1 << ", ��С����" << _eyes_y_dif 
					<< ", ��0�߶�/10:" << _eyes[0].height/10 << ",  ��1�߶�/10:" << _eyes[1].height/10 << endl;
				string filename = "MedianData//simpleFaceDetection1";// ".png";
				std::ostringstream oss;
				oss << filename << i_face << ".png";
				imwrite(oss.str(),rotateImg);

				if(_eyes_y_dif1 != 0){  //���ĵ�Y���겻���
					if(_eyes_y_dif == 0)
						_eyes_y_dif = _eyes_y_dif1;
					else {
							cout << " ���� " << endl;
						if(abs(_eyes_y_dif) > abs(_eyes_y_dif1)) // �����ת�󣬾���С��
							_eyes_y_dif = _eyes_y_dif1; //��¼С�ľ���
						else if(abs(_eyes_y_dif1)<_eyes[0].height/10 && abs(_eyes_y_dif1)<_eyes[1].height/10)
						{ //������Ǵ��˻���ȣ������ת
							rotateAngle = (i_angle-1)*angle; // ��¼�����ת�Ƕ�
							//���������ȫ�ֲ���
							this->faces = _faces;

							cout<< "��⵽2ֻ�۾�  ���ĵ�Y����:" << leftEyeMiddleY << " " << rightEyeMiddleY << " " << _eyes_y_dif1 << endl;
							return _faces.size();
						}
					}
				} else {
					rotateAngle = (i_angle-1)*angle; // ��¼�����ת�Ƕ�
					//���������ȫ�ֲ���
					this->faces = _faces;

					cout<< "��⵽2ֻ�۾�  ���ĵ�Y����:" << leftEyeMiddleY << " " << rightEyeMiddleY << " " << _eyes_y_dif1 << endl;
					return _faces.size();
				}
			} else if(_eyes.size() == 1){  //�����⵽��1ֻ�۾��������ǲ�����������
				//���ֻ��⵽һֻ�۾���Ҫ�����۾�����������Ե��󴦣�Ϊ˫��ƽ�нǶȡ����Ĳ���������Ĳ��۾����������ߣ��Գƹ��̣�������һֻ�۾���

				cout<< "��⵽һֻ�۾�" <<  endl;
				//break;

			} else {  //���û�м�⵽�۾�
				cout<< "û�м�⵽�۾�" << endl;
				//break;
			}
		}
	}

	if(_faces.size() <1){
		cout << " û�м�⵽����" << endl;
		//�ÿ�ͼ��

	}

	//���������ȫ�ֲ���
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
	// ����ת��������ͼ���м� 
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
	int i = 0;//���õ�һ����

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

	//std::vector<Rect> faces; //��Ϊȫ�ֱ�����.
	Mat frame_gray;
	
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- ��ת���������ȡ��˫��ƽ�е�ͼ��
	int faceCount  = rotateDetectFaces();
	cout << faceCount << endl;
	if(faceCount > 0 ){

		rotate();//��ת����ͼ��
		frame = rotate(frame,this->rotateAngle); // ��תframe��׼������������١�
		frame.copyTo(debugFrame);//��ʼ��debugFrame.

		setFaceParameters(frame);//�������Ĳ���
		debugFace();

		cvtColor( frame, frame_gray, CV_BGR2GRAY );//��Ϊ��ת�ˣ�������Ҫ���´���һ��ԴͼƬ��
		equalizeHist( frame_gray, frame_gray );
		Mat faceROI = frame_gray( faceDetectedRect );

		for( size_t i = 0; i < faces.size(); i++ )
		{
			//-- In each face, detect eyes
			std::vector<Rect> __eyes;
			__eyes = detectEyes(faceROI);//����۾�
			if(__eyes.size()>0){
				setEyesParameters(__eyes,faces[i]);
				debugEyes();
			}
			else{
				cout << " û�м�⵽�۾����˳�����" << endl;
				return;
			}
			
			//����������
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
				cout << " û�м�⵽�졣 " << endl;
			}

			//�����������
			Rect nr = Rect(
				eyeDetectedRects[0].x+eyeDetectedRects[0].width/2, // ��������X
				eyeDetectedRects[0].y+eyeDetectedRects[0].height/2, //��������Y
				eyeDetectedRects[1].x+eyeDetectedRects[1].width/2 -(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2), // �������ĵ��������ĵĿ��
				mouthDetectedRect.y+mouthDetectedRect.height/2-(eyeDetectedRects[0].y+eyeDetectedRects[0].height/2)); // �����ĵ��������ĵĸ߶�
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
				cout << " û�м�⵽���ӡ� " << endl;
			}


			//ȷ����������faceMiddleRect��������ڱ��ӣ��Ͱ����ӣ�������������죬�Ͱ��죻���򣬾Ͱ�����
			if(noseDetectedRect.x > 0)
				faceMiddleRect = Rect(noseDetectedRect.x + noseDetectedRect.width/2, faces[i].y, 2, faces[i].height);
			else if(mouthDetectedRect.x>0)
				faceMiddleRect = Rect(mouthDetectedRect.x+mouthDetectedRect.width/2, faces[i].y, 2, faces[i].height);
			else
				faceMiddleRect = Rect(faceDetectedRect.x+faceDetectedRect.width/2, faces[i].y, 2, faces[i].height);

		}
	}
	else{
		cout << " û�м�⵽����ʲôҲ������" << endl;
		return;
	}
	imwrite("MedianData//simpleFaceDetection.png",frame);
}

//
void appFace::simpleFaceDetection(){
	Mat frame;
	imageOrigine.copyTo(frame);

	//std::vector<Rect> faces; //��Ϊȫ�ֱ�����.
	Mat frame_gray;
	
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	if(faces.size() <1)
		cout << " û�м�⵽����" << endl;
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

					if (*itr == eyes[j]) eyes.erase(itr);//ɾ��ֵΪ3��Ԫ��
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

		//����������
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
			//����첻�����������غϣ���������֮�䣬�������졣
			int mx = int(faces[i].x+mouths[mi].x+mouths[mi].width/2);
			int my = int(faces[i].y+mouths[mi].y+mouths[mi].height/2);
			//cout << mx << ": " << x1 << ": " << x2<< "|| my :"<<my<<"  "<<y1 << "  "<< y2 << endl;
			bool _betwin = false;
			if((mx>=x1 && mx<=x2) || mx<= x1 && mx >=x2) _betwin = true;
			if(_betwin && my>y1 && my>y2) 
			{
				//���������һ�������졣
				if(mouths[mi].y>_y){
					_y = mouths[mi].y;
					_ym = mi;
				}
			}
		}
		//�ҵ���һ����
		if(_ym>=0){
			this->mouthDetectedRect = Rect(this->faceDetectedRect.x+mouths[_ym].x,this->faceDetectedRect.y+mouths[_ym].y,mouths[_ym].width,mouths[_ym].height);
			//Rect _rect_ml = Rect(faces[i].x + mouthDetectedRect.x, faces[i].y + mouthDetectedRect.y, mouthDetectedRect.width, mouthDetectedRect.height);
			rectangle(frame, mouthDetectedRect, Scalar(100,100,100));
			Rect _rect_mlm = Rect(mouthDetectedRect.x+mouthDetectedRect.width/2, faces[i].y, 2, faces[i].height);
			rectangle(frame, _rect_mlm, Scalar(100,100,100));
		}


				//�����������
		//Mat faceROI = frame_gray( faceDetectedRect );
		std::vector<Rect> noses;
		//�������ֵӰ��޴�
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
				//�������棬���۾����棬������֮�䣬�����Ǳ��ӡ�

				//ȡ��������Ǹ�
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


		//ȷ����������faceMiddleRect��������ڱ��ӣ��Ͱ����ӣ�������������죬�Ͱ��죻���򣬾Ͱ�����
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
	//�˾�������
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
	//Mat faceSample;//��ģ�͵���ʱ�ļ�
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

//���ݲ�ͬģʽ��������ģ�塣
void appFace::resizeFaceModel(int mode){
	Mat faceModel = chm.currentHead.faceModel;
	//������������ź�ģ������Ҫ��������С���Ͱ�ģ���������滻����ȿ�
	//if(_fcols<faceChangeRect.width) _fcols=faceChangeRect.width;
	//TODO �����ǽ���ֵ����Ϊ��ȷ��Ӧ���Ǹ������������ж�����С��
	int faceWidth = -1;
	//�ж�������
	int leftFaceWidth = getLeftFaceWidth();
	//�ж�������
	int rightFaceWidth = getRightFaceWidth();
	cout << leftFaceWidth << "  " << rightFaceWidth << endl;
	//���������>0����������ȼ���
	if(leftFaceWidth > 0 && leftFaceWidth>rightFaceWidth) faceWidth = leftFaceWidth*2;
	//���������>0����������ȼ���
	if(rightFaceWidth > 0 && rightFaceWidth>leftFaceWidth) faceWidth = rightFaceWidth*2;
	//faceWidth = 0;
	//������Ҷ�û�У������µ����ĵȱ�����

	//��Ϊ��ͷ��ʱ�����������Ե����¸߶�,��ȵȱ�����
	if(faceWidth > 0){
		//������Ե��������Ȱ����ӿ�10%�������ᱣ�ֱ������Ρ�����������Ҫȥ���������С�
		faceWidth = faceWidth*1;
		fWidth = (double)  faceWidth / faceModel.cols;
		fHeight = (double) faceChangeRect.height / faceModel.rows;
		//�������������Ŀ�ȣ����޸�faceChangeRect
		faceChangeRect = Rect(faceChangeRect.x+(faceChangeRect.width-faceWidth)/2,faceChangeRect.y,faceChangeRect.width,faceChangeRect.height);
	}
	else{ // TODO: ������൲�������ƽ�ָ���������
		faceWidth = faceModel.cols *  ((double)faceChangeRect.height / (double)faceModel.rows);
		fWidth = (double)  faceWidth / faceModel.cols;
		fHeight = (double) faceChangeRect.height / faceModel.rows;
		faceChangeRect = Rect(faceChangeRect.x+(faceChangeRect.width-faceWidth)/2,faceChangeRect.y,faceChangeRect.width,faceChangeRect.height);
	}

	//��С��ֵ���ţ���Ϊ����Ŀ��ܻ��۾����硣
	double _fcols = 0.0;
	//���faceWidth>0��
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

//��resultImageͼ�У���modeģʽ����������faceModel��
void appFace::replaceFace(Mat faceModel,Mat &resultImage,int mode){

	Mat frame;resultImage.copyTo(frame);
	//Mat faceSample;
	//-- Change face
	Mat faceSampleBGR;
	cvtColor(faceSample, faceSampleBGR, CV_BGRA2BGR);

	Mat _bgraFrame,_bgraFrameLight,_bgrFrameSkin,_skinMask;
	//������Ҫ����
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

	//�����ϵ���ʾ����
	uchar* mask_face_replace_data = maskFaceReplace.ptr<uchar>(0);
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);

	//����ģ��ӿ��ˣ����ܴ�����ƥ�䣬��Ӧ�ô��м���롣��Ҫ��faceChangeRect�����Ƶ����ߡ�
	int middle = -1*(this->faceMiddleRect.x - faceChangeRect.x-faceChangeRect.width/2) +  faceSample.cols/2 - faceChangeRect.width/2;

	
	//����ƽ�⴦��
	IplImage* imageFace = &IplImage(faceSampleBGR);
	IplImage* imageBgr = &IplImage(_bgraFrameLight);
	IplImage* imageBgrSkin = &IplImage(_bgrFrameSkin);
	double gFace = get_avg_gray(imageFace);
	double gBgr = get_avg_gray(imageBgr);
	//Ϊ�˵�����������ע�͵���
	set_avg_gray(imageBgr,imageBgr,gFace*0.9);

	//��ɫ����
	CvSize imageSize = cvSize(imageBgrSkin->width, imageBgrSkin->height);
	IplImage *imageSkin = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
	
	cvSkinSegment(imageBgrSkin,imageSkin);
	//cvSkinYUV(imageBgrSkin,imageSkin);
	//cvSkinHSV(imageBgrSkin,imageSkin);
	Mat skinMat= Mat(imageSkin);

	imwrite("MedianData//skinTemp.png", skinMat);
	imwrite("MedianData//faceTemp.png", faceSampleBGR);
	//д�������Ⱥ���ļ�
	imwrite("MedianData//bgrLight.png", _bgraFrameLight);

	//��������
	imwrite("MedianData//bgrLightBeforeChangeFaceNoTransparent.png", _bgraFrameLight);
	//changeFace(_bgraFrameLight,mask_face_replace_data,faceSample);
	imwrite("MedianData//bgrLightAfterChangeFaceNoTransparent.png", _bgraFrameLight);

	for (int _row = 0; _row < faceSample.rows ; _row++)
	{
		Vec3b *colData = faceSampleBGR.ptr<Vec3b>(_row);
		Vec4b *colDataBGRA = faceSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < faceSample.cols ; _col++){
			int r = faceChangeRect.y + _row; 
			//����ģ��ӿ��ˣ����ܴ�����ƥ�䣬��Ӧ�ô��м���롣��Ҫ��faceChangeRect�����Ƶ����ߡ�
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
					//�仯������
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
					//ȥ���������Ե�����
					//if(vb_hsv[2]<200)
					{
						vf_hsv[0] = vb_hsv[0];
						//��������ģ����Ͼ��Ե����ˣ����ࡣ
						//vf_hsv[1] = vf_hsv[1]+vb_hsv[1]*0.3;N

						//vf_hsv[2] = vf_hsv[2]+vb_hsv[2]*0.3;
					}
					vf_rgb = kcvHSV2RGB(vf_hsv);
					//�����ģ����ͬλ�ò�͸����˵���������ݵ�
					if(colDataBGRA[_col][3]>5){
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
						bgra_frame_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
						bgra_frame_light_data[index] = Vec4b(vf_rgb[0],vf_rgb[1],vf_rgb[2], (1-rate)*mask_real_face_data[index]);
					} else {
						//�����ģ����ͬλ��͸����˵���������Ĳ��֣�Ҫ��ȥ��

					}
					//*/
					//�ȸĳɲ�͸��
					//bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
					continue;
				}
				if(mask_real_face_data[index] < 32){
					continue;
				}
				//���������ΧΪ��ɫ��ֱ��ȡ��ɫֵ��
				if (mask_real_face_data[index] > 223){
					bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
					bgra_frame_light_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
				}
				//����Ҫ��͸��������
				else {
					//bgra_frame_data[index][3] = 255;
					//double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
					//bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2], 255 - mask_real_face_data[index]);
					//�仯������
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
					
					/*//����͸������ģ������
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

	//���꣬�Ȼ����ӡ��趨���ӵ�λ�ã����У����ϡ�
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
		//�趨���ӵ�����
		noseRect = Rect(
			//��ģ���Ƶ�����Ϊ���ĵ�λ�ã���noseDectctedRect��y����
			this->faceMiddleRect.x - noseWidth/2,
			//�����ϱ�Ҫ����ǰ���ģ��±��ڱ��ӿ������,��������1/8.
			eyeDetectedRects[0].y+eyeDetectedRects[0].height/2+noseHeight/10,
			noseWidth,
			noseHeight);
		//�������Ŵ���С���ӿ��
		//ȡ��ֵ�����ű���������ģ��	
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

		//�����ű�С��ֵ���ȱȷ����۾�
		double eHeight=0.0;
		double eWeith=0.0;
		resizeEyeRate = 0.0;

		//�������ǿдʵ���8��������۾���С����
		if(mode == REALMODEPLUS){		//REALMODE��дʵ�档��ʵ����ٴ�С��λ�ã����Խӽ�1��ϵ��������١�
			//��ȡ����ģ��3����
			Point ps[2][3];//ȡ�۾�ģ������ҵ㡣lr:0ȡ���۰׵㣬1ȡ���۰׵㣬2ȡ���۱�Ե��3ȡ���۱�Ե��4ȡ�۰��ϱ�Ե��5ȡ���ϱ�Ե��
			ps[0][0] = this->getEyeModelPoint(leftEyeModel,0);
			ps[0][1] = this->getEyeModelPoint(leftEyeModel,3);
			ps[0][2] = this->getEyeModelPoint(leftEyeModel,5);
			ps[0][0] = this->getEyeModelPoint(rightEyeModel,1);
			ps[0][1] = this->getEyeModelPoint(rightEyeModel,2);
			ps[0][2] = this->getEyeModelPoint(rightEyeModel,5);

			resizeModel(leftEyeModel,ps[0][0],ps[0][1],ps[0][2],eyesPoint[2],eyesPoint[3],eyesPoint[0]);
			resizeModel(rightEyeModel,ps[1][0],ps[1][1],ps[1][2],eyesPoint[6],eyesPoint[7],eyesPoint[4]);

		}



		//�����дʵ���1��������۾���С����
		if(mode == REALMODE){		//REALMODE��дʵ�档��ʵ����ٴ�С��λ�ã����Խӽ�1��ϵ��������١�
			eHeight = (double)((double)eyeDetectedRects[0].height / (double)leftEyeModel.rows);
			eWeith = (double)((double)eyeDetectedRects[0].width / (double)leftEyeModel.cols);
			if(eWeith>eHeight)
				resizeEyeRate = eHeight;
			else
				resizeEyeRate = eWeith;

			//�����ǵȱ���С��0.9����ʵ���Ӧ�ø������ڿ����ʵ�ʱ�����
			resize(leftEyeModel, leftEyeSample, Size(leftEyeModel.cols*resizeEyeRate*0.9, leftEyeModel.rows*resizeEyeRate*0.9));
			resize(rightEyeModel, rightEyeSample, Size(rightEyeModel.cols*resizeEyeRate*0.9, rightEyeModel.rows*resizeEyeRate*0.9));
			resize(leftEyeWithBrowModel, leftEyeWithBrowSample, Size(leftEyeWithBrowModel.cols*resizeEyeRate*0.9, leftEyeWithBrowModel.rows*resizeEyeRate*0.9));
			resize(rightEyeWithBrowModel, rightEyeWithBrowSample, Size(rightEyeWithBrowModel.cols*resizeEyeRate*0.9, rightEyeWithBrowModel.rows*resizeEyeRate*0.9));
			resize(leftEyePupilModel,leftEyePupilSample,Size(leftEyePupilModel.cols*resizeEyeRate*0.9, leftEyePupilModel.rows*resizeEyeRate*0.9));
			resize(rightEyePupilModel,rightEyePupilSample,Size(rightEyePupilModel.cols*resizeEyeRate*0.9, rightEyePupilModel.rows*resizeEyeRate*0.9));
		}
		//�����Q�ȷ��2���������ű��������۾�
		if(mode == QMODE){
			resize(leftEyeModel, leftEyeSample, Size(0, 0), fWidth, fHeight);
			resize(rightEyeModel, rightEyeSample, Size(0, 0), fWidth, fHeight);
			resize(leftEyePupilModel,leftEyePupilSample,Size(0,0),fWidth, fHeight);
			resize(rightEyePupilModel,rightEyePupilSample,Size(0,0),fWidth, fHeight);
		}

		//����ǿ�ͨ���3����С�ĵȱ�����
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
	//���ֻ��⵽һֻ�۾������ݶԳƣ��Ƶ�����һֻ�۾���
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
	//�ƶ��۾�
	//��Ϊ��������������Ҫ�������ߣ��ƶ��۾�����Ӧλ�á�
	//����۾�eyeDetectedRects[0]��������
	//if (this->eyeDetectedRects[0].x < this->faceDetectedRect.x + 0.5 * this->faceDetectedRect.width)
	int leftEyeNum = 0,rightEyeNum = 1;
	if (this->eyeDetectedRects[0].x > faceMiddleRect.x){
		leftEyeNum = 1;
		rightEyeNum = 0;
	}

	//��Ϊ�������߶��롣_dif�����۾���ƫ������Ҳ����ͫ���������ĵ�ƫ������Ҳ���۾��ƶ�����
	//***************** �������ۼ�� ****************
	//�۾����
	int _tj = (eyeDetectedRects[rightEyeNum].x - (eyeDetectedRects[leftEyeNum].x+eyeDetectedRects[leftEyeNum].width))/2;
	//����ƫΪ��������ƫΪ��
	int _py = ((eyeDetectedRects[rightEyeNum].x-_tj) - faceMiddleRect.x);
			
	//�۾�����ͫ��y������Ϊ���ģ����¾��У�������Ϊ���ģ������ۼ�ࡣ
	leftEyeRect = Rect(
		//����X = ���ߣ������۾�ģ�;��룬����ͫ��
		faceMiddleRect.x - leftEyeSample.cols - _tj,
		eyeDetectedRects[leftEyeNum].y + 0.5*eyeDetectedRects[leftEyeNum].height-0.5*leftEyeSample.rows, 
		leftEyeSample.cols, 
		leftEyeSample.rows );
	rightEyeRect = Rect(
		//����X= ���ߣ�������ͫ��
		faceMiddleRect.x + _tj,
		eyeDetectedRects[rightEyeNum].y + 0.5*eyeDetectedRects[rightEyeNum].height-0.5*leftEyeSample.rows, 
		rightEyeSample.cols, 
		rightEyeSample.rows );
	/*//����������ģʽ��
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
	// �����ۼ����ģʽ��������Ϊ��׼�����Ե�üë����ͻ������ʱ���ͻ�ƫ��Y�����۵�Ϊƽ��������չ��
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
		//��ͫ��X = �������� + �����ű���ƫ��
		leftEyeRect.x + leftEyeRect.width/2 - leftEyePupilSample.cols/2 + _py*resizeEyeRate,
		eyeDetectedRects[leftEyeNum].y+ (0.5*eyeDetectedRects[leftEyeNum].height  - 0.5*leftEyePupilSample.rows),
		leftEyePupilSample.cols, 
		leftEyePupilSample.rows );
	rightEyePupilRect = Rect(
		//��ͫ��X = �������� + �����ű���ƫ��
		rightEyeRect.x + rightEyeRect.width/2 - rightEyePupilSample.cols/2 + _py*resizeEyeRate,
		eyeDetectedRects[rightEyeNum].y+ 0.5*eyeDetectedRects[leftEyeNum].height  - 0.5*rightEyePupilSample.rows,
		rightEyePupilSample.cols, 
		rightEyePupilSample.rows );
	//Ҫ���۵����Ķ�λ�������Ǵ���ˣ������������۹����ߵ����⡣

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
			//�ҵ����λ�õ㣬�۾�ģ���е����ݣ���Ϊ���۾�ʱ�Ѿ��仯Ϊ�۾�ֵ������ȡ�������ݾ����ˡ�
			//�ù�ʽ�������ɫ��Ϻ����ɫ
			//������ˡ�Ӧ���Ǳ�����͸��ֵ����ȡ�����ֵ��
			/*
			int _r = (int)(bgra_frame_data[index][0]*bgra_frame_data[index][3]*(1-colData[_col][3])+colData[_col][0]*colData[_col][3])/(bgra_frame_data[index][3]+colData[_col][3]+bgra_frame_data[index][3]*colData[_col][3]);
			int _g = (int)(bgra_frame_data[index][1]*bgra_frame_data[index][3]*(1-colData[_col][3])+colData[_col][1]*colData[_col][3])/(bgra_frame_data[index][3]+colData[_col][3]+bgra_frame_data[index][3]*colData[_col][3]);
			int _b = (int)(bgra_frame_data[index][2]*bgra_frame_data[index][3]*(1-colData[_col][3])+colData[_col][2]*colData[_col][3])/(bgra_frame_data[index][3]+colData[_col][3]+bgra_frame_data[index][3]*colData[_col][3]);
			*/
			//��ֵ��͸����255
			//����û���۾�ģ�壬��ע�͵����š�
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
	//����졣�趨���λ�ã����У����ϡ�
	Mat mouthSample;
	Rect mouthRect;
	if(this->mouthDetectedRect.x>0){
		double mouthResize = 0.0;
		int mouthWidth ;
		int mouthHeight ;

		//�����дʵ���
		if(mode == REALMODE){

			//���ȱ������ţ�ʡ�£�û�з��
			mouthResize = (fWidth+fHeight)/2;

			//�����������˵����������ˡ�
			if(mouthModel.cols*mouthResize > this->mouthDetectedRect.width){
				//����ȡ����ȵ�4/5��
				mouthWidth = this->mouthDetectedRect.width;
				mouthHeight = mouthModel.rows*((double)mouthWidth/(double)mouthModel.cols);
				//�������Ŵ���С��
				//ȡ��ֵ�����ű���������ģ��	
				resize(mouthModel, mouthSample, Size(mouthWidth, mouthHeight));
				//�趨������򡣽�ģ���Ƶ�����Ϊ���ĵ�λ�ã���mouthDectctedRect��y����
			} else {
				//resize(mouthModel, mouthSample, Size(mouthWidth, mouthHeight));
				resize(mouthModel, mouthSample, Size(0, 0),mouthResize,mouthResize);
			}
		}

		if(mode == QFITMODE){
			//���ȱ������ţ�ʡ�£�û�з��
			mouthResize = (fWidth+fHeight)/2;
			resize(mouthModel, mouthSample, Size(0, 0),mouthResize,mouthResize);
		}

		if(mode == QMODE){
			//���ȱ������ţ�ʡ�£�û�з��
			mouthResize = (fWidth+fHeight)/2;
			resize(mouthModel, mouthSample, Size(0, 0),mouthResize,mouthResize);
		}


		//��������Ǳ��ӵ��������⵽��������ص��м䣬�����1/5
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
	initCounter(); // ��ʼ��
	initTempMat();

	// ================= ���� ========================
	resizeFaceModel(mode);
	Mat faceModel = chm.currentHead.faceModel;
	replaceFace(faceModel,resultImage,mode);

	// ================= ������ ========================
	resizeNoseModel(mode);

	// ================= ���۾� ========================
	resizeEyes(mode);
	replaceEyes(mode);

	// ================= ���� ========================
	resizeMouth(mode);




	saveImages(_bgraFrameLight,"lightBeforeChangeFace.png");
	saveImages(_bgraFrame,"beforeChangeFace.png");
	//��������
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


//frame��ԭʼͼ��
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

	//�����������
	uchar* mask_face_data = maskFaceReplace.ptr<uchar>(0);
	//��������
	Vec4b* contour_data = _contour.ptr<Vec4b>(0);
	//�������Ե
	uchar* mask_real_contour_data = imageRealContourSM.ptr<uchar>(0);
	//��������������Ե
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
	//����ƽ�⴦��
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

	//�����������
	uchar* mask_face_data = maskFaceReplace.ptr<uchar>(0);
	//��������
	Vec4b* contour_data = _contour.ptr<Vec4b>(0);
	//�������Ե
	uchar* mask_real_contour_data = imageRealContourSM.ptr<uchar>(0);
	//��������������Ե
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
	//����ƽ�⴦��
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
	//˳����ܴ���
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
        //H = 2.0/3.0;          //Windows��SֵΪ0ʱ��Hֵʼ��Ϊ160��2/3*240��
        H = 0;                  //HSL results = 0 �� 1
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
    if (S == 0)                       //HSL values = 0 �� 1
    {
        R = L * 255.0;                   //RGB results = 0 �� 255
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
				//ת��������
				Vec3d p = kcvRGB2HSL(colDataBGR[j]);
				//�ۼ�
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
				//ת��������
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

//������
//�������ݣ�Ҫ�滻����MASK����ģ�塣�򵥵Ľ��������ƶ���ȥ���������ӡ�
void appFace::changeFace(Mat _bgraFrameLight,uchar *mask_face_replace_data,Mat faceSample){
	//Vec4b *bgra_frame_data = _bgraFrame.ptr<Vec4b>(0);
	Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);

	int leftCol = this->calFirstColOfContour(faceSample);
	int rightCol = this->calLastColOfContour(faceSample);
	int topRow = this->calFirstRowOfContour(faceSample);
	int buttomRow = this->calLastRowOfContour(faceSample);

	//cout << "leftCol:" << leftCol <<" rightCol:"<< rightCol <<" topRow:"<< topRow <<" buttomRow:"<< buttomRow << endl;
	//���п�ʼ��
	for(int _col = faceSampleRect.x;_col<faceSampleRect.x+faceSampleRect.width;_col++){
		int firstRow = 0,lastRow = 0;
		//�����һ�п�ʼ��
		for(int _row=faceSampleRect.y+faceSampleRect.height;_row>faceSampleRect.y;_row--){
			int index = _row*_col+_col;
			//�����ǰ����mask_face_replace_data��
			if(mask_face_replace_data[index]>0){
				if(firstRow == 0) firstRow = _row;

				//�����ǰ����faceSample��
				//���ﻹû�ж�faceSample��λ
				//��� ������������
				//if(faceSampleRect.x < _col && faceSampleRect.x+faceSampleRect.width > _col && faceSampleRect.y < _row && faceSampleRect.y+faceSampleRect.height > _row )
				{
					//����ģ��ĵ�����ֵ��
					int _row_ = _row - faceSampleRect.y;
					int _col_ = _col - faceSampleRect.x;

					Vec4b *colDataBGRA = faceSample.ptr<Vec4b>(_row_);

					//�����ģ����������ǲ�͸���ģ�˵����ֵ��
					if(colDataBGRA[_col_][3]>0){
						if(lastRow == 0) lastRow = _row;
						if(firstRow != lastRow) {

							//Ӧ��������İ취���Ȱѱ�Ե����ɾ���
							//�ⲿ���Ƕ�����ģ���Ҫ�ġ�
							CvSize imageSize = cvSize(1, firstRow - lastRow);
							IplImage *imageSkin = cvCreateImage(imageSize, IPL_DEPTH_8U, 4);
							Mat mt = Mat(imageSkin);
							Vec4b *_d = mt.ptr<Vec4b>(0);
							for(int i=0;i<firstRow - lastRow;i++){
								_d[i] = bgra_frame_light_data[(firstRow+i)*_bgraFrameLight.cols+_col];
							}
							//cout << "��һ�� mt.rows = " << mt.rows << " " << endl;
							resize(mt, mt, Size(1, 2*(firstRow - lastRow)));
							Vec4b *_d_ = mt.ptr<Vec4b>(0);
						
							//cout << "�ڶ��� mt.rows = " << mt.rows << " " << endl;

							for(int j=0;j<2*(firstRow - lastRow);j++){
							//cout << "j = " << j << " " << endl;
								//bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col] = mt.data[j];
							
								if((_row+j)*_bgraFrameLight.cols+_col < _row_){
									cout << " д������������" << endl;
								}else {
									bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col][0] = _d_[j][0];
									bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col][1] = _d_[j][1];
									bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col][2] = _d_[j][2];
									bgra_frame_light_data[(lastRow+j)*_bgraFrameLight.cols+_col][3] = _d_[j][3];
								}
							}
						//ɨ����һ��
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
* frame ԭͼ
* mouthModel ���ӵ�ģ��ͼ
* realSource
*/
void appFace::replaceNose(Mat _bgraFrameLight,Mat noseSample,Rect noseRect,Mat maskRealFace)
{
			
	//�趨���λ�ã����У����ϡ�
	//Mat mouthSample;
//	mouthSample = mouthModel.
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *bgra_frame_data = _bgraFrameLight.ptr<Vec4b>(0);
	//Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);

	if(this->noseDetectedRect.x>0){

		//�趨���ӵ����򡣽�ģ���Ƶ�����Ϊ���ĵ�λ�ã���mouthDectctedRect��y����
		//noseRect = Rect(this->faceMiddleRect.x - noseSample.cols/2,noseDetectedRect.y+noseDetectedRect.height-noseSample.rows,noseSample.cols,noseSample.rows);
		//������
		if(this->noseDetectedRect.x>0){
			for (int _row = 0; _row < noseSample.rows ; _row++)
			{
				Vec4b *colData = noseSample.ptr<Vec4b>(_row);

				for (int _col = 0 ; _col < noseSample.cols ; _col++){
					int r = noseRect.y + _row; 
					int c = noseRect.x + _col;
					int index = r*_bgraFrameLight.cols + c;//frame


					//-- Get valid area of nose model
					//Ϊ�˽���ױ����⣬�趨͸����Ϊ<250 �����򡣱��Ӳ��У��ϱ�Ե�����࣬��Ҫ���䣬�����ںϡ�
					if (colData[_col][3] == 0)
					{
						continue;
					}

					//���ӣ�ֻȡɫ���ǲ����ˣ����ǻơ�
					
					colData[_col][0] = superimposingTransparent(colData[_col][0],bgra_frame_data[index][0],colData[_col][3],255);
					colData[_col][1] = superimposingTransparent(colData[_col][1],bgra_frame_data[index][1],colData[_col][3],255);
					colData[_col][2] = superimposingTransparent(colData[_col][2],bgra_frame_data[index][2],colData[_col][3],255);
					//colData[_col][3] = 255;
					
					//-- Override face where mask > 0
					if(true){
						//�仯������
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
						//ȥ���������Ե�����
						//if(vb_hsv[2]<200)
						{
							vf_hsv[0] = vb_hsv[0];
							//vf_hsv[1] = vb_hsv[1];
							//vf_hsv[2] = vb_hsv[2];
						}
						vf_rgb = kcvHSV2RGB(vf_hsv);
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255; // mask_real_face_data
						//�ȸĳɲ�͸��
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
* frame ԭͼ
* mouthModel ���ģ��ͼ
* realSource
*/
void appFace::replaceMouth(Mat _bgraFrameLight,Mat mouthSample,Rect mouthRect,Mat maskRealFace)
{
	
	//�趨���λ�ã����У����ϡ�
	//Mat mouthSample;
//	mouthSample = mouthModel.
	//Rect mouthRect;
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *bgra_frame_data = _bgraFrameLight.ptr<Vec4b>(0);
	//Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);

	if(this->mouthDetectedRect.x>0){

		//������
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
						//�仯������
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
						//ȥ���������Ե�����
						//if(vb_hsv[2]<200)
						{
							vf_hsv[0] = vb_hsv[0];
							//vf_hsv[1] = vb_hsv[1];
							//vf_hsv[2] = vb_hsv[2];
						}
						vf_rgb = kcvHSV2RGB(vf_hsv);
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255; // mask_real_face_data
						//�ȸĳɲ�͸��
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



Mat appFace::createROI(Mat m,string name,int pre,int mode,int range){ // mԴͼ��pre��Ԥ����0���ޣ�1��ֱ��ͼ��mode��1��cvThreshold��2��cvAdaptiveThreshold��3��Canny��Ե��⡣
		//�ٳ�Ŀ��
		Mat _gray;
		Mat mROI;
		//�ȵ���ԭͼƽ������
		IplImage* _imageBgr = &IplImage(m);
		Mat _tar ;
		m.copyTo(_tar);
		IplImage* _imageTar = &IplImage(_tar);
		set_avg_gray(_imageTar,_imageTar,(double)128.0);//����ƽ�⴦��
		imwrite("MedianData//"+name+"_128Light.jpg", _tar);
		
		//cvSmooth(_imageBgr,_imageTar);
		cvSmooth(_imageTar,_imageTar);
		cvtColor( _tar, _gray, CV_BGR2GRAY );
		Mat mt2 = Mat(_gray);
		//eye_gray_r = removeNoise(eye_gray_r,3);//��˹ȥ��
		IplImage* eye_gray_r = &IplImage(_gray);
		//imwrite("MedianData//"+name+"_le1.jpg", mt2);

		int offSet = 0 ;
		if(m.cols > m.rows)
			offSet = ((float)(m.rows));
		else
			offSet = ((float)(m.cols));
		//cout << "m.rows:" << m.rows << " - m.cols:" << m.cols <<  " . offSet: " << offSet << endl;

		if(pre == 1){
			//�۾�ֱ��ͼ��Ч�����á���Ϊ�۾��ĺڰ��Ѿ��������ˡ����Ӻ������Ҫ.
			equalizeHist( _gray, _gray );
		//imwrite("MedianData//le2.jpg", mt2);
		}

		if(mode == 3){
			//****************************************************************************
			//���ֵ�����ϵġ�����ԭ���ǣ����ȵ���128�����۾���˵�����߸�Ϊͻ���ˡ�
			Mat edge;
			Canny(_gray, edge, 50, 150);
			//imwrite("MedianData//"+name+"_leedge.jpg", edge);
			//****************************************************************************
			reverseROI(_gray); // ת�ɺڵװ���
		}

		IplImage *graySource = &IplImage(_gray);
		CvSize imageSize1 = cvSize(_gray.cols, _gray.rows);
		IplImage *grayDst = cvCreateImage(imageSize1, IPL_DEPTH_8U, 1);

		if(mode == 1){ // ȷ����ֵ
			cvThreshold(graySource, grayDst, range, 255,CV_THRESH_BINARY);
			reverseROI(grayDst); // ת�ɺڵװ���
		}

		if(mode == 2 || mode == 3){ // ����Ӧ��ֵ
			range =  m.rows/range; // ��ΧΪ���1/range
			if(range % 2 == 1 && range > 1){}else range++;
			if(mode == 2) // ��ֵ�㷨
				cvAdaptiveThreshold(graySource, grayDst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, range, range);

			if(mode == 3) // ��˹�㷨
				cvAdaptiveThreshold(graySource, grayDst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, range, range);
			
			reverseROI(grayDst); // ת�ɺڵװ���
			filterBlock(grayDst,5,5,255); // ���˴�С��Ҫ�������
		}

		Mat mt = Mat(grayDst);
		imwrite("MedianData//"+name+"_le2.jpg", mt);
		//imwrite("MedianData//"+name+"_le.jpg", mt2);
		return mt;
}

void appFace::filterBlock(Mat frame,int w,int h,int v){ // ���˵�С��(w,h)�Ķ����顣stack overflow!!
	//������涼д���ˡ�Ӧ������������
	//�ҵ�һ���㣬Ȼ������������������Σ��ҳ���ͨ�飬��������С��ָ�����Σ�ɾ���������¼����飬�����Ѱ����һ��������ٰ�����鸴�ƻ�ȥ��
	//��������һ�����Mat��һ���Ƿ���ֵ����һ������ʱ������
	//cout << " w: " << w << " " << h << endl;
	Mat src,tar;
	int b=0,bt=50,f=255,ft=250;
	//imwrite( "MedianData//frameClear.png",frame);
	frame.copyTo(src);
	//imwrite( "MedianData//srcClear.png",src);
	tar = frame.clone();frame.setTo(Scalar(b));
	//��ԭͼ�в��ң�����ҵ���ֵ�㣬��������ֱ��������ͨ�㶼�ҵ����������ŵ���ʱ������ԭͼ�������顣�ж�������С���������ָ����Χ�����Ƶ�����ֵMat�
	for(int _row=0;_row<src.rows;_row++){
		for(int _col=0;_col<src.cols;_col++){
			int index = _row * src.cols + _col;
			if(src.data[index] ) { //�õ���ֵ��
				//cout << " col: " << _col << " " << _row << endl;
				tar.setTo(Scalar(b)); // ��ɱ���ɫ��
				nineBox(src,tar,_col,_row,true);//��������nineBox
				//Mat reverse; tar.copyTo(reverse);reverse.setTo(Scalar(b));
				imwrite("MedianData//filterBlockReverse.png",tar);
				Rect r = getROIRect(tar);
				//reverseROI(tar);
				//��תSRC��ʹ�������Ͽ��������
				int width = r.width;//this->calLastColOfContour(reverse) - this->calFirstColOfContour(reverse);
				int height = r.height;//this->calLastRowOfContour(reverse) - this->calFirstRowOfContour(reverse);
				//cout <<"width:"<< width << " height:" << height << " x:" << r.x << " y:" << r.y << " w:" << w << " h:" << h << endl;
				if(width >= w && height >= h) {
					copyROI(tar,frame); // ������ӵ�һ�������ָ����ߣ��͸��ƻ�ȥ��
					imwrite("MedianData//frameBlock.png",frame);
				}
			}
		}
	}
	//reverseROI(frame);
	return;
}

//����MASK������ͨ���ˡ����frame�е�һ������MASK�ڣ���ô����MASK֮�����ͨ��Ҳ������(��MASK�ڵģ�����ͨ����)������ɾ����
void appFace::filterBlock(Mat frame,Mat mask,bool blackBackground){
	Mat ff,t,ret;
	if(!blackBackground){reverseROI(frame);reverseROI(mask);	}
	int b=0,f=255;
	frame.copyTo(ff);frame.copyTo(t);frame.copyTo(ret);ret.setTo(Scalar(b));// ����ȫ�������Ƴ��������������ԭͼӰ�졣
	for(int _row = 0;_row < ff.rows;_row++){
		for(int _col = 0;_col<ff.cols;_col++){
			if(ff.at<uchar>(_row,_col)){ //f.data[_row*f.cols+_col]<50){ // ����������ֵ
				t.setTo(Scalar(b));
				nineBox(ff,t,_col,_row,blackBackground);

				if(joinMask(t,mask,true)) {//��ΪcopyROIҪ���Ǻڵװ��֣�������Ҫ��תһ�¡���Ϊ�ױ���
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

// �ж�t�еĵ��Ƿ�����mask��Χ�ڵġ�
bool appFace::joinMask(Mat t,Mat mask,bool blackBackground){ 
	int f=255,ft=200,b=0,bt=50;
	if(!blackBackground){ // ��ת�ɰ׵׺���
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
					
					if(!blackBackground){ // ��ת�ɰ׵׺���
						reverseROI(t);reverseROI(mask);
					}
					
					return true;
				}
			}
		}
	}
	if(!blackBackground){ // ת�ɰ׵׺����ˣ��ٸ�ת��ȥ��
		reverseROI(t);reverseROI(mask);
	}	
	return false;
}

int appFace::getROIWidth(Mat _roi,int mode){ // mode:0���Ǻڵף�1�ǰ׵�

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
	submatt = submatt + _submat;//дĿ��
	submat.setTo(Scalar(b));//���Դ

	for(int _r=0;_r<r.height;_r++){//�жϵݹ�
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

//ȡSRC��ָ���㣨startX,startY������ͨMASK���������Ƴ���TAR�С�Ҫ����ʼʱ��ָ������ǰ��ֵ��ͼ��Ϊ�׵׺��֡�
Mat appFace::nineBox(Mat frameSrc,Mat frameTar,int startX,int startY){
	return nineBox(frameSrc,frameTar,startX,startY,true);
}

Rect appFace::getROIRect(Mat src){
	int left=src.cols,top=src.rows,right=0,buttom=0;
	// TODO: ���þ���Ӽ���
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

Rect appFace::copyROI(Mat src,Mat tar){//��src�����0�ĵ㣬���Ƶ�tar��.
	if(!tar.cols){
		src.copyTo(tar);
	}
	int left=src.cols,top=src.rows,right=0,buttom=0;
	// TODO: ���þ���Ӽ���
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

//  ���������ĸ�������飬ÿ���������ң���һ�����ۣ��ڶ������ۡ�����ȷ����Ե�ĵ㣬Ϊ��0��0��.
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

	//�߼��жϣ����

	return eyes;
}

Point* appFace::getNosePoint(Mat noseROI){ //  ��������������飬���ҡ�����ȷ����Ե�ĵ㣬Ϊ��0��0��.
	Point  nose[2];
	return nose;
}

Point* appFace::getMouthPoint(Mat mouthROI){ //  �����ĸ�������飬�������ҡ�����ȷ����Ե�ĵ㣬Ϊ��0��0��.
	Point  mouth[4];
	return mouth;
}

void appFace::removeBrow(Mat eyeMask,Rect r){ // �趨�۾�ROI����1/6,һ�������üë�򲻰����۾���
	// �����1/6�����е�һ����ͨ��
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

//ȡ�۾�ģ������ҵ㡣lr:0ȡ���۰׵㣬1ȡ���۰׵㣬2ȡ���۱�Ե��3ȡ���۱�Ե��4ȡ�۰��ϱ�Ե��5ȡ���ϱ�Ե��
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


//��ģ�尴ָ�����������ת�����š�
void appFace::resizeModel(Mat modelImg,Point ml,Point mr,Point mt,Point el,Point er,Point et){
	double arcS;
	int xmlength = mr.x - ml.x; // TODL: mr.xֵ����
	int ymlength = mr.y - ml.y; // ģ�����ҵ�Yֵ��
	int xelength = er.x - el.x;
	int yelength = er.y - el.y;

	//* ���Ŵ�����һ��û�п����۾����㣬ֱ�������������ţ��۾��ᱻ���·Ŵ�
	bool yReverse = false;
	if((ymlength > 0 && yelength < 0) || (ymlength < 0 && yelength > 0)) yReverse = true;
	cout << xelength << " " << xmlength << " " << yelength << " " << ymlength << endl;
	resize(modelImg,modelImg,Size(0,0),xelength/xmlength,yelength/ymlength);
	if(yReverse) flip(modelImg,modelImg,0);
	//*/

	/*//*���Ŵ�����������ת�������š�
	// TODO: ����ģ��ת��ˮƽ�������ţ�����ת���۾��Ƕȣ����������ţ���ģ�嶥�����۾�������ͬ�߶ȣ���ʱ���۾�ģ�����¸߶���һ������ֵ��
	arcS = atan2((double)modelImg.rows,(double)modelImg.cols);
	arcS = arcS*180/CV_PI; // ����ģ�ͶԽ���ԭʼ�Ƕ�
	
	//��ת��ˮƽ

	//�������ţ����۾������ҵ㳤�ȡ�

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

