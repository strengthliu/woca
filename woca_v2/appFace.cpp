#include "appFace.h"
#include "xmlParser.h"
#include <iostream>
#include <stdio.h>  
#include <math.h>
#include <vector>
#define M_PI       3.14159265358979323846
using namespace std;
//using namespace cv;

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
	int _fl = CFace::calFirstColOfContour(imageFaceContourSM);
	int _flr = CFace::calFirstColOfContour_Row(imageFaceContourSM);
	//��ԵС��ȫ����1/8�������ǽ��ŵġ�
	int w = (CFace::calLastColOfContour(imageFaceContourSM)-CFace::calFirstColOfContour(imageFaceContourSM))/8;
	int wt = 0;
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
	int _fl = CFace::calLastColOfContour(imageFaceContourSM);
	int _flr = CFace::calLastColOfContour_Row(imageFaceContourSM);
	//��ԵС��ȫ����1/8�������ǽ��ŵġ�
	int w = (CFace::calLastColOfContour(imageFaceContourSM)-CFace::calFirstColOfContour(imageFaceContourSM))/8;
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

int appFace::calFirstRowOfContourHuman(){//��дΪ�����·��������ֲ�̫�ȶ����ȸĻ�ȥ�á�
	for (int _row = 0; _row < imageContourSM.rows ; _row++)
	{
		uchar* rowData = imageContourSM.ptr<uchar>(_row);
		for (int _col = 0 ; _col < imageContourSM.cols ;_col++)
		{
			if (rowData[_col] > 0)
				return _row;
		}
	}
	return -1;
}

vector<Rect> appFace::detectEyes(Mat _face){//�����м���۾�
		//��ʼ����۾���
	Mat faceROI = _face;//frame_gray( faceDetectedRect );
	std::vector<Rect> eyes;
	int minSize = faceROI.rows / 5;
	for(int t=0;t<3;t++){
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));
		if(eyes.size()!=2) continue;
		else t=3;
	}

	if(eyes.size()>2){	
		eyes.erase(eyes.begin()+2,eyes.end());//ɾ������[2,��β];�����0��ʼ
	}
	else if(eyes.size()>0)
		return eyes;
	else
		cout << " û�м�⵽�۾��� " << endl;

	return eyes;
}

void appFace::setEyesParameters(vector<Rect> __eyes){
	//	cout << " ��⵽�۾�����ʼ���ò���������" << endl;
	eyeNumber = __eyes.size();
	int leftN,rightN;
	if(eyeNumber == 2){
		if(__eyes[0].x+__eyes[0].width > __eyes[1].x+__eyes[1].width/2){
			leftN = 1;rightN = 0;
		}else {
			leftN = 0;rightN = 1;
		}
		this->eyeDetectedRects[0] = Rect(faces[0].x + __eyes[leftN].x, faces[0].y + __eyes[leftN].y, __eyes[leftN].width, __eyes[leftN].height);
		this->eyeDetectedRects[1] = Rect(faces[0].x + __eyes[rightN].x, faces[0].y + __eyes[rightN].y, __eyes[rightN].width, __eyes[rightN].height);
		if(eyeDetectedRects[0].y == eyeDetectedRects[1].y) {
			cout << " ���ۼ���߶�һ�����߶ȷֱ�Ϊ��" << eyeDetectedRects[0].height << ", " << eyeDetectedRects[1].height << endl;
			cout << "" << endl;
		} else {
			cout << " ���ۼ���߶Ȳ�һ�����߶ȷֱ�Ϊ��" << eyeDetectedRects[0].y << ", " << eyeDetectedRects[0].height << " | " << eyeDetectedRects[1].y << ", " << eyeDetectedRects[1].height << endl;
			cout << "" << endl;
		}

		Mat roi ;
		imageOrigine(eyeDetectedRects[0]).copyTo(roi);
		//=======================================================================
		mask1 = CFace::createROI(roi,"eyeDetectedRects[0]1",1,2,3); // ��ֵ�㷨 1/3�뾶
		mask2 = CFace::createROI(roi,"eyeDetectedRects[0]2",0,3,3); // ��ֵ�㷨 1/3�뾶

		imwrite("MedianData//lineLeft.png", lineImage(eyeDetectedRects[0]));
		imwrite("MedianData//lineRight.png", lineImage(eyeDetectedRects[1]));

	Mat cannyGray,_gray;
	cvtColor(roi,_gray,CV_BGR2GRAY);
	_gray.copyTo(cannyGray);
	//equalizeHist( cannyGray, cannyGray );
	IplImage* imageCvThreshold = &IplImage(roi);
	double numGray =  CFace::get_avg_gray(imageCvThreshold);
	Canny(cannyGray, cannyGray, numGray, numGray,3,true);
	CFace::reverseROI(cannyGray);
	imwrite("MedianData//eyeDetectedRects[0]22.jpg", cannyGray);

	cout << eyeDetectedRects[0] << endl;
	cout << eyeDetectedRects[1] << endl;
		mask6 = CFace::createROI(roi,"eyeDetectedRects[0]6",0,2,3); // ��ֵ�㷨 1/3�뾶
		mask7 = CFace::createROI(roi,"eyeDetectedRects[0]7",0,2,roi.rows/5); // ��ֵ�㷨 5�뾶
		//=======================================================================
		Rect rBrow = Rect(mask1.cols/3,mask1.rows/3,mask1.cols/3,mask1.rows/3);
		Mat mask1tmp;mask1.copyTo(mask1tmp);
		CFace::removeBrow(mask1,rBrow);//MASKȥ��üë����
		mask1=mask1tmp-mask1;
		imwrite("MedianData//eyeDetectedRects[0]1-1.png",mask1);
		CFace::filterBlock(mask7,mask1,true); // ����ϸ��MASKͼ
		imwrite("MedianData//eyeDetectedRects[0]7-1.png",mask7);
		CFace::filterBlock(mask6,mask1,true); // ���˴���MASKͼ
		imwrite("MedianData//eyeDetectedRects[0]6-1.png",mask7);

		imageOrigine(eyeDetectedRects[1]).copyTo(roi);
		//=======================================================================
		imwrite("MedianData//roi11.png",roi);
		mask11 = CFace::createROI(roi,"eyeDetectedRects[1]1",1,2,3); // ��ֵ�㷨 1/3�뾶
		mask61 = CFace::createROI(roi,"eyeDetectedRects[1]6",0,2,3); // ��ֵ�㷨 1/3�뾶
		mask71 = CFace::createROI(roi,"eyeDetectedRects[1]7",0,2,roi.rows/5); // ��ֵ�㷨 5�뾶
		//=======================================================================
		rBrow = Rect(mask11.cols/3,mask11.rows/3,mask11.cols/3,mask11.rows/3);
		mask11.copyTo(mask1tmp);
		CFace::removeBrow(mask11,rBrow);//MASKȥ��üë����
		mask11=mask1tmp-mask11;
		CFace::filterBlock(mask71,mask11,true); // ����ϸ��MASKͼ
		CFace::filterBlock(mask61,mask11,true); // ���˴���MASKͼ
		imwrite("MedianData//eyeDRects[0]6.png",mask6);
		imwrite("MedianData//eyeDRects[0]7.png",mask7);
		imwrite("MedianData//eyeDRects[0]1.png",mask1);
		imwrite("MedianData//eyeDRects[1]71.png",mask71);
		imwrite("MedianData//eyeDRects[1]61.png",mask61);
		imwrite("MedianData//eyeDRects[1]1.png",mask11);
		Mat m7,m71;mask7.copyTo(m7);mask71.copyTo(m71);
		getEyePoint(mask6,mask7,mask71,mask61,m7,m71); // ���������۵�ROI������㡣

		//=========================����ȡ������==================================
		//imageOrigine(eyeDetectedRects[0]).copyTo(roi);
		Mat testMASK,testMASK1,em11,em12,em13,em14,em15;
		this->dtLeftEyeRect = Rect(eyesPoint[2].x,eyesPoint[0].y,eyesPoint[3].x-eyesPoint[2].x,eyesPoint[1].y-eyesPoint[0].y);
		imageOrigine(eyeDetectedRects[0]).copyTo(roi);
		testMASK1 = roi(dtLeftEyeRect);
		Mat src_gray,grad;
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;
		GaussianBlur( testMASK1, testMASK1, Size(3,3), 0, 0, BORDER_DEFAULT );
		cvtColor( testMASK1, src_gray, CV_BGR2GRAY );
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
		Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		convertScaleAbs( grad_x, abs_grad_x );
		Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		convertScaleAbs( grad_y, abs_grad_y );
		addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
		imwrite("MedianData//edr1.png",grad);
		imwrite("MedianData//edr2.png",testMASK1);

		em11 = CFace::createROI(testMASK1,"ED11",1,2,3); // ��ֵ�㷨 1/3�뾶
		em15 = CFace::createROI(testMASK1,"ED15",0,3,3); // ��ֵ�㷨 1/3�뾶
		em12 = CFace::createROI(testMASK1,"ED12",0,2,3); // ��ֵ�㷨 1/3�뾶
		em13 = CFace::createROI(testMASK1,"ED13",0,2,roi.rows/5); // ��ֵ�㷨 1/3�뾶
		//----------------------------------------
		em14 = CFace::createROI(testMASK1,"ED14",1,1,128); // ��ֵ�㷨 1/3�뾶cvAvg
		//������������ֱ���֣��������ȱ仯��ȡ���������ұ߽硣
		IplImage *src=&IplImage(em14);
		IplImage* paintx=cvCreateImage( cvGetSize(src),IPL_DEPTH_8U, 1 );  
		IplImage* painty=cvCreateImage( cvGetSize(src),IPL_DEPTH_8U, 1 );  
		cvZero(painty);  
		cvZero(paintx);  
		int* v=new int[mask11.cols];  
		int* h=new int[mask11.rows];  
		memset(v,0,mask11.cols*4);  
		memset(h,0,mask11.rows*4);  
		int x,y; CvScalar s,t;
        for(x=0;x<src->width;x++)
        {
                for(y=0;y<src->height;y++)
                {
                        s=cvGet2D(src,y,x);                        
                        if(s.val[0]==0)
                                v[x]++;                                        
                }                
        }
        for(x=0;x<src->width;x++)
        {
                for(y=0;y<v[x];y++)
                {                
                        t.val[0]=255;
                        cvSet2D(paintx,y,x,t);                
                }                
		}

        for(y=0;y<src->height;y++)
        {
                for(x=0;x<src->width;x++)
                {
                        s=cvGet2D(src,y,x);                        
                        if(s.val[0]==0)
                                h[y]++;                
                }        
        }
        for(y=0;y<src->width;y++)
        {
                for(x=0;x<h[y];x++)
                {                        
                        t.val[0]=255;
                        
                        cvSet2D(painty,y,x,t);                        
                }                
        }
		imwrite("MedianData//ED14-1.png",Mat(painty));
		imwrite("MedianData//ED14-2.png",Mat(paintx));
		cvReleaseImage(&painty);  
		cvReleaseImage(&paintx);  

		/*
		cout << eyeDetectedRects[1] << endl;
		//----------------------------------------
		Mat testMASK2,testMASK21;
		this->dtRightEyeRect = Rect(eyesPoint[6].x,eyesPoint[4].y,eyesPoint[7].x-eyesPoint[6].x,eyesPoint[5].y-eyesPoint[4].y);
		cout << dtRightEyeRect << endl;
		imageOrigine(eyeDetectedRects[1]).copyTo(roi);
		testMASK21 = roi;//(dtRightEyeRect);
		imwrite("MedianData//testMASK21.png",testMASK21);
		mask11 = CFace::createROI(testMASK21,"ED21",1,2,3); // ��ֵ�㷨 1/3�뾶
		mask11 = CFace::createROI(testMASK21,"ED22",0,2,3); // ��ֵ�㷨 1/3�뾶
		mask11 = CFace::createROI(testMASK21,"ED23",0,2,roi.rows/5); // ��ֵ�㷨 1/3�뾶
		mask11 = CFace::createROI(testMASK21,"ED24",1,1,128); // ��ֵ�㷨 1/3�뾶
		*/
		//=========================������Ҫȡüë�������ò�����==================================
		//���趨üë�ļ������
		this->browDetectedRects[0] = Rect(eyeDetectedRects[0].x-eyeDetectedRects[0].width/8,
			eyeDetectedRects[0].y-eyeDetectedRects[0].height/4,
			eyeDetectedRects[0].width*1.25,
			eyeDetectedRects[0].height*0.5);
		this->browDetectedRects[1] = Rect(eyeDetectedRects[1].x-eyeDetectedRects[1].width/8,
			eyeDetectedRects[1].y-eyeDetectedRects[1].height/4,
			eyeDetectedRects[1].width*1.25,
			eyeDetectedRects[1].height*0.5);
		
		Mat broi ;
		imageOrigine(browDetectedRects[0]).copyTo(broi);
		//=======================================================================
		bmask1 = CFace::createROI(broi,"browDetectedRects[0]1",1,2,3); // ��ֵ�㷨 1/3�뾶
		//=======================================================================
		Rect lBrow = Rect(0,bmask1.rows/4,bmask1.cols*3/4,bmask1.rows/2);
		bmask1.copyTo(bmask2);
		CFace::removeBrow(bmask1,lBrow);//MASKȥ��üë����
		bmask1 = bmask2 - bmask1;
		imwrite("MedianData//browDetectedRects[0]71.png",bmask1);
		
		Mat broi1 ;
		imageOrigine(browDetectedRects[1]).copyTo(broi1);
		//=======================================================================
		bmask11 = CFace::createROI(broi1,"browDetectedRects[1]1",1,2,3); // ��ֵ�㷨 1/3�뾶
		//=======================================================================
		rBrow = Rect(bmask11.cols/4,bmask11.rows/4,bmask11.cols*3/4,bmask11.rows/2);
		bmask11.copyTo(bmask21);
		CFace::removeBrow(bmask11,rBrow);//MASKȥ��üë����
		bmask11 = bmask21 - bmask11;
		imwrite("MedianData//browDetectedRects[1]71.png",bmask11);
		
		getBrowPoint(bmask1,bmask11); // ���������۵�ROI������㡣
		cout << "" << endl;

		//==========================================================================

	for(int i=0;i<8;i++){
		Point ep = browPoint[i];
		cout << "��" << i << "���㣬���꣺" << ep.x <<  "," << ep.y << endl;
	}

	} else 
	{
		for(int i=0;i<eyeNumber;i++){
			eyeDetectedRects[i] = Rect(faceDetectedRect.x+__eyes[i].x,faceDetectedRect.y+__eyes[i].y,__eyes[i].width,__eyes[i].height);
		}
	}
}
void appFace::setMouthsParameter(Vector<Rect> mouths){
		cout << " ��⵽�죬��ʼ���ò���������" << endl;

	//======================================= ��֤���Ƿ���ȷ ===============================================
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
	//===================================================================================================
	//�ҵ���һ����
	if(_ym>=0){
		this->mouthDetectedRect = Rect(this->faceDetectedRect.x+mouths[_ym].x,this->faceDetectedRect.y+mouths[_ym].y,mouths[_ym].width,mouths[_ym].height);


	} else {
		cout << " û���ҵ���ȷ���� " << endl;
	}
}
void appFace::setMouthsParameter(Vector<Rect> mouths,Rect mouthRegion){
		cout << " ��⵽�죬��ʼ���ò���������" << endl;
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


	/* ��ɫģ��
            y = img_ycbcr(i,j,1);
            cb = img_ycbcr(i,j,2);
            cr = img_ycbcr(i,j,3);
            
            if(y > 70 && cb > 100 && cr > 163)
                if(y < 130 && cb < 150 && cr < 180)
                   final_image(i,j)=1;
                end
            end
	*/
	Mat bmask11,bmask21;
	Mat broi1 ;
	imageOrigine(mouthDetectedRect).copyTo(broi1);
	//=======================================================================
	bmask11 = CFace::createROI(broi1,"mouth1",1,2,3); // ��ֵ�㷨 1/3�뾶
	//=======================================================================
	Rect rBrow = Rect(bmask11.cols/4,bmask11.rows/4,bmask11.cols*3/4,bmask11.rows/2);
	bmask11.copyTo(bmask21);
	CFace::removeBrow(bmask11,rBrow);//MASKȥ��üë����
	bmask11 = bmask21 - bmask11;
	imwrite("MedianData//mouth11.png",bmask11);


	getMouthPoint(bmask11); // �������ROI������㡣


}

void appFace::setNoseParameter(Vector<Rect> noses){

	//==========================================  �������Ƿ���ȷ  =====================================================
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
	//==========================================================================================================
	if(_yn>=0){
		this->noseDetectedRect = Rect(this->faceDetectedRect.x+noses[_yn].x,this->faceDetectedRect.y+noses[_yn].y,noses[_yn].width,noses[_yn].height);
	} else {
		cout << " û���ҵ����ӡ� " << endl;
	}
}
void appFace::setNoseParameter(Vector<Rect> noses,Rect noseRegion){
		cout << " ��⵽���ӣ���ʼ���ò���������" << endl;

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
	else {
		cout << " û���ҵ����ӡ� " << endl;
		return;
	}

		Mat roi;
		imageOrigine(noseDetectedRect).copyTo(roi);
		Mat mask7,mask1,mask2,mask3,mask4,mask5,mask6,mask8,mask9,mask10,mask11,mask12;
		//=======================================================================
		//bmask1 = createROI(broi,"browDetectedRects[0]1",1,2,3); // ��ֵ�㷨 1/3�뾶
		//createROI(Mat m,string name,int pre,int mode,int range){ 
		// mԴͼ��pre��Ԥ����0���ޣ�1��ֱ��ͼ��mode��1��cvThreshold��2��cvAdaptiveThreshold��3��Canny��Ե��⡣
		//mask1 = CFace::createROI(roi,"nose1",1,1,3); // ��ֵ�㷨 1/3�뾶
		mask2 = CFace::createROI(roi,"nose2",1,2,3); // ��ֵ�㷨 1/3�뾶
		//mask3 = CFace::createROI(roi,"nose3",1,3,30); // ��ֵ�㷨 5�뾶
		//mask4 = CFace::createROI(roi,"nose4",1,1,roi.rows/5); // ��ֵ�㷨 1/3�뾶
		//mask5 = CFace::createROI(roi,"nose5",1,2,roi.rows/5); // ��ֵ�㷨 1/3�뾶
		//mask6 = CFace::createROI(roi,"nose6",1,3,roi.rows/3); // ��ֵ�㷨 5�뾶

		//=======================================================================
		/*
		Rect rBrow = Rect(0,0,mask1.cols,mask1.rows/4);
		imwrite("MedianData//eyeDetectedRects[0]10.png",mask1);
		removeBrow(mask1,rBrow);//MASKȥ��üë����
		filterBlock(mask7,mask1,true); // ����ϸ��MASKͼ
		filterBlock(mask6,mask1,true); // ���˴���MASKͼ
		imwrite("MedianData//eyeDetectedRects[0]71.png",mask7);
		*/
		getNosePoint(mask2);
		cout << " " << endl;
}

void appFace::setEyesParameters(vector<Rect> __eyes, Rect faces){
	setEyesParameters(__eyes);
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
		for(int t=0;t<3;t++){
			mouth_cascade.detectMultiScale( faceROI, mouths, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize1*3, minSize1));
			if(mouths.size()==1) t=3;
			else continue;
		}
		return mouths;
}

vector<Rect> appFace::detectNose(Mat _face){//�����м�����

		//�����������
		Mat faceROI = _face;//frame_gray( faceDetectedRect );
		std::vector<Rect> noses;
		//�������ֵӰ��޴�
		int minSize2 = faceROI.rows / 8;
		for(int t=0;t<3;t++){
			nose_cascade.detectMultiScale( faceROI, noses, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize2, minSize2));
			if(noses.size()==1) t=3;
			else continue;
		}
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
	int angleDegree = (45)/angle;
	for(int i_angle=1;i_angle<angleDegree;i_angle++){ //��1�ȿ�ʼ
		Mat rotateImg;  
		imwrite("MedianData//simpleFaceDetectionb.png",frame);
		
		//if(direct == 2)
			rotateImg = CFace::rotate(frame,i_angle*angle);
		//if(direct == 1)
		//	rotateImg = rotate(frame,(360-i_angle*angle));
		
		Mat frame_gray;
		cvtColor( rotateImg, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );
		
		//-- Detect faces
		cout << " ��ת�����¼����һ�Ρ�����" << endl;
		Rect maskFaceRect;
		for(int t=0;t<3;t++){
			face_cascade.detectMultiScale( frame_gray, _faces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
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
			}else{
				t=3;
				break;
			}
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
				if(leftEyeMiddleY > rightEyeMiddleY && angle > 0 && i_angle == 1) angle = -1 * angle;
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


void appFace::rotate(){
//	maskContourHuman
	if(imageOrigine.cols)
	imageOrigine = CFace::rotate(imageOrigine,this->rotateAngle);
	//imwrite("MedianData//imageOrigine.png",imageOrigine);
	if(imageContourSM.cols)
	imageContourSM = CFace::rotate(imageContourSM,this->rotateAngle);
	//imwrite("MedianData//imageContourSM.png",imageContourSM);
	if(imageFaceContourSM.cols)
	imageFaceContourSM = CFace::rotate(imageFaceContourSM,this->rotateAngle);
	//imwrite("MedianData//imageFaceContourSM.png",imageFaceContourSM);
//	imageHairContourSM = rotate(imageHairContourSM,this->rotateAngle); 
//	imageRealHairContourSM = rotate(imageRealHairContourSM,this->rotateAngle);
	if(imageRealFaceContourSM.cols)
	imageRealFaceContourSM = CFace::rotate(imageRealFaceContourSM,this->rotateAngle);
	//imwrite("MedianData//imageRealFaceContourSM.png",imageRealFaceContourSM);
	if(imageRealContourSM.cols)
	imageRealContourSM = CFace::rotate(imageRealContourSM,this->rotateAngle);
	//imwrite("MedianData//imageRealContourSM.png",imageRealContourSM);

	if(maskFace.cols)
		maskFace = CFace::rotate(maskFace,this->rotateAngle);
	if(maskFaceReplace.cols)
		maskFace = CFace::rotate(maskFaceReplace,this->rotateAngle);
	if(maskRealFace.cols)
		maskFace = CFace::rotate(maskRealFace,this->rotateAngle);
	if(maskContourHuman.cols)
		maskFace = CFace::rotate(maskContourHuman,this->rotateAngle);


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

	int e = CFace::calLastRowOfContour(this->imageFaceContourSM);
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
		frame = CFace::rotate(frame,this->rotateAngle); // ��תframe��׼������������١�
		imwrite("MedianData//simpleFaceDetection1.png",frame);
		frame.copyTo(debugFrame);//��ʼ��debugFrame.

		setFaceParameters(frame);//�������Ĳ���
		debugFace();

	//
	Mat grayDst;
	cvtColor(frame,grayDst,CV_BGR2GRAY);
	IplImage* imagecvAdaptiveThresholdGray = &IplImage(grayDst);
	cvAdaptiveThreshold(imagecvAdaptiveThresholdGray, imagecvAdaptiveThresholdGray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 3);
	CFace::reverseROI(grayDst);
	CFace::absROI(grayDst);
	//CFace::filterBlock(grayDst,5,5,255);
	imwrite("ResultData//imageOrigine_pre1.jpg", grayDst);
	grayDst.copyTo(lineImage);



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
				//createROI(roi,"nose");

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

		int e = CFace::calLastRowOfContour(this->imageFaceContourSM);
		if(e>0)
			rowEnd = e;
//		cout <<  e << endl;

		this->faceChangeRect = Rect(colBegin, rowBegin, colEnd - colBegin, rowEnd - rowBegin);

		cout << "simpleFaceDetection => faceChangeRect.x:" << faceChangeRect.x << " faceChangeRect.y:" << faceChangeRect.y << " faceChangeRect.width:" << faceChangeRect.width << " faceChangeRect.height:" << faceChangeRect.height << endl;
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
	//Ϊ�˼������ȰѸ��Ӽ��ע�͵���
	simpleFaceDetection1();
	//simpleFaceDetection();
	//colorBasedFaceDetection();//���ʵ�ڲ�֪����ʲô�ô���ע�͵��ȡ�
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
	CFace::filterMirrorBackground(resultImage);
	hsvCompromise(faceModel);
	replaceFaceAndEyes(faceModel, leftEyeModel, rightEyeModel,leftEyeWithBrowModel,rightEyeWithBrowModel, resultImage,leftEyePupilModel,rightEyePupilModel,mouthModel,noseModel,mode);
	//saveImages(resultImage);
}
void appFace::FaceChange(int isHair){
	Mat resultImage;
	imageOrigine.copyTo(resultImage);
	//insideComponent2(imageRealContourSM, 30);
	CFace::filterMirrorBackground(resultImage);
	Mat faceModel = chm.currentHead.faceModel;
	//hsvCompromise(faceModel);
	replaceFaceAndEyes( resultImage,isHair);
	//saveImages(resultImage);
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
	//Mat frame;

	faceModel = chm.currentHead.faceModel;
	leftEyeModel = chm.currentExpression.leftEye.model;
	rightEyeModel = chm.currentExpression.rightEye.model;
	leftEyeWithBrowModel = chm.currentExpression.leftEyeBrow;
	rightEyeWithBrowModel = chm.currentExpression.rightEyeBrow;
	leftBrowModel = chm.currentExpression.leftBrow.model;
	rightBrowModel = chm.currentExpression.rightBrow.model;
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
		CFace::absROI(imageFaceContourSM);
		imageFaceContourSM.copyTo(maskFace);
		insideComponent(maskFace);
		imwrite("MedianData//maskFace.png", maskFace);
	}

	if (imageHairContourSM.cols)
	{
		CFace::absROI(imageHairContourSM);
		imageHairContourSM.copyTo(maskHair);
		imageRealHairContourSM.copyTo(maskRealHair);
		insideComponent(maskHair);
		//insideComponent(maskRealHair);
		imwrite("MedianData//maskHair.png", maskHair);
		imwrite("MedianData//maskReal.png",maskRealHair);
	}

	if (imageRealFaceContourSM.cols)
	{
		CFace::absROI(imageRealFaceContourSM);
		imageRealFaceContourSM.copyTo(maskRealFace);
	}

	maskFace.copyTo(maskFaceReplace);
	maskHair.copyTo(maskHairReplace);
}

void appFace::calFaceParameters(){}

//���ݲ�ͬģʽ��������ģ�塣
void appFace::resizeFaceModel(int mode){
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
		fWidth = ((double)  faceWidth) / (double)faceModel.cols;
		fHeight = ((double) faceChangeRect.height) / (double)faceModel.rows;
		//�������������Ŀ�ȣ����޸�faceChangeRect
		faceChangeRect = Rect(faceChangeRect.x+(faceChangeRect.width-faceWidth)/2,faceChangeRect.y,faceChangeRect.width,faceChangeRect.height);
	}
	else{ // TODO: ������൲�������ƽ�ָ���������
		faceWidth = faceModel.cols *  ((double)faceChangeRect.height / (double)faceModel.rows);
		fWidth = (double)  faceWidth / faceModel.cols;
		fHeight = (double) faceChangeRect.height / faceModel.rows;
		cout << "resizeFaceModel=> fWidth:" << fWidth << " fHeight:" << fHeight << endl;
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
	cout << "faceWidth:" << faceWidth << " fWidth:" << fWidth << " fHeight:" << fHeight << endl;
	faceSampleRect = Rect((faceMiddleRect.x-faceWidth/2),(faceChangeRect.y+faceChangeRect.height-faceSample.rows),(faceSample.cols),(faceSample.rows));
}
Mat appFace::replaceFace(Mat faceModel,Mat &resultImage,int mode){
	Mat ret;
	if(mode == REALMODEPLUS){
		ret = replaceFaceByREALMODEPLUS(faceModel,resultImage);
		return ret;
	}

	if(mode == DRAWMODE){ // ת�ֻ档

	}
	 return replaceFaceByDefault(faceModel,resultImage);
}


//��resultImageͼ�У���modeģʽ����������faceModel��
Mat appFace::replaceFaceByREALMODEPLUS(Mat faceModel,Mat &resultImage){
	imwrite("ResultData//OrigionFace_t1.png",resultImage);
	Mat frame;resultImage.copyTo(frame);
	//if(true) return frame;//Ϊ�˲����黯�������ģ�巽��������ֱ�ӷ����ˡ�
	imwrite("ResultData//OrigionFace_t2.png",frame);
	Vec4b *bgra_frame_data = frame.ptr<Vec4b>(0);

	//=======================================  ��ͼЧ���������ƽ��  ===================================
	Mat imgSmooth;resultImage.copyTo(imgSmooth);
	IplImage* _bodyWithoutBackground = &IplImage(imgSmooth);
	for(int times=0;times<40;times++){
		cvSmooth(_bodyWithoutBackground,_bodyWithoutBackground);
	}
	Vec4b *smooth_bgra_frame_data = imgSmooth.ptr<Vec4b>(0);
	//imgSmooth.copyTo(_bgraFrame);
	imwrite("ResultData//_smoothBodyWithoutBackground_2.jpg", imgSmooth);

	//-- Change face

	imwrite("ResultData//faceSampleBGRA.png",faceModel);
	if(faceSample.channels() == 4) {
		cvtColor(faceSample, faceSampleBGR, CV_BGRA2BGR);
	}else
		cout << "" << endl;
	imwrite("ResultData//faceSampleBGR.png",faceSampleBGR);
		
	//cvSplit(hsv_img, h_img, s_img, v_img, NULL);  	
	//Mat _bgrFrameSkin,_skinMask;
	//cvtColor(frame, _skinMask, CV_BGR2BGRA);
	
	
	uchar *maskData = maskFace.ptr<uchar>(0);
	uchar *maskHairData = maskHair.ptr<uchar>(0);
	uchar *maskRealHairData = maskRealHair.ptr<uchar>(0);

	//�����ϵ���ʾ����
	uchar* mask_face_replace_data = maskFaceReplace.ptr<uchar>(0);
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);

	//����ģ��ӿ��ˣ����ܴ�����ƥ�䣬��Ӧ�ô��м���롣��Ҫ��faceChangeRect�����Ƶ����ߡ�
	int middle = -1*(this->faceMiddleRect.x - faceChangeRect.x-faceChangeRect.width/2) +  faceSample.cols/2 - faceChangeRect.width/2;
	

	//����������
	//imwrite("MedianData//bgrLightBeforeChangeFaceNoTransparent.png", _bgraFrameLight);
	//	imwrite("MedianData//bgrTemp1.png", _bgraFrameLight);
	//changeFace(frame,mask_face_replace_data,faceSample);
	//imwrite("ResultData//OrigionFace_t3.png",resultImage);
	//imwrite("MedianData//bgrLightAfterChangeFaceNoTransparent.png", _bgraFrameLight);

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
			
			if (colDataBGRA[_col][3] == 0){ // ���ģ����û�����ݾͽ�������ط���Ϊ0���൱���������ˣ����ǻ�û�аѱ������졣
				//mask_face_replace_data[index] = 0;//�����޸���ԭMASK����
				bgra_frame_data[index] = Vec4b(0,0,0,0);
				continue;
			}
			
			//
			if (mask_face_replace_data[index] == 255)
			{
				//�仯������
				Vec3b vf_hsv,vf_BGR,vb_hsv,vb_BGR,cartoon_vb_BGR,cartoon_vb_hsv,vb_smooth_BGR,vb_smooth_hsv;
				vf_BGR = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
				vb_BGR = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
				vb_smooth_BGR = Vec3b(smooth_bgra_frame_data[index][0],smooth_bgra_frame_data[index][1],smooth_bgra_frame_data[index][2]);

				/*
				if(vb_BGR[0]<200) vb_BGR[0]=200;
				if(vb_BGR[1]<200) vb_BGR[1]=200;
				if(vb_BGR[2]<200) vb_BGR[2]=200;
				*/
				vf_hsv = CFace::kcvBGR2HSV(vf_BGR);
				vb_hsv = CFace::kcvBGR2HSV(vb_BGR);
				vb_smooth_hsv = CFace::kcvBGR2HSV(vb_smooth_BGR);
				//if(vb_hsv[2]<200)//ȥ���������Ե�����
				{
					vf_hsv[0] = vb_hsv[0];
					//��������ģ����Ͼ��Ե����ˣ����ࡣ
					vf_hsv[1] = vb_smooth_hsv[1];//+vb_hsv[1]*0.3; 
					vf_hsv[2] = vf_hsv[2];//+vb_hsv[2]*0.3;
				}
				vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
				//�����ģ����ͬλ�ò�͸����˵���������ݵ�
				if(colDataBGRA[_col][3]>5){
					double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
					//bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
					bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
				} else {
					//�����ģ����ͬλ��͸����˵���������Ĳ��֣�Ҫ��ȥ��

				}
				continue;
			}

			//FACEMASK���⣬����realFace����
			if(mask_real_face_data[index] < 32){
				continue;
			}
			//���������ΧΪ��ɫ��ֱ��ȡ��ɫֵ��
			if (mask_real_face_data[index] > 223){
					double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
				bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],(1-rate)*mask_real_face_data[index]);
			}
			// �����������ܱ����Ӱ�͸�����������ˡ�Ч�����á�
			else { //����Ҫ��͸��������
				//�仯������
				Vec3b vf_BGR = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
				//����ȡ���ǰ������ĵ����ݡ����������ģ��������⡣
				Vec3b vb_BGR = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
				Vec3b vf_hsv = CFace::kcvBGR2HSV(vf_BGR);
				Vec3b vb_hsv = CFace::kcvBGR2HSV(vb_BGR);
					
				vf_hsv[0] = vb_hsv[0];//ֻȡɫ��ֵ
				//vf_hsv[1] = vb_hsv[1];
				vf_hsv[2] = vb_hsv[2];
					
				vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
				double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
				//bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], 255);
				//bgra_frame_light_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], 255);
				bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
			}					
		}
	}

	//rectangle(_bgraFrame, this->faceDetectedRect, Scalar(0,0,255));
	//rectangle(_bgraFrame, this->faceMiddleRect, Scalar(0,0,255));

	//imwrite("MedianData//bgrLightTemp.png", _bgraFrameLight);
	//imwrite("MedianData//bgrTemp.png", _bgraFrame);

	return frame;
}
Mat appFace::replaceFaceByDefault(Mat faceModel,Mat &resultImage){

	Mat frame;resultImage.copyTo(frame);
	//-- Change face
	if(faceSampleBGR.channels() == 4) cvtColor(faceSample, faceSampleBGR, CV_BGRA2BGR);

	Mat _bgrFrameSkin,_skinMask;
	Mat _bgraFrame,_bgraFrameLight;
	//������Ҫ����
	//_bgraFrameLight = _bgraFrame.copyTo(
	if(frame.channels() <4) {
		cvtColor(frame, _bgraFrame, CV_BGR2BGRA);
		cvtColor(frame, _bgraFrameLight, CV_BGR2BGRA);
	}
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
	double gFace = CFace::get_avg_gray(imageFace);
	double gBgr = CFace::get_avg_gray(imageBgr);
	//Ϊ�˵�����������ע�͵���
	CFace::set_avg_gray(imageBgr,imageBgr,gFace*0.9);

	//��ɫ����
	CvSize imageSize = cvSize(imageBgrSkin->width, imageBgrSkin->height);
	IplImage *imageSkin = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
	
	CFace::cvSkinSegment(imageBgrSkin,imageSkin);
	//cvSkinYUV(imageBgrSkin,imageSkin);
	//cvSkinHSV(imageBgrSkin,imageSkin);
	Mat skinMat= Mat(imageSkin);

	imwrite("MedianData//skinTemp.png", skinMat);
	imwrite("MedianData//faceTemp.png", faceSampleBGR);
	//д�������Ⱥ���ļ�
	imwrite("MedianData//bgrLight.png", _bgraFrameLight);

	//��������
	imwrite("MedianData//bgrLightBeforeChangeFaceNoTransparent.png", _bgraFrameLight);
	changeFace(_bgraFrameLight,mask_face_replace_data,faceSample);
	imwrite("MedianData//bgrLightAfterChangeFaceNoTransparent.png", _bgraFrameLight);
	// ���￪ʼ��ģʽ����
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
			
			if(true)
			{
				if (mask_face_replace_data[index] == 255)
				{
					//�仯������
					Vec3b vf_hsv,vf_BGR,vb_hsv,vb_BGR,cartoon_vb_BGR,cartoon_vb_hsv;
					vf_BGR = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
					//vb_BGR = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
					vb_BGR = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
					//cartoon_vb_BGR = Vec3b(cartoonBgra_frame_data[index][0],cartoonBgra_frame_data[index][1],cartoonBgra_frame_data[index][2]);
					/*
					if(vb_BGR[0]<200) vb_BGR[0]=200;
					if(vb_BGR[1]<200) vb_BGR[1]=200;
					if(vb_BGR[2]<200) vb_BGR[2]=200;
					*/
					vf_hsv = CFace::kcvBGR2HSV(vf_BGR);
					vb_hsv = CFace::kcvBGR2HSV(vb_BGR);
					//cartoon_vb_hsv = kcvBGR2HSV(cartoon_vb_BGR);
					//ȥ���������Ե�����
					//if(vb_hsv[2]<200)
					{
						vf_hsv[0] = vb_hsv[0];
						//��������ģ����Ͼ��Ե����ˣ����ࡣ
						//vf_hsv[1] = vf_hsv[1]+vb_hsv[1]*0.3;N

						//vf_hsv[2] = vf_hsv[2]+vb_hsv[2]*0.3;
					}
					vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
					//�����ģ����ͬλ�ò�͸����˵���������ݵ�
					if(colDataBGRA[_col][3]>5){
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
						bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
						bgra_frame_light_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
					} else {
						//�����ģ����ͬλ��͸����˵���������Ĳ��֣�Ҫ��ȥ��

					}
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
					Vec3b vf_BGR = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
					Vec3b vb_BGR = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
					Vec3b vf_hsv = CFace::kcvBGR2HSV(vf_BGR);
					Vec3b vb_hsv = CFace::kcvBGR2HSV(vb_BGR);
					
					vf_hsv[0] = vb_hsv[0];
					//vf_hsv[1] = vb_hsv[1];
					//vf_hsv[2] = vb_hsv[2];
					
					vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
					double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
					//bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], 255);
					bgra_frame_light_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], 255);
					bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
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
	rectangle(_bgraFrame, this->faceMiddleRect, Scalar(0,0,255));
	return frame;
}

Mat appFace::resizeNoseModel(int mode,Mat face){

	if(mode == DRAWMODE){//ת�ֻ棬����MASKȡԭͼֵ��
	}

	//if(mode == REALMODEPLUS) return face;

	//Mat noseModel = this->chm.currentHead.nose.model;
	//���꣬�Ȼ����ӡ��趨���ӵ�λ�ã����У����ϡ�
	Rect noseRect;
	if(this->noseDetectedRect.x>0){
		float noseHeight,noseWidth;

		if(mode == REALMODEPLUS){
		cout << " дʵ��ǿ�棬��ʼ���ű��ӡ�����" << endl;
		
		int noseHeight1 = (((float)(noseDetectedRect.y)+((float)(noseDetectedRect.height))/2) - (eyeDetectedRects[0].y))*1.1;//((float)(eyeDetectedRects[0].y)+((float)(eyeDetectedRects[0].height))/2))*1.1;
			int noseModelBridge = chm.currentHead.nose.points[0].y;
			//cout << noseModelBridge << " " << (noseModel.rows - noseModelBridge) << " " << (double)((double)noseHeight1/(double)noseModelBridge) << endl;
			noseHeight = noseHeight1 + (noseModel.rows - noseModelBridge)*(double)((double)noseHeight1/(double)noseModelBridge);
			//noseWidth = (rightEyeRect.x - (leftEyeRect.x+leftEyeRect.width))*1;//noseModel.cols * ((double)noseModel.rows/ (double)noseHeight );
			noseWidth = (eyeDetectedRects[1].x+eyesPoint[6].x - (eyeDetectedRects[0].x+eyesPoint[3].x))*1;//noseModel.cols * ((double)noseModel.rows/ (double)noseHeight );
		cout << " дʵ��ǿ�棬���ű��ӣ�" << rightEyeRect.x << " " << leftEyeRect.x << " " << leftEyeRect.width << endl;
		cout << " дʵ��ǿ�棬���ű�������ߣ�" << noseWidth << " " << noseHeight << endl;
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
			eyeDetectedRects[0].y,//+eyeDetectedRects[0].height/2,
			noseWidth,
			noseHeight);
		//�������Ŵ���С���ӿ��
		//ȡ��ֵ�����ű���������ģ��	
		//resize(noseModel, noseSample, Size(0, 0), (fWidth+fHeight)/2, (fWidth+fHeight)/2);
		resize(noseModel, noseSample, Size(noseWidth, noseHeight));
		return replaceNose(face, this->noseSample,noseRect,maskRealFace);
	}
	//rectangle(face, noseRect, Scalar(0,0,0));
	//rectangle(face, this->noseDetectedRect, Scalar(0,0,0));
	return this->faceChanged;
}

void appFace::resizeEyes(int mode){
	//-- Resize eyeModels to the eyes' sizes

	if(mode == DRAWMODE) return;

	int mean_width = eyeDetectedRects[0].width/2 + eyeDetectedRects[1].width/2;
	int mean_height = eyeDetectedRects[0].height/2 + eyeDetectedRects[1].height/2;
	if(mode != REALMODEPLUS) { // ������˫�ۼ����ƽ���ˡ��⽫���³����ֻ�ʱ���۾��߶ȳ���
		eyeDetectedRects[0] = Rect(eyeDetectedRects[0].x + eyeDetectedRects[0].width/2 - mean_width/2, 
			eyeDetectedRects[0].y + eyeDetectedRects[0].height/2 - mean_height/2,
			mean_width,mean_height);
		eyeDetectedRects[1] = Rect(eyeDetectedRects[1].x + eyeDetectedRects[1].width/2 - mean_width/2, 
			eyeDetectedRects[1].y + eyeDetectedRects[1].height/2 - mean_height/2,
			mean_width,mean_height);
	}
	if (this->eyeNumber == 2){

		//�����ű�С��ֵ���ȱȷ����۾�
		double eHeight=0.0;
		double eWeith=0.0;
		resizeEyeRate = 0.0;

		//�������ǿдʵ���8��������۾���С����
		if(mode == REALMODEPLUS){		//REALMODE��дʵ�档��ʵ����ٴ�С��λ�ã����Խӽ�1��ϵ��������١�
		cout << " дʵ��ǿ�棬��ʼ�����۾�������" << endl;
			//��ȡ����ģ��3����
			Point pEyes[2][4];//ȡ�۾�ģ������ҵ㡣lr:0ȡ���۰׵㣬1ȡ���۰׵㣬2ȡ���۱�Ե��3ȡ���۱�Ե��4ȡ�۰��ϱ�Ե��5ȡ���ϱ�Ե��
			pEyes[0][0] = chm.currentExpression.leftEye.points[2];
			pEyes[0][1] = chm.currentExpression.leftEye.points[3];
			pEyes[0][2] = chm.currentExpression.leftEye.points[0];
			pEyes[0][3] = chm.currentExpression.leftEye.points[1];
			pEyes[1][0] = chm.currentExpression.rightEye.points[2];
			pEyes[1][1] = chm.currentExpression.rightEye.points[3];
			pEyes[1][2] = chm.currentExpression.rightEye.points[0];
			pEyes[1][3] = chm.currentExpression.rightEye.points[1];

			int leftEyeModelWidth = leftEyeModel.cols;
			int leftEyeModelHeight = leftEyeModel.rows;
			int rightEyeModelWidth = rightEyeModel.cols;
			int rightEyeModelHeight = rightEyeModel.rows;
		/*
		dilate(mask6,mask6,Mat(2,2,CV_8U),Point(-1,-1),3);
		face = fillImageWithMask(oSourceImage,face,mask6,this->eyeDetectedRects[0]);
		dilate(mask61,mask61,Mat(2,2,CV_8U),Point(-1,-1),3);
		face = fillImageWithMask(oSourceImage,face,mask61,eyeDetectedRects[1]);
		dilate(bmask2,bmask2,Mat(2,2,CV_8U),Point(-1,-1),5);

		face = fillImageWithMask(oSourceImage,face,bmask2,this->browDetectedRects[0]);
		dilate(bmask21,bmask21,Mat(2,2,CV_8U),Point(-1,-1),5);
		face = fillImageWithMask(oSourceImage,face,bmask21,browDetectedRects[1]);
		//face = fillImageWithMask(oSourceImage,face,bodyLine,Rect(0,0,0,0));
		*/

			//leftEyeModel = resizeModel(leftEyeModel,pEyes[0][0],pEyes[0][1],pEyes[0][2],pEyes[0][3],eyesPoint[2],eyesPoint[3],eyesPoint[0],eyesPoint[1]);
			leftEyeModel = resizeModel(leftEyeModel,pEyes[0][0],pEyes[0][1],pEyes[0][2],pEyes[0][3],eyesPoint[2],eyesPoint[3],eyesPoint[0],eyesPoint[1],mode);
			//resizeModel(leftEyeModel,mask6);
			leftEyeModel.copyTo(leftEyeSample);
			cout << " дʵ��ǿ�棬�������۾�����������" << endl;

			//rightEyeModel = resizeModel(rightEyeModel,pEyes[1][0],pEyes[1][1],pEyes[1][2],pEyes[1][3],eyesPoint[6],eyesPoint[7],eyesPoint[4],eyesPoint[5]);
			cout << " дʵ��ǿ�棬�������۾���ʼ������" << endl;
			if(rightEyeModel.cols) cout << "resizeModel����������ģ�������� " << endl; else  cout << "resizeModel����������ģ��ͼ��Ϊ�ա� " << endl;
			rightEyeModel = resizeModel(rightEyeModel,pEyes[1][0],pEyes[1][1],pEyes[1][2],pEyes[1][3],eyesPoint[6],eyesPoint[7],eyesPoint[4],eyesPoint[5],mode);
			rightEyeModel.copyTo(rightEyeSample);
			cout << " дʵ��ǿ�棬�������۾�����������" << endl;
				imwrite("MedianData//leftEyeSample.png",leftEyeSample);

			int leftEyeModelWidth1 = leftEyeModel.cols;
			int leftEyeModelHeight1 = leftEyeModel.rows;
			int rightEyeModelWidth1 = rightEyeModel.cols;
			int rightEyeModelHeight1 = rightEyeModel.rows;

			//�������Ű��ȱȣ�����б��
			//double resizeEyeRate = 0.0;
			double eHeight=0.0;
			double eWeith=0.0;
			//�۾�����Ŀ����ģ���ȱȡ�
			
			eHeight = (double)((double)(eyesPoint[1].y-eyesPoint[0].y) / (double)(pEyes[0][3].x - pEyes[0][2].x));
			eWeith = (double)((double)(eyesPoint[3].x-eyesPoint[2].x) / (double)leftEyeModel.cols);
			//cout << eHeight << " " << eWeith << endl;
			if(mode == 8) {
			if(eWeith>eHeight)
				resizeEyeRate = eHeight;
			else
				resizeEyeRate = eWeith;
			//	resizeEyeRate = (eWeith + eHeight)/2;
			}
			cout << leftEyePupilModel.cols << " " << leftEyePupilModel.rows << " " << resizeEyeRate << endl;
			//Ӧ���������������������
			resize(leftEyePupilModel,leftEyePupilSample,Size(leftEyePupilModel.cols*resizeEyeRate, leftEyePupilModel.rows*resizeEyeRate));
			resize(rightEyePupilModel,rightEyePupilSample,Size(rightEyePupilModel.cols*resizeEyeRate, rightEyePupilModel.rows*resizeEyeRate));
			//leftEyePupilModel.copyTo(leftEyePupilSample);

			// ============================================== ����üë ===================================
			Point pBrows[2][4];//ȡ�۾�ģ������ҵ㡣lr:0ȡ���۰׵㣬1ȡ���۰׵㣬2ȡ���۱�Ե��3ȡ���۱�Ե��4ȡ�۰��ϱ�Ե��5ȡ���ϱ�Ե��
			// ===================================  ����� ��üë���� ================================
			pBrows[0][0] = chm.currentExpression.leftBrow.points[2];
			pBrows[0][1] = chm.currentExpression.leftBrow.points[3];
			pBrows[0][2] = chm.currentExpression.leftBrow.points[0];
			pBrows[0][3] = chm.currentExpression.leftBrow.points[1];
			pBrows[1][0] = chm.currentExpression.rightBrow.points[2];
			pBrows[1][1] = chm.currentExpression.rightBrow.points[3];
			pBrows[1][2] = chm.currentExpression.rightBrow.points[0];
			pBrows[1][3] = chm.currentExpression.rightBrow.points[1];

			cout << "дʵ��ǿ�棬����üëģ�壺" << endl;
			cout << " ����ǰ��üģ���ߣ�" << leftBrowModel.cols << " " << leftBrowModel.rows << endl;
			cout << " ��ü���㣺" << browPoint[2] << " " << browPoint[3] << " " << browPoint[0] << " " << browPoint[1] << endl;
			cout << " ��üģ��㣺" << pBrows[0][0] << " " << pBrows[0][1] << " " << pBrows[0][2] << " " << pBrows[0][3] << endl;
			int browWidthLeft = 0;
			int browHeightLeft = 0;
			browWidthLeft =  browPoint[3].x - browPoint[2].x;
			browHeightLeft = leftBrowModel.rows * (double)( (double)browWidthLeft /(double)leftBrowModel.cols );
			resize(leftBrowModel,leftBrowSample,Size(browWidthLeft,browHeightLeft));
			cout << " ���ź���üģ���ߣ�" << leftBrowSample.cols << " " << leftBrowSample.rows << endl;
			int browWidthRight = 0;
			int browHeightRight = 0;
			browWidthRight =  browPoint[7].x - browPoint[6].x;
			browHeightRight = rightBrowModel.rows * (double)((double) browWidthLeft /(double)rightBrowModel.cols );
			resize(rightBrowModel,rightBrowSample,Size(browWidthRight,browHeightRight));
			
			/*
			double _leftEyeModelWidth = (double)leftEyeModelWidth1/(double)leftEyeModelWidth ;
			double _leftEyeModelHeight = (double)leftEyeModelHeight1/(double)leftEyeModelHeight ;
			double _rightEyeModelWidth = (double)rightEyeModelWidth1/(double)rightEyeModelWidth ;
			double _rightEyeModelHeight = (double)rightEyeModelHeight1/(double)rightEyeModelHeight ;
			resize(leftBrowModel,leftBrowSample,Size(0,0),_leftEyeModelWidth,_leftEyeModelHeight);
			resize(rightBrowModel,rightBrowSample,Size(0,0),_rightEyeModelWidth,_rightEyeModelHeight);
			*/
			return;
			// ============================================== ����üë ===================================
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

Mat appFace::replaceEyesByREALMODEPLUS(Mat face,Rect &left,Rect &right){
	//�ƶ��۾�
	//��Ϊ��������������Ҫ�������ߣ��ƶ��۾�����Ӧλ�á�
	//����۾�eyeDetectedRects[0]��������
	//if (this->eyeDetectedRects[0].x < this->faceDetectedRect.x + 0.5 * this->faceDetectedRect.width)
	int leftEyeNum = 0,rightEyeNum = 1;
	if (this->eyeDetectedRects[0].x > faceMiddleRect.x){
		leftEyeNum = 1;
		rightEyeNum = 0;
	}

	cout << " дʵ��ǿ�棬��ʼ���۾�������" << endl;

	//��Ϊ�������߶��롣_dif�����۾���ƫ������Ҳ����ͫ���������ĵ�ƫ������Ҳ���۾��ƶ�����
	//==================================== �������ۼ�� ===============================================
	//�۾����
	int _tj = (eyeDetectedRects[rightEyeNum].x - (eyeDetectedRects[leftEyeNum].x+eyeDetectedRects[leftEyeNum].width))/2;
	//����ƫΪ��������ƫΪ��
	int _py = ((eyeDetectedRects[rightEyeNum].x-_tj) - faceMiddleRect.x);
		
	//=============================================== �۾���û�а�ʵ�ʼ���޸�
	//�۾�����ͫ��y������Ϊ���ģ����¾��У�������Ϊ���ģ������ۼ�ࡣ
	imwrite("MedianData//leftEyeSample.png",leftEyeSample);
	cout << eyesPoint[4].y << " " << this->chm.currentExpression.leftEye.points[4].y << " " << eyeDetectedRects[0].y << " " << eyeDetectedRects[1].y << endl;
	cout << eyeDetectedRects[0].y << " " << eyeDetectedRects[1].y << endl;

	// дʵ��ģ��۾����á������ҵ�Ҫ��ʵ�ʼ����غϡ������һ�೬���ˣ��Ǿ��Ǳ�ͷ�����ˡ�

	cout << chm.currentExpression.leftEye.points[2].x << endl;
	this->leftEyeRect = Rect(
		eyeDetectedRects[0].x + eyesPoint[3].x-leftEyeSample.cols,//+chm.currentExpression.leftEye.points[2].x,//�۾��ҵ����
		eyeDetectedRects[0].y+eyeDetectedRects[0].height/2 - leftEyeSample.rows/2,//+eyesPoint[0].y,// - this->chm.currentExpression.leftEye.points[0].y, //�۾��ϱ�Ե����
		leftEyeSample.cols, 
		leftEyeSample.rows );
	cout << chm.currentExpression.rightEye.points[2].x << endl;
	cout << " ��λ�۾���0��" << eyeDetectedRects[1].x << " " <<  eyesPoint[6].x << " " << chm.currentExpression.rightEye.points[2].x << endl;
	this->rightEyeRect = Rect(
		eyeDetectedRects[1].x + eyesPoint[6].x,//-chm.currentExpression.rightEye.points[2].x,
		eyeDetectedRects[1].y + eyeDetectedRects[1].height/2 -rightEyeSample.rows/2,// eyesPoint[4].y,// - this->chm.currentExpression.rightEye.points[0].y,
		rightEyeSample.cols, 
		rightEyeSample.rows );
	cout << " ��λ�۾���" << leftEyeRect.x << " " << leftEyeRect.y << " " << rightEyeRect.x << " " << rightEyeRect.y << endl;



	leftEyePupilRect = Rect(
		//��ͫ��X = �������� + �����ű���ƫ��
		eyeDetectedRects[leftEyeNum].x + eyeDetectedRects[leftEyeNum].width/2 - leftEyePupilSample.cols/2,
		eyeDetectedRects[leftEyeNum].y+ (0.5*eyeDetectedRects[leftEyeNum].height  - 0.5*leftEyePupilSample.rows),
		leftEyePupilSample.cols, 
		leftEyePupilSample.rows );
	rightEyePupilRect = Rect(
		//��ͫ��X = �������� + �����ű���ƫ��
		eyeDetectedRects[rightEyeNum].x + eyeDetectedRects[rightEyeNum].width/2 - rightEyePupilSample.cols/2,
		eyeDetectedRects[rightEyeNum].y+ 0.5*eyeDetectedRects[rightEyeNum].height  - 0.5*rightEyePupilSample.rows,
		rightEyePupilSample.cols, 
		rightEyePupilSample.rows );

	Mat maskFace;imageFaceContourSM.copyTo(maskFace);
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *face_data = face.ptr<Vec4b>(0);
	uchar *maskData = maskFace.ptr<uchar>(0);

	imwrite("MedianData//leftEyeSample.png",leftEyeSample);
	imwrite("MedianData//rightEyeSample.png",rightEyeSample);
	imwrite("MedianData//maskFace1.png",maskFace);

	//
	CFace::smoothRect(face(this->eyeDetectedRects[0]),Rect(0,0,eyeDetectedRects[0].width,eyeDetectedRects[0].height),Point(0,0),Point(0,0),1);

	//-- Change left eye
	for (int _row = 0; _row < leftEyeSample.rows ; _row++)
	{
		Vec4b *colData = leftEyeSample.ptr<Vec4b>(_row);
		//Vec4b *colData = leftEyeWithBrowSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < leftEyeSample.cols ; _col++){
			//-- Get valid area of face model
			if (colData[_col][3] == 0)
			{
				continue;
			}
			
			//if (colData[_col][0] > 250 && colData[_col][1] > 250 && colData[_col][2] > 250)
			//	continue;
			int r = leftEyeRect.y + _row; 
			int c = leftEyeRect.x + _col;
			int index = r*face.cols + c;

			//-- Override face where mask > 0�����������⣬ȫ����Ϊ0�ˡ���ȥ����
			if (maskData[index] == 0) continue;
			//frameData[index] = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
			face_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
		}
	}

	//-- Change right eye
	for (int _row = 0; _row < rightEyeSample.rows ; _row++)
	{
		Vec4b *colData = rightEyeSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < rightEyeSample.cols ; _col++){
			//-- Get valid area of face model
			if (colData[_col][3] == 0)
			{
				continue;
			}
			int r = rightEyeRect.y + _row; 
			int c = rightEyeRect.x + _col;
			int index = r*face.cols + c;
			
			//-- Override face where mask > 0
			if (maskData[index] == 0)	continue;

			face_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
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
			int index = r*face.cols + c;

			//-- Override face where mask > 0
			if (maskData[index] == 0)	continue;
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
			if(face_data[index][3]<160 && face_data[index][3]>100){
				face_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
			}
			else{
				face_data[index] = Vec4b(face_data[index][0],face_data[index][1],face_data[index][2],255);
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
			int index = r*face.cols + c;

			//-- Override face where mask > 0
			if (maskData[index] == 0)	continue;

			if(face_data[index][3]<160 && face_data[index][3]>100){
				face_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],255);
			}
			else{
				face_data[index] = Vec4b(face_data[index][0],face_data[index][1],face_data[index][2],255);
			}
		}
	}


	//-- Change left brow
	for (int _row = 0; _row < leftBrowSample.rows ; _row++)
	{
		Vec4b *colData = leftBrowSample.ptr<Vec4b>(_row);
		//Vec4b *colData = leftEyeWithBrowSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < leftBrowSample.cols ; _col++){
			//-- Get valid area of face model
			if (colData[_col][3] == 0)
			{
				continue;
			}
			int r = leftBrowRect.y + _row; 
			int c = leftBrowRect.x + _col;
			int index = r*face.cols + c;
			//cout << leftBrowRect.y << " " << _row << " " << r << " | " << leftBrowRect.x << " " << _col << " " << endl;

			if (maskData[index] == 0) continue;
			face_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
		}
	}

	//-- Change right brow
	for (int _row = 0; _row < rightBrowSample.rows ; _row++)
	{
		Vec4b *colData = rightBrowSample.ptr<Vec4b>(_row);
		for (int _col = 0 ; _col < rightBrowSample.cols ; _col++){
			//-- Get valid area of face model
			if (colData[_col][3] == 0)
			{
				continue;
			}
			int r = rightBrowRect.y + _row; 
			int c = rightBrowRect.x + _col;
			int index = r*face.cols + c;
			
			//-- Override face where mask > 0
			if (maskData[index] == 0)	continue;
			face_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
		}
	}

	Rect leftEDetect = Rect(this->eyesPoint[2].x,this->eyesPoint[0].y,this->eyesPoint[3].x-this->eyesPoint[2].x,this->eyesPoint[1].y-this->eyesPoint[0].y);
	Rect rightEDetect = Rect(this->eyesPoint[6].x,this->eyesPoint[4].y,this->eyesPoint[7].x-this->eyesPoint[6].x,this->eyesPoint[5].y-this->eyesPoint[4].y);
	Rect leftBDetect = Rect(this->browPoint[2].x,this->browPoint[0].y,this->browPoint[3].x-this->browPoint[2].x,this->browPoint[1].y-this->browPoint[0].y);
	Rect rightBDetect = Rect(this->browPoint[6].x,this->browPoint[4].y,this->browPoint[7].x-this->browPoint[6].x,this->browPoint[5].y-this->browPoint[4].y);
	//rectangle(face, rightEyeRect, Scalar(0,0,0));
	//rectangle(face, leftEyeRect, Scalar(0,0,0));
	//rectangle(face, eyeDetectedRects[0], Scalar(255,255,255));
	//rectangle(face, eyeDetectedRects[1], Scalar(255,255,255));
	//rectangle(face, rightEyePupilRect, Scalar(255,255,255));
	//rectangle(face, leftEyePupilRect, Scalar(255,255,255));
	Rect rlxR = Rect(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2,eyeDetectedRects[0].y,1,eyeDetectedRects[0].height);
	Rect rlyR = Rect(eyeDetectedRects[0].x,eyeDetectedRects[0].y+eyeDetectedRects[0].height/2,eyeDetectedRects[0].width,1);
	//rectangle(frame,rlxR,Scalar(0,0,0));
	//rectangle(frame,rlyR,Scalar(0,0,0));
	Rect rrxR = Rect(eyeDetectedRects[1].x+eyeDetectedRects[1].width/2,eyeDetectedRects[1].y,1,eyeDetectedRects[1].height);
	Rect rryR = Rect(eyeDetectedRects[1].x,eyeDetectedRects[1].y+eyeDetectedRects[1].height/2,eyeDetectedRects[1].width,1);
	//rectangle(frame,rrxR,Scalar(0,0,0));
	//rectangle(frame,rryR,Scalar(0,0,0));
	//rectangle(face, leftEDetect, Scalar(0,0,0));
	//rectangle(face, rightEDetect, Scalar(0,0,0));
	//rectangle(face, leftBDetect, Scalar(0,0,0));
	//rectangle(face, rightBDetect, Scalar(0,0,0));
	//rectangle(face, leftBrowRect, Scalar(0,0,0));
	//rectangle(face, rightBrowRect, Scalar(0,0,0));
	//rectangle(face, rightEyeWithBrowRect, Scalar(255,255,255));
	//rectangle(face, leftEyeWithBrowRect, Scalar(255,255,255));
	return face;
}

Mat appFace::fillImageWithMask(Mat imgS,Mat imgT,Mat mask,Rect r1){
		Mat grayTar ;
		if(imgS.channels() < 4){
			cvtColor(imgS(r1), grayTar, CV_BGR2GRAY);
			cvtColor(imgS, imgS, CV_BGR2BGRA);
		} else{
			Mat temp;
			cvtColor(imgS, temp, CV_BGRA2BGR);
			cvtColor(temp(r1), grayTar, CV_BGR2GRAY);
		}
		equalizeHist(grayTar,grayTar);
		imwrite("ResultData//mask6grayTar.png",grayTar);


		Vec4b *face_data = imgT.ptr<Vec4b>(0);
		Vec4b *face_data_s = imgS.ptr<Vec4b>(0);
		Mat mask6t = mask(Rect(0,0,mask.cols,mask.rows-4));
		imwrite("ResultData//mask6t.png",grayTar);

	//erode(mask6t,mask6t,Mat(2,2,CV_8U),Point(-1,-1),3);
	//morphologyEx(mask6t,mask6t,MORPH_OPEN,Mat(2,2,CV_8U),Point(-1,-1),30);
	//morphologyEx(mask6t,mask6t,MORPH_CLOSE,Mat(2,2,CV_8U),Point(-1,-1),30);

		for(int _row=0;_row<mask6t.rows;_row++){
			for(int _col=0;_col<mask6t.cols;_col++){
				int r = r1.y + _row; 
				int c = r1.x + _col;
				int index = r*imgT.cols + c;

				int indext = _row*mask6t.cols + _col;
				if(mask6t.at<uchar>(_row,_col)){
					//�仯������
					Vec3b vf_hsv,vf_BGR,vb_hsv,vb_BGR,cartoon_vb_BGR,cartoon_vb_hsv;
					vf_BGR = Vec3b(face_data_s[index][0],face_data_s[index][1],face_data_s[index][2]);
					//vb_BGR = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
					/*
					if(vb_BGR[0]<200) vb_BGR[0]=200;
					if(vb_BGR[1]<200) vb_BGR[1]=200;
					if(vb_BGR[2]<200) vb_BGR[2]=200;
					*/
					vf_hsv = CFace::kcvBGR2HSV(vf_BGR);
					//vb_hsv = CFace::kcvBGR2HSV(vb_BGR);
					//if(vb_hsv[2]<200)//ȥ���������Ե�����
					{
						//vf_hsv[0] = vb_hsv[0];
						//��������ģ����Ͼ��Ե����ˣ����ࡣ
						//vf_hsv[1] = vf_hsv[2];
						if(vf_hsv[1]>255) vf_hsv[1] = 255;
						vf_hsv[2] = grayTar.at<uchar>(_row,_col);
					}
					vf_BGR = CFace::kcvHSV2BGR(vf_hsv);

					face_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2],face_data_s[index][3]);
				}
			}
		}
		//����ͨͨ����ֱ��ͼ����ͻ��������
		return imgT;
}

Mat appFace::replaceEyes(int mode,Mat face){

	if(mode == DRAWMODE){
	//������MASK��ȡoSourceImage�����ɫ
	//Mat mask7,mask1,mask6;
	//Mat mask71,mask11,mask61;
	//Mat bmask2,bmask1;
	//Mat bmask21,bmask11;
		//����ģ�嵽�۾���С

		//ȡģ������

		dilate(mask6,mask6,Mat(2,2,CV_8U),Point(-1,-1),3);
		imwrite("ResultData//mask6.png",mask6);
		face = fillImageWithMask(oSourceImage,face,mask6,this->eyeDetectedRects[0]);
		dilate(mask61,mask61,Mat(2,2,CV_8U),Point(-1,-1),3);
		face = fillImageWithMask(oSourceImage,face,mask61,eyeDetectedRects[1]);
		dilate(bmask2,bmask2,Mat(2,2,CV_8U),Point(-1,-1),5);

		face = fillImageWithMask(oSourceImage,face,bmask2,this->browDetectedRects[0]);
		dilate(bmask21,bmask21,Mat(2,2,CV_8U),Point(-1,-1),5);
		face = fillImageWithMask(oSourceImage,face,bmask21,browDetectedRects[1]);
		//face = fillImageWithMask(oSourceImage,face,bodyLine,Rect(0,0,oSourceImage.rows,oSourceImage.cols));

		imwrite("ResultData//t.png",face);
	imwrite("ResultData//t1.png",oSourceImage);

		return face;
	}

		if(mode == REALMODEPLUS)
		{		//REALMODE��дʵ�档��ʵ����ٴ�С��λ�ã����Խӽ�1��ϵ��������١�
			return replaceEyesByREALMODEPLUS(face,leftEyeRect,rightEyeRect);
		}

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
	//========================== �������ۼ�� ========================================
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

	//������
	//this->faceChangedLight = resizeNoseModel(mode,this->faceChangedLight);
	face = resizeNoseModel(mode,face);


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

	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *bgra_frame_data = face.ptr<Vec4b>(0);
	Vec4b *bgra_frame_light_data = face.ptr<Vec4b>(0);
	uchar *maskData = maskFace.ptr<uchar>(0);

	imwrite("MedianData//leftEyeWithBrowSample.png",leftEyeWithBrowSample);
	imwrite("MedianData//rightEyeWithBrowSample.png",rightEyeWithBrowSample);


	imwrite("MedianData//maskFace1.png",maskFace);
	//-- Change left eye

	for (int _row = 0; _row < leftEyeWithBrowSample.rows ; _row++)
	{
		Vec4b *colData = leftEyeWithBrowSample.ptr<Vec4b>(_row);
		//Vec4b *colData = leftEyeWithBrowSample.ptr<Vec4b>(_row);
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
			int index = r*face.cols + c;

			//-- Override face where mask > 0�����������⣬ȫ����Ϊ0�ˡ���ȥ����
			//if (maskData[index] == 0) continue;
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
			int index = r*face.cols + c;
			
			//-- Override face where mask > 0
			//if (maskData[index] == 0)	continue;
			//frameData[index] = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
			bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
			bgra_frame_light_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],colData[_col][3]);//255);
			//bgra_frame_data[index] = Vec4b(0,0,0,255);//255);
			//bgra_frame_light_data[index] = Vec4b(0,0,0,255);//255);
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
			int index = r*face.cols + c;

			//-- Override face where mask > 0
			//if (maskData[index] == 0)	continue;
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
			int index = r*face.cols + c;

			//-- Override face where mask > 0
			//if (maskData[index] == 0)	continue;

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
	/*
	rectangle(_bgraFrame, rightEyeRect, Scalar(0,0,0));
	rectangle(_bgraFrame, leftEyeRect, Scalar(0,0,0));
	rectangle(_bgraFrame, rightEyeWithBrowRect, Scalar(255,255,255));
	rectangle(_bgraFrame, leftEyeWithBrowRect, Scalar(255,255,255));
	//rectangle(_bgraFrame, eyeDetectedRects[0], Scalar(255,255,255));
	//rectangle(_bgraFrame, eyeDetectedRects[1], Scalar(255,255,255));
	rectangle(_bgraFrame, rightEyePupilRect, Scalar(255,255,255));
	rectangle(_bgraFrame, leftEyePupilRect, Scalar(255,255,255));

	rectangle(_bgraFrameLight, rightEyeRect, Scalar(0,0,0));
	rectangle(_bgraFrameLight, leftEyeRect, Scalar(0,0,0));
	rectangle(_bgraFrameLight, rightEyeWithBrowRect, Scalar(255,255,255));
	rectangle(_bgraFrameLight, leftEyeWithBrowRect, Scalar(255,255,255));
	//rectangle(_bgraFrame, eyeDetectedRects[0], Scalar(255,255,255));
	//rectangle(_bgraFrame, eyeDetectedRects[1], Scalar(255,255,255));
	rectangle(_bgraFrameLight, rightEyePupilRect, Scalar(255,255,255));
	rectangle(_bgraFrameLight, leftEyePupilRect, Scalar(255,255,255));
	*/
	return face;
}

Mat appFace::resizeMouth(int mode,Mat face){
	//����졣�趨���λ�ã����У����ϡ�
	Mat mouthSample;
	Rect mouthRect;
	if(this->mouthDetectedRect.x>0){
		double mouthResize = 0.0;
		int mouthWidth ;
		int mouthHeight ;


		cout << "������֮ǰ�����ģ�Ϳ�ߣ�" << mouthModel.cols << " " << mouthModel.rows << endl;
		
		//if(mode == REALMODEPLUS)
		{
			mouthWidth = mouthPoint[3].x-mouthPoint[2].x;
			mouthHeight = mouthModel.rows *  (double)( (double)mouthWidth / (double)mouthModel.cols );
			resize(mouthModel, mouthSample, Size(mouthWidth, mouthHeight));

			mouthRect = Rect(
			this->faceMiddleRect.x - mouthSample.cols/2,
			mouthDetectedRect.y+mouthDetectedRect.height/2 -mouthSample.rows/2,
			mouthSample.cols,
			mouthSample.rows);
		}
		
		//�����дʵ���
		if(mode == REALMODE || mode == REALMODEPLUS ){

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
		if(mode == REALMODE || mode == REALMODEPLUS ){
			int yt= mouthDetectedRect.height/2;
			//if(mouthPoint[3].y>0 &&mouthPoint[2].y>0) yt = (mouthPoint[3].y+mouthPoint[2].y)/2;
			mouthRect = Rect(this->faceMiddleRect.x - mouthSample.cols/2,
				mouthDetectedRect.y+yt-mouthSample.rows/2,//+mouthDetectedRect.height*1/10,
				mouthSample.cols,
				mouthSample.rows);
		}

		if(mode == QFITMODE || mode == QMODE){
			mouthRect = Rect(this->faceMiddleRect.x - mouthSample.cols/2,
				mouthDetectedRect.y+mouthDetectedRect.height/3,
				mouthSample.cols,
				mouthSample.rows);
		}

		return replaceMouth(face, mouthSample,mouthRect,maskRealFace);
	}

	//rectangle(_bgraFrame, mouthDetectedRect, Scalar(0,0,255));
	//rectangle(_bgraFrame, mouthRect, Scalar(0,0,0));
	return face;
}





Mat appFace::removeFace(Mat body,Mat face,Mat bodyWithoutFace){
	int totalPixels = body.rows * body.cols;
	//��������
	if(body.channels()<4) cvtColor(body, body, CV_BGR2BGRA);
	body.copyTo(face);
	Vec4b* face_data = face.ptr<Vec4b>(0);
	//û�����ı�������
	body.copyTo(this->bodyWithoutFaceLight);
	Vec4b* bodyWithoutFace_data = this->bodyWithoutFaceLight.ptr<Vec4b>(0);

	//��������
	Vec4b* contour_data = body.ptr<Vec4b>(0);
	//�����������
	uchar* mask_face_data = maskFaceReplace.ptr<uchar>(0);
	//�������Ե
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Mat imageContourSample;imageContourSM.copyTo(imageContourSample);
	//��������������Ե
	uchar* mask_contour_data = imageContourSample.ptr<uchar>(0);
	
	for(int i = 0 ; i < totalPixels ; i++)
	{
		if (mask_contour_data[i] == 0 ){
			//face_data[i] = Vec4b(face_data[i][0], face_data[i][1], face_data[i][2],mask_real_contour_data[i]);
			face_data[i] = Vec4b(0,0,0,0);
			continue;
		}

		if (mask_face_data[i] == 0) {
			face_data[i] = Vec4b(0,0,0,0);
			continue;
		}

		if (mask_face_data[i] > 0) {
			bodyWithoutFace_data[i] = Vec4b(0,0,0,0);
			continue;
		}
		
		//face_data[i][3] = 1.0 * (255 - mask_real_face_data[i]) /255;
		//��û���������崦����һ���������͸�����ָ�Ϊ�գ��ڶ���������һ����͸����ʲôҲ�����������ڣ�����ȫ��͸����
		//bodyWithoutFace_data[i] = Vec4b(0,0,0,0);
		//bodyWithoutFace_data[i][3] = 1.0 * (255 - mask_real_face_data[i]) /255;

	}
	cvSmooth(&IplImage(face),&IplImage(face));
	return face;
}

void appFace::replaceFaceAndEyes(Mat &resultImage,int mode){
	initCounter(); // ��ʼ��
	Mat _bgraFrame;
	resultImage.copyTo(_bgraFrame);
	resultImage.copyTo(oSourceImage);
	initTempMat();
	faceModel.copyTo(faceSample);

	Mat grayDst;
	cvtColor(resultImage,grayDst,CV_BGR2GRAY);
	IplImage* imagecvAdaptiveThresholdGray = &IplImage(grayDst);
	cvAdaptiveThreshold(imagecvAdaptiveThresholdGray, imagecvAdaptiveThresholdGray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 3);
	CFace::reverseROI(grayDst);
	CFace::absROI(grayDst);
	//CFace::filterBlock(grayDst,5,5,255);
	imwrite("ResultData//imageOrigine_pre1.jpg", grayDst);
	grayDst.copyTo(lineImage);

	// ================= ������ ============
	imwrite("ResultData//BodyWithoutBackground1.png",_bgraFrame);
	bodyWithoutBackground = removeBackground(_bgraFrame,bodyWithoutBackground);
	imwrite("ResultData//BodyWithoutBackground.png",bodyWithoutBackground);

	//����������
	//changeFace(_bgraFrameLight,mask_face_replace_data,faceSample);


	// ==================ɫ��ƽ�� =================
	bodyWithoutBackgroundLight = CFace::lightBalanceFrame(bodyWithoutBackground,faceSample,bodyWithoutBackgroundLight);
	imwrite("ResultData//BodyWithoutBackgroundLight.png",bodyWithoutBackgroundLight);

	// ================= ���� ========================
	//origionFace = removeFace(bodyWithoutBackground,origionFace,this->bodyWithoutFace);
	origionFaceLight = removeFace(bodyWithoutBackgroundLight,origionFaceLight,this->bodyWithoutFaceLight);
	imwrite("ResultData//OrigionFaceLight.png",origionFaceLight);
	imwrite("ResultData//BodyWithoutFaceLight.png",this->bodyWithoutFaceLight);
	imwrite("ResultData//OrigionFace.png",origionFace);
	imwrite("ResultData//BodyWithoutFace.png",this->bodyWithoutFace);

	// ==================��ɫƽ��================  ��û��ʵ��
	//skinBalance(bodyWithoutBackground,faceSample);
	//skinBalance(bodyWithoutBackgroundLight,faceSample);

	// ================= ���� ========================
	resizeFaceModel(mode);
	Mat faceModel = chm.currentHead.faceModel;
	//replaceFace(faceModel,resultImage,mode);
	this->faceChangedLight = replaceFace(faceModel,origionFaceLight,mode);
	imwrite("ResultData//faceChangedLight.png",this->faceChangedLight);

	// ================= ������ ========================
	// �ŵ����۾������м䣬��ΪҪ�õ��۾��������
	//this->faceChangedLight = resizeNoseModel(mode,this->faceChangedLight);
	//imwrite("ResultData//faceWithNose.png",this->faceChangedLight);

	// ================= ���۾� ========================

	//mode = DRAWMODE;// ���ԣ�ת�ֻ档
	resizeEyes(mode);
	this->faceChangedLight = replaceEyes(mode,this->faceChangedLight);
	imwrite("ResultData//faceWithEyes.png",this->faceChangedLight);

	// ================= ���� ========================
	this->faceChangedLight = resizeMouth(mode,this->faceChangedLight);
	imwrite("ResultData//faceWithMouth.png",this->faceChangedLight);

	/*
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
	*/

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


//�ӱ����а�����ٳ���
Mat appFace::removeBackground(Mat srcFrame,Mat body){
	//imwrite("ResultData//BodyWithoutBackground2.png",body);
	if(srcFrame.channels() < 4){
		cvtColor(srcFrame, srcFrame, CV_BGR2BGRA);
	} 
	cvtColor(srcFrame, body, CV_BGR2BGRA);
	//imwrite("ResultData//BodyWithoutBackground2.png",body);
	//srcFrame.copyTo(body);//���������뱳��ԭͼ��һ�������ݡ�������ֻ��Ҫȥ����Ҫ�ģ�����͸��������
	//��������
	Vec4b* body_data = body.ptr<Vec4b>(0);
	int totalPixels = body.rows * body.cols;
	//�������Ե
	Mat imageRealSample;
	imageRealContourSM.copyTo(imageRealSample);
	uchar* mask_real_contour_data = imageRealSample.ptr<uchar>(0);
	//��������������Ե
	Mat imageContourSample; imageContourSM.copyTo(imageContourSample);
	uchar* mask_contour_data = imageContourSample.ptr<uchar>(0);
	//imwrite("ResultData//BodyWithoutBackground2.png",body);

	for(int i = 0 ; i < totalPixels ; i++)
	{
		
		if(mask_contour_data[i]>0){
			continue;
		}

		if (mask_real_contour_data[i] == 0)
		{
			body_data[i] = Vec4b(0,0,0,0);
			continue;
		}
		
		//body_data[i] = Vec4b(0,0,0,0);
		body_data[i][3] = 1.0 * (255 - mask_real_contour_data[i]) /255;


	}
	//imwrite("ResultData//BodyWithoutBackground4.png",body);

	return body;
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

	changeFace(_contour,mask_face_data,_face);

	
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

	//changeFace(_contour,mask_face_data,_face);

	
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




//������
//�������ݣ�Ҫ�滻����MASK����ģ�塣�򵥵Ľ��������ƶ���ȥ���������ӡ�
void appFace::changeFace(Mat &_bgraFrameLight,uchar *mask_face_replace_data,Mat faceSample){
	//	imwrite("MedianData//bgrTemp2.png", _bgraFrameLight);
	//Vec4b *bgra_frame_data = _bgraFrame.ptr<Vec4b>(0);
	Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);

	int leftCol = CFace::calFirstColOfContour(faceSample);
	int rightCol = CFace::calLastColOfContour(faceSample);
	int topRow = CFace::calFirstRowOfContour(faceSample);
	int buttomRow = CFace::calLastRowOfContour(faceSample);

	//cout << "leftCol:" << leftCol <<" rightCol:"<< rightCol <<" topRow:"<< topRow <<" buttomRow:"<< buttomRow << endl;
	//���п�ʼ��
	for(int _col = faceSampleRect.x;_col<faceSampleRect.x+faceSampleRect.width;_col++){
		int firstRow = 0,lastRow = 0;
		//�����һ�п�ʼ��
		for(int _row=faceSampleRect.y+faceSampleRect.height-1;_row>=faceSampleRect.y;_row--){
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
		//imwrite("MedianData//cartoonFilter//test.jpg", _bgraFrameLight);

}

/**
* frame ԭͼ
* mouthModel ���ӵ�ģ��ͼ
* realSource
*/
Mat appFace::replaceNose(Mat face,Mat noseSample,Rect noseRect,Mat maskRealFace)
{
			
	//�趨���λ�ã����У����ϡ�
	//Mat mouthSample;
//	mouthSample = mouthModel.
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *face_data = face.ptr<Vec4b>(0);

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
					int index = r*face.cols + c;//frame


					//-- Get valid area of nose model
					//Ϊ�˽���ױ����⣬�趨͸����Ϊ<250 �����򡣱��Ӳ��У��ϱ�Ե�����࣬��Ҫ���䣬�����ںϡ�
					if (colData[_col][3] == 0)
					{
						continue;
					}

					//���ӣ�ֻȡɫ���ǲ����ˣ����ǻơ�
					
					colData[_col][0] = superimposingTransparent(colData[_col][0],face_data[index][0],colData[_col][3],255);
					colData[_col][1] = superimposingTransparent(colData[_col][1],face_data[index][1],colData[_col][3],255);
					colData[_col][2] = superimposingTransparent(colData[_col][2],face_data[index][2],colData[_col][3],255);
					//colData[_col][3] = 255;
					
					//-- Override face where mask > 0
					if(true){
						//�仯������
						Vec3b vf_hsv,vf_BGR,vb_hsv,vb_BGR;
						vf_BGR = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
						vb_BGR = Vec3b(face_data[index][0],face_data[index][1],face_data[index][2]);
						/*
						if(vb_BGR[0]<200) vb_BGR[0]=200;
						if(vb_BGR[1]<200) vb_BGR[1]=200;
						if(vb_BGR[2]<200) vb_BGR[2]=200;
						*/
						vf_hsv = CFace::kcvBGR2HSV(vf_BGR);
						vb_hsv = CFace::kcvBGR2HSV(vb_BGR);
						//ȥ���������Ե�����
						//if(vb_hsv[2]<200)
						{
							vf_hsv[0] = vb_hsv[0];
							//vf_hsv[1] = vb_hsv[1];
							vf_hsv[2] = vb_hsv[2];
						}
						vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255; // mask_real_face_data
						//�ȸĳɲ�͸��
						face_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
						//continue;
					}
				}
			}
		}
	}

	//rectangle(face,noseRect,Scalar(0,0,0));
	//rectangle(face,this->noseDetectedRect,Scalar(0,0,0));
	imwrite("MedianData//nose.png", face);
	return face;
}

/**
* frame ԭͼ
* mouthModel ���ģ��ͼ
* realSource
*/
Mat appFace::replaceMouth(Mat face,Mat mouthSample,Rect mouthRect,Mat maskRealFace)
{
	
	//�趨���λ�ã����У����ϡ�
	//Mat mouthSample;
//	mouthSample = mouthModel.
	//Rect mouthRect;
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *face_data = face.ptr<Vec4b>(0);

	if(this->mouthDetectedRect.x>0){

	CFace::smoothRect(face(this->mouthDetectedRect),Rect(0,0,mouthDetectedRect.width,mouthDetectedRect.height),Point(0,0),Point(0,0),1);
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
					int index = r*face.cols + c;//frame

					//-- Override face where mask > 0
					if(true){
						//�仯������
						Vec3b vf_hsv,vf_BGR,vb_hsv,vb_BGR;
						vf_BGR = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
						vb_BGR = Vec3b(face_data[index][0],face_data[index][1],face_data[index][2]);
						/*
						if(vb_BGR[0]<200) vb_BGR[0]=200;
						if(vb_BGR[1]<200) vb_BGR[1]=200;
						if(vb_BGR[2]<200) vb_BGR[2]=200;
						*/
						vf_hsv = CFace::kcvBGR2HSV(vf_BGR);
						vb_hsv = CFace::kcvBGR2HSV(vb_BGR);
						//ȥ���������Ե�����
						//if(vb_hsv[2]<200)
						{
							//vf_hsv[0] = vb_hsv[0];//�촽��ʹ��ģ��ԭɫ
							//vf_hsv[1] = vb_hsv[1];
							//vf_hsv[2] = vb_hsv[2];
						}
						vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255; // mask_real_face_data
						//�ȸĳɲ�͸��
						face_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2],(1-rate)*mask_real_face_data[index]);
						//continue;
					}
				}
			}
		}
	}
	return face;
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




//  ���������ĸ�������飬ÿ���������ң���һ����üë���ڶ�����üë������ȷ����Ե�ĵ㣬Ϊ��0��0��.
Point* appFace::getBrowPoint(Mat bmask1,Mat bmask11){
	//��һ�������⣬�Ͱ���������Ϊ׼�����ҶԳ����á�
	int llx,lly,lrx,lry,ltx,lty,lbx,lby,rlx,rly,rrx,rry,rtx,rty,rbx,rby;

	llx = CFace::calFirstColOfContour(bmask1);
	//imwrite("MedianData//matLlresult.png",bmask1);
	lly = CFace::calFirstColOfContour_Row(bmask1);
	cout << " llx " << llx << " " << lly << endl;
	//eyesPoint[2] = Point(llx,lly);
	lrx = CFace::calLastColOfContour(bmask1);
	lry = CFace::calLastColOfContour_Row(bmask1);
	cout << " lrx " << lrx << " " << lry << endl;
	ltx = CFace::calFirstRowOfContour_Col(bmask1);
	lty = CFace::calFirstRowOfContour(bmask1);
	cout << " ltx " << ltx << " " << lty << endl;

	rlx = CFace::calFirstColOfContour(bmask11);
	//imwrite("MedianData//t111_30.png",bmask11);
	rly = CFace::calFirstColOfContour_Row(bmask11);
	rrx = CFace::calLastColOfContour(bmask11);
	//imwrite("MedianData//t112_30.png",bmask11);
	rry = CFace::calLastColOfContour_Row(bmask11);
	rtx = CFace::calFirstRowOfContour_Col(bmask11);
	rty = CFace::calFirstRowOfContour(bmask11);
	cout << " rlx " << rlx << " " << rly << endl;

	lby = CFace::calLastRowOfContour(bmask1(Rect(0,0,bmask1.cols,bmask1.rows-2)));
	rby = CFace::calLastRowOfContour(bmask11(Rect(0,0,bmask11.cols,bmask11.rows-2)));

	Mat frame;
	this->imageOrigine.copyTo(frame);

	leftBrowRect = Rect(browDetectedRects[0].x+llx,browDetectedRects[0].y+lty,lrx-llx,lby-lty);
	rectangle(frame,leftBrowRect,Scalar(0,0,0));
	rectangle(frame,browDetectedRects[0],Scalar(0,0,0));

	rightBrowRect = Rect(browDetectedRects[1].x+rlx,browDetectedRects[1].y+rty,rrx-rlx,rby-rty);
	rectangle(frame,rightBrowRect,Scalar(0,0,0));
	rectangle(frame,browDetectedRects[1],Scalar(0,0,0));
	imwrite("MedianData//eyeBrowRect.png",frame);
	//�߼��жϣ�


	
	//1��üëҪ���۾���
	if(browDetectedRects[0].x+llx>eyeDetectedRects[0].x+eyesPoint[2].x) llx = eyeDetectedRects[0].x+eyesPoint[2].x - browDetectedRects[0].x;
	if(browDetectedRects[0].x+lrx<eyeDetectedRects[0].x+eyesPoint[3].x) lrx = eyeDetectedRects[0].x+eyesPoint[3].x - browDetectedRects[0].x;

	if(browDetectedRects[1].x+rrx<eyeDetectedRects[1].x+eyesPoint[7].x) rrx = eyeDetectedRects[1].x+eyesPoint[7].x - browDetectedRects[1].x;
	if(browDetectedRects[1].x+rlx>eyeDetectedRects[1].x+eyesPoint[6].x) rlx = eyeDetectedRects[1].x+eyesPoint[6].x - browDetectedRects[1].x;
	
	/*
	//0�����������ԳƲ���
	int llxt,lrxt,rlxt,rrxt,mx; // ��ʵ������
	llxt = browDetectedRects[0].x+llx;
	lrxt = browDetectedRects[0].x+lrx;
	rlxt = browDetectedRects[1].x+rlx;
	rrxt = browDetectedRects[1].x+rrx;
	mx = this->faceMiddleRect.x;
	if(mx-llxt> rrxt-mx){ //����ߵ㲻һ��
		//rrx =mx-browDetectedRects[0].x-llx + mx +  browDetectedRects[1].x;
		rrx = mx + (mx - (browDetectedRects[0].x + llx)) - browDetectedRects[1].x;
	} else{
		llx = mx - (browDetectedRects[1].x + rrx - mx) + browDetectedRects[0].x;
	}
	cout << llx << " " << mx << " " << browDetectedRects[1].x << " " << rrx << " " << browDetectedRects[0].x << endl;
	if(mx-lrxt > rlxt-mx){
		lrx = mx - (browDetectedRects[1].x + rlx - mx) + browDetectedRects[0].x;
	} else {
		rlx = mx + (mx -  (browDetectedRects[0].x + lrx)) + browDetectedRects[1].x;
	}
	cout << rlx << " " << mx << " " << browDetectedRects[0].x << " " << lrx << " " << browDetectedRects[1].x << endl;
	*/
	
	int browWidthLeft = lrx - llx;
	int browWidthRight = rrx -rlx;
	int browHeightLeft = lby - lty;
	int browHeightRight = rby -rty;
	//��üëһ����һ���ߣ����������
	if(browWidthLeft > browWidthRight){

	}

	if(lty>rty) {
		lty=rty; 
		lby=lty -(rty-rby);
	}else{
		rty=lty;
		rby=rty - (lty-lby);
	}

	cout << browDetectedRects[0].x << " " << eyesPoint[2].x << " " << llx << endl;
	cout << " " << endl;
	//2���Գƴ���

	int middle = this->faceMiddleRect.x;

	browPoint[1] = Point(0,lby);//������
	browPoint[5] = Point(0,rby);//������

	browPoint[0] = Point(ltx,lty);//������
	browPoint[4] = Point(rtx,rty);//������
	cout << middle << endl;
	cout << " �ϵ�= x:" << middle - ltx << "-" << rtx - middle << "  y:" << lty << "-" << rty << endl;
	cout << " �ϵ�= x:" << middle - eyeDetectedRects[0].x - ltx << "-" << eyeDetectedRects[1].x+rtx - middle << "  y:" << eyeDetectedRects[0].y+lty << "-" << eyeDetectedRects[1].y+rty << endl;

	browPoint[2] = Point(llx,lly);//������
	browPoint[7] = Point(rrx,rry);//������
	cout << " ���= x:" << middle - llx << "-" << rrx - middle << "  y:" << lly << "-" << rry << endl;

	browPoint[3] = Point(lrx,lry);//������
	browPoint[6] = Point(rlx,rly); //������
	cout << " �ҵ�= x:" << middle - lrx << "-" << rlx - middle << "  y:" << lry << "-" << rly << endl;

	for(int i=0;i<8;i++){
		Point ep = browPoint[i];
		cout << "üë��⣬��" << i << "���㣬���꣺" << ep.x <<  "," << ep.y << endl;
	}



	return browPoint;
}
//  ���������ĸ�������飬ÿ���������ң���һ�����ۣ��ڶ������ۡ�����ȷ����Ե�ĵ㣬Ϊ��0��0��.
Point* appFace::getEyePoint(
	Mat leftEyeROI_lt,
	Mat leftEyeROI_rt,
	Mat rightEyeROI_lt,
	Mat rightEyeROI_rt,
	Mat leftEyeROI_,
	Mat rightEyeROI_){ 
	Point eyes[8] ;
	int llx,lly,lrx,lry,ltx,lty,lbx,lby,rlx,rly,rrx,rry,rtx,rty,rbx,rby;
	Rect rectl = Rect(leftEyeROI_lt.cols/4,leftEyeROI_lt.rows/4,leftEyeROI_lt.cols/2,leftEyeROI_lt.rows/2);
	Rect rectr = Rect(rightEyeROI_lt.cols/4,rightEyeROI_lt.rows/4,rightEyeROI_lt.cols/2,rightEyeROI_lt.rows/2);
	imwrite("MedianData//rightEyeROI_lt-1.png",rightEyeROI_lt);
	imwrite("MedianData//leftEyeROI_lt-1.png",leftEyeROI_lt);

	Mat matLl = leftEyeROI_lt(rectl);
	Mat matLr = leftEyeROI_rt(rectl);
	Mat matRl = rightEyeROI_lt(rectr);
	Mat matRr = rightEyeROI_rt(rectr);

	Mat matL = leftEyeROI_(rectl);
	Mat matR = rightEyeROI_(rectr);
	
	imwrite("MedianData//tttt2.png",rightEyeROI_lt);
	//==================================================
	Mat leftEyeROI;
	leftEyeROI_.copyTo(leftEyeROI);
	CFace::removeBrow(leftEyeROI_,rectl);
	leftEyeROI = leftEyeROI - leftEyeROI_;

	imwrite("MedianData//tttt3.png",rightEyeROI_lt);
	Mat rightEyeROI;
	rightEyeROI_.copyTo(rightEyeROI);
	CFace::removeBrow(rightEyeROI_,rectr);
	rightEyeROI = rightEyeROI - rightEyeROI_;
	//==================================================
	imwrite("MedianData//tttt4.png",rightEyeROI_lt);

	Mat leftEyeROI_r;
	leftEyeROI_rt.copyTo(leftEyeROI_r);
	CFace::removeBrow(leftEyeROI_rt,rectl);
	leftEyeROI_r = leftEyeROI_r - leftEyeROI_rt;

	Mat leftEyeROI_l;
	leftEyeROI_lt.copyTo(leftEyeROI_l);
	CFace::removeBrow(leftEyeROI_lt,rectl);
	imwrite("MedianData//leftEyeROI_l-1.png",leftEyeROI_l);
	imwrite("MedianData//leftEyeROI_lt.png",leftEyeROI_lt);
	leftEyeROI_l = leftEyeROI_l - leftEyeROI_lt;
	imwrite("MedianData//leftEyeROI_l.png",leftEyeROI_l);

	Mat rightEyeROI_l;
	rightEyeROI_lt.copyTo(rightEyeROI_l);//���۵����ROI���ݵ�rightEyeROI_l��
	imwrite("MedianData//t111_30_1.png",rightEyeROI_lt);
	CFace::removeBrow(rightEyeROI_lt,rectr);//������ԭͼ��ָ��������Ķ������
	imwrite("MedianData//t111_30_2.png",rightEyeROI_lt);
	rightEyeROI_l = rightEyeROI_l - rightEyeROI_lt; 
	imwrite("MedianData//t111_30_3.png",rightEyeROI_l);
	imwrite("MedianData//t111_30_4.png",rightEyeROI_lt);

	Mat rightEyeROI_r;
	rightEyeROI_rt.copyTo(rightEyeROI_r);
	CFace::removeBrow(rightEyeROI_rt,rectr);
	rightEyeROI_r = rightEyeROI_r - rightEyeROI_rt;

	llx = CFace::calFirstColOfContour(leftEyeROI_l);
	imwrite("MedianData//matLlresult.png",leftEyeROI_l);
	lly = CFace::calFirstColOfContour_Row(leftEyeROI_l);
	cout << " llx " << llx << " " << lly << endl;
	//eyesPoint[2] = Point(llx,lly);
	lrx = CFace::calLastColOfContour(leftEyeROI_r);
	lry = CFace::calLastColOfContour_Row(leftEyeROI_r);
	cout << " lrx " << lrx << " " << lry << endl;
	ltx = CFace::calFirstRowOfContour_Col(leftEyeROI_r);
	lty = CFace::calFirstRowOfContour(leftEyeROI_r);
	cout << " ltx " << ltx << " " << lty << endl;

	rlx = CFace::calFirstColOfContour(rightEyeROI_l);
	//imwrite("MedianData//t111_30.png",rightEyeROI_l);
	rly = CFace::calFirstColOfContour_Row(rightEyeROI_l);
	rrx = CFace::calLastColOfContour(rightEyeROI_r);
	//imwrite("MedianData//t112_30.png",rightEyeROI_r);
	rry = CFace::calLastColOfContour_Row(rightEyeROI_r);
	rtx = CFace::calFirstRowOfContour_Col(rightEyeROI_l);
	rty = CFace::calFirstRowOfContour(rightEyeROI_l);
	cout << " rlx " << rlx << " " << rly << endl;

	// ȡ�۾�����
	lby = CFace::calLastRowOfContour(leftEyeROI(Rect(leftEyeROI_.cols*3/8,0,leftEyeROI_.cols/4,leftEyeROI_.rows-4)));
	rby = CFace::calLastRowOfContour(rightEyeROI(Rect(rightEyeROI_.cols*3/8,0,rightEyeROI_.cols/4,rightEyeROI_.rows-4)));

	cout << "����۾������¶ˣ�" << lby << " " << rby << endl;

	Mat frame;
	this->imageOrigine.copyTo(frame);
	Rect rl = Rect(eyeDetectedRects[0].x+llx,eyeDetectedRects[0].y+lty,lrx-llx,lby-lty);
	rectangle(frame,rl,Scalar(0,0,0));
	Rect rlxR = Rect(eyeDetectedRects[0].x+eyeDetectedRects[0].width/2,eyeDetectedRects[0].y,1,eyeDetectedRects[0].height);
	Rect rlyR = Rect(eyeDetectedRects[0].x,eyeDetectedRects[0].y+eyeDetectedRects[0].height/2,eyeDetectedRects[0].width,1);
	rectangle(frame,rlxR,Scalar(0,0,0));
	rectangle(frame,rlyR,Scalar(0,0,0));
	Rect rrxR = Rect(eyeDetectedRects[1].x+eyeDetectedRects[1].width/2,eyeDetectedRects[1].y,1,eyeDetectedRects[1].height);
	Rect rryR = Rect(eyeDetectedRects[1].x,eyeDetectedRects[1].y+eyeDetectedRects[1].height/2,eyeDetectedRects[1].width,1);
	rectangle(frame,rrxR,Scalar(0,0,0));
	rectangle(frame,rryR,Scalar(0,0,0));
	rectangle(frame,eyeDetectedRects[0],Scalar(0,0,0));
	Rect rr = Rect(eyeDetectedRects[1].x+rlx,eyeDetectedRects[1].y+rty,rrx-rlx,rby-rty);
	rectangle(frame,rr,Scalar(0,0,0));
	rectangle(frame,eyeDetectedRects[1],Scalar(0,0,0));
	imwrite("MedianData//eyeRect.png",frame);


	
	//�߼��жϣ�
	//1���۱�Ե�������ۼ����Ե�غϡ�
	int middletemp = (eyeDetectedRects[0].x+eyeDetectedRects[0].width/2 + eyeDetectedRects[1].x+eyeDetectedRects[1].width/2)/2;//��ʱ���ߣ�������ԣ�
	if(llx<=0 && rrx>0) llx = middletemp - (eyeDetectedRects[1].x+rrx-middletemp)-eyeDetectedRects[0].x;//eyeDetectedRects[0].x
	if(rrx<=0 && llx>0) rrx = middletemp + (middletemp - (eyeDetectedRects[0].x+llx)) - eyeDetectedRects[1].x;
	if(lrx<=0 && rlx>0) lrx = middletemp - (eyeDetectedRects[1].x+rlx-middletemp)-eyeDetectedRects[0].x;
	if(rlx<=0 && lrx>0) rlx = middletemp + (middletemp - (eyeDetectedRects[0].x+lrx)) - eyeDetectedRects[1].x;

	cout << llx << " " << lrx << " " << rlx << " " << rrx << endl;
	//2������غ��ˣ������������Գ�ȡֵ��
	//3��������඼������غϣ�������Ϊ��ɡ��Ժ��Ϊ��Ϊ-1���滻�۾�ʱ����ģ��������
	
	//4�������ҵ㣬������㣬��������ȫ�Գ�

	cout << (eyeDetectedRects[1].y + eyeDetectedRects[1].height/2) << " " << (eyeDetectedRects[0].y + eyeDetectedRects[0].height/2) << endl;
	cout << " �ҵ�= x:" <<  lrx << "-" << rlx  << "  y:" << lry << "-" << rly << endl;
	
	//���ۿ�Ȳ�һ����ȡ��ģ��ȿ�
	//��Ϊȡ����ס����֧�������⣬�������������Գƴ����Դﵽ��ʱ�õ�Ч������ͷ�����ĵ�����Ϊ�����ᵼ������һֻ�۵�ƫ��˵���ֻ�۲���һ���ġ�
	int _wLeftEye = lrx - llx;
	int _wRightEye = rrx -rlx;
	if(_wLeftEye > _wRightEye){
		//rly = (eyeDetectedRects[1].height/2) - ((eyeDetectedRects[0].height/2) - lry);//����Գƣ�ԭ�����Ӧ���ǶԵģ���������
		//-----------------------
		rrx = rlx + _wLeftEye;
		//================== �����ǶԳƴ��� ============================
		rry = rly - (lry - lly);
		rtx = rlx + (lrx -ltx);
		rty = rly - (lry - lty);//���۸߶�һ���������ϵ�һ���ߡ�
		rby = rly - (lry - lby);
	}
	else {
		llx = lrx - _wRightEye;
		//================== �����ǶԳƴ��� ============================
		lly = lry - (rly - rry);
		ltx = lrx - (rtx - rlx);
		lty = lry - (rly - rty);
		lby = lry - (rly -rby);
	}
	cout << " �ҵ�= x:" <<  lrx << "-" << rlx  << "  y:" << (eyeDetectedRects[0].y )+lry << "-" << (eyeDetectedRects[1].y )+rly << endl;
	/*
	//���۸߶Ȳ�һ����ȡ��ģ��ȸߡ�
	int _hLeftEye = lry - lty;
	int _hRightEye = rty -rly;
	if(_hLeftEye > _hRightEye)	rty = rly - _hLeftEye ;
	else	lty = lry - _hRightEye;
	*/

	int middle = this->faceMiddleRect.x;

	eyesPoint[1] = Point(0,lby);//������
	eyesPoint[5] = Point(0,rby);//������

	eyesPoint[0] = Point(ltx,lty);//������
	eyesPoint[4] = Point(rtx,rty);//������
	cout << middle << endl;
	cout << " �ϵ�= x:" << middle - ltx << "-" << rtx - middle << "  y:" << lty << "-" << rty << endl;
	cout << " �ϵ�= x:" << middle - eyeDetectedRects[0].x - ltx << "-" << eyeDetectedRects[1].x+rtx - middle << "  y:" << eyeDetectedRects[0].y+lty << "-" << eyeDetectedRects[1].y+rty << endl;

	eyesPoint[2] = Point(llx,lly);//������
	eyesPoint[7] = Point(rrx,rry);//������
	cout << " ���= x:" << middle - llx << "-" << rrx - middle << "  y:" << lly << "-" << rry << endl;

	eyesPoint[3] = Point(lrx,lry);//������
	eyesPoint[6] = Point(rlx,rly); //������
	cout << " �ҵ�= x:" << middle - lrx << "-" << rlx - middle << "  y:" << lry << "-" << rly << endl;

	for(int i=0;i<8;i++){
		Point ep = eyesPoint[i];
		cout << "��" << i << "���㣬���꣺" << ep.x <<  "," << ep.y << endl;
	}

	return eyes;
}

Point* appFace::getNosePoint(Mat noseROI){ //  ��������������飬���ҡ�����ȷ����Ե�ĵ㣬Ϊ��0��0��.
	nosePoint[0];
	//��λ�����е㣬����һ��1/4�ߵľ��ο�

	//������ɨ�裬�ҵ�����һ���׵㣬��¼������������ߣ��������ڱߣ����ҵ������ڱߣ�
	return nosePoint;
}

Point* appFace::getMouthPoint(Mat mouthROI){ //  �����ĸ�������飬�������ҡ�����ȷ����Ե�ĵ㣬Ϊ��0��0��.
	int lx,ly,rx,ry;
	lx = CFace::calFirstColOfContour(mouthROI);
	ly = CFace::calFirstColOfContour_Row(mouthROI);
	rx = CFace::calLastColOfContour(mouthROI);
	ry = CFace::calLastColOfContour_Row(mouthROI);
	mouthPoint[2] = Point(lx,ly);
	mouthPoint[3] = Point(rx,ry);
	return mouthPoint;
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
	for(int _col=model.cols-1;_col>=0;_col--){
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
	for(int _col=model.cols-1;_col>=0;_col--){
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
		for(int _col=model.cols-1;_col>=0;_col--){
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

/*��ģ�尴ָ�����������ת�����š���Ϊrotate(mat,angle)��һ����ת��ʧȥԭͼ��С�����ܱ���ԭ��Сת��ȥ��
����������Ҫ�ȼ������ţ�Ȼ��һ��ת����ɡ��ȳ��Զ����ȥ������ɫ��
*/
Mat appFace::resizeModel2(Mat modelImg,Point ml,Point mr,Point mt,Point mb,Point el,Point er,Point et,Point eb,int mode){
	if(modelImg.cols) cout << "resizeModel������ͼ�������� " << endl; else  cout << "resizeModel������ͼ��Ϊ�ա� " << endl;
	int xmlength = mr.x - ml.x; // TODO: mr.xֵ����
	int ymlength = mr.y - ml.y; // ģ�����ҵ�Yֵ��
	int xelength = er.x - el.x;
	int yelength = er.y - el.y;

	if(mode == 9) {//����ת��ֱ�ӿ�����š��������ȫ�ˡ�
		double dwidth = (double)xelength / (double)xmlength;
		double dheight = (double)yelength / (double)ymlength;
		resize(modelImg,modelImg,Size(0,0),dwidth,dheight);
		return modelImg;
	}


	//���Ŵ�����������ת�������š�//�����������һ�����⣬��Ҫ��дrotate(mat,angle)��ת���ܣ����򱨴�
	// TODO: ����ģ��ת��ˮƽ�������ţ�����ת���۾��Ƕȣ����������ţ���ģ�嶥�����۾�������ͬ�߶ȣ���ʱ���۾�ģ�����¸߶���һ������ֵ��
	double arcMS,arcES;
	arcMS = atan2(abs((double)(mr.y-ml.y)),abs((double)(mr.x-ml.x)));
	arcMS = arcMS*180/CV_PI; // ����ģ�ͶԽ���ԭʼ�Ƕ�
	arcES = atan2(abs((double)(er.y-el.y)),abs((double)(er.x-el.x)));
	arcES = arcES*180/CV_PI; // //�������۶Խ��߽Ƕ�
	
	//�����۾���ߵ㵽�Խ��ߵĸ߶�
	double l_e,l_ea,l_eb,l_ec;
	double l_eat = (el.x-er.x)*(el.x-er.x)+(el.y-er.y)*(el.y-er.y);
	l_ea=sqrt(l_eat);//����ģ��Խ��߳���
	cout << " (el.x-er.x):" << (el.x-er.x) << " (el.y-er.y):" << (el.y-er.y) << " l_ea:" << l_ea << endl;
	double l_ebt = (el.x-et.x)*(el.x-et.x)+(el.y-et.y)*(el.y-et.y);
	l_eb=sqrt(l_ebt);
	double l_ect = (er.x-et.x)*(er.x-et.x)+(er.y-et.y)*(er.y-et.y);
	l_ec=sqrt(l_ect);
	//CosC=(a^2+b^2-c^2)/2ab ;b*CosC=a1;((a^2+b^2-c^2)/2a)=a1;l_e=squr(b^2-((a^2+b^2-c^2)/2a)^2)
	double l_et = l_eb*l_eb-((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea))*((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea));
	l_e=sqrt(l_et);
	cout << " l_e" << l_e;
	//����ģ����ߵ㵽�Խ��ߵĸ߶�
	int l_m,l_ma,l_mb,l_mc;
	double l_mat = (ml.x-mr.x)*(ml.x-mr.x)+(ml.y-mr.y)*(ml.y-mr.y);
	l_ma=sqrt(l_mat);//�������۶Խ��߳���
	cout << " (ml.x-mr.x):" << (ml.x-mr.x) << " (ml.y-mr.y):" << (ml.y-mr.y) << " l_ma:" << l_ma << endl;
	double l_mbt = (ml.x-mt.x)*(ml.x-mt.x)+(ml.y-mt.y)*(ml.y-mt.y);
	l_mb=sqrt(l_mbt);
	double l_mct = (mr.x-mt.x)*(mr.x-mt.x)+(mr.y-mt.y)*(mr.y-mt.y);
	l_mc=sqrt(l_mct);
	double l_mt = l_mb*l_mb-((l_ma*l_ma+l_mb*l_mb-l_mc*l_mc)/(2*l_ma))*((l_ma*l_ma+l_mb*l_mb-l_mc*l_mc)/(2*l_ma));
	l_m=sqrt(l_mt);
	//����������ű���������һ������ֵ����������ģ������۶���תƽ���ٱ����ߵ㡣
	double _dh_em = (double)l_e/(double)l_m;//ȱ�����ˡ������򵥵��㷨���Ȳ��ϣ������档
	double _dl_em = (double)l_ea/(double)l_ma;
	cout << " l_eb:" << l_eb << " l_ec:" << l_ec << " l_mb:" << l_mb << " l_mc:" << l_mc << " " << endl;
	cout << " _dh_em:" << _dh_em << " (l_e:" << l_e << ")/(l_m:" << l_m << ")|   _dl_em:" << _dl_em << " (l_ea:" << l_ea << ")/(l_ma:" << l_ma << ")"<< endl;
	
	//�ж���ת����
	int directe,directm;
	if(el.y>er.y) directe = 1; // ��������
	if(el.y<=er.y) directe = 0; // ��������
	if(ml.y>mr.y) directm = 1; // ��������
	if(ml.y<=mr.y) directm = 0; // ��������
	//��ģ����ת��ˮƽ

	cout << "arcMs:" << arcMS << endl;

	if(directm)	
		CFace::rotate(modelImg,-1*arcMS);
	else 
		CFace::rotate(modelImg,arcMS); 
	//cout << modelImg.cols << " " << modelImg.rows << endl;
	//���ţ����۾������ҵ㳤�ȡ��߶ȡ�
	if(mode == 1){
		resize(modelImg,modelImg,Size(0,0),_dl_em,_dh_em);
		//��ת���۾���б�Ƕ�
		if(directe) CFace::rotate(modelImg,-1*arcES); else  CFace::rotate(modelImg,arcES);
		imwrite("MedianData//rotateEye.jpg",modelImg);
		return modelImg;
	}

			cout << " ��ת��ɣ�׼�����š�����" << endl;


	//cout << modelImg.cols << " " << modelImg.rows << endl;
	resize(modelImg,modelImg,Size(0,0),_dl_em,1);
	//��ת���۾���б�Ƕ�
	if(directe) CFace::rotate(modelImg,-1*arcES); else  CFace::rotate(modelImg,arcES);

	//int model_h = mt.y -mb.y;
	//int eye_h = et.y - eb.y;
	//_dh_em = (double)eye_h/(double)model_h;//ȱ�����ˡ������򵥵��㷨���Ȳ��ϡ�
	int model_h = mt.y -mb.y;
	int eye_h = et.y - eb.y;
	_dh_em = (double)eye_h/(double)model_h;//ȱ�����ˡ������򵥵��㷨���Ȳ��ϡ�
	cout << model_h << " " << eye_h << " " << _dh_em << endl;
	
	resize(modelImg,modelImg,Size(0,0),1,_dh_em);
	resizeEyeRate = _dh_em;
	imwrite("MedianData//rotateEye.jpg",modelImg);
	return modelImg;
}


/*��ģ�尴ָ�����������ת�����š���Ϊrotate(mat,angle)��һ����ת��ʧȥԭͼ��С�����ܱ���ԭ��Сת��ȥ��
����������Ҫ�ȼ������ţ�Ȼ��һ��ת����ɡ��ȳ��Զ����ȥ������ɫ��
*/
Mat appFace::resizeModel(Mat modelImg,Point ml,Point mr,Point mt,Point mb,Point el,Point er,Point et,Point eb,int mode){
	if(modelImg.cols) cout << "resizeModel������ͼ�������� " << endl; else  cout << "resizeModel������ͼ��Ϊ�ա� " << endl;
	if(mode == 8) {//����ת��ֱ�ӿ�����š��������ȫ�ˡ�
		int xmlength = mr.x - ml.x; // TODO: mr.xֵ����
		int ymlength = mb.y - mt.y; // ģ�����ҵ�Yֵ��
		int xelength = er.x - el.x;
		int yelength = eb.y - et.y;

		double dwidth = (double)xelength / (double)xmlength;
		double dheight = (double)yelength / (double)ymlength;
		cout << dwidth << " " << dheight << endl;
		resize(modelImg,modelImg,Size(0,0),dwidth,dheight);
		return modelImg;
	}


	//���Ŵ�����������ת�������š�//�����������һ�����⣬��Ҫ��дrotate(mat,angle)��ת���ܣ����򱨴�
	// TODO: ����ģ��ת��ˮƽ�������ţ�����ת���۾��Ƕȣ����������ţ���ģ�嶥�����۾�������ͬ�߶ȣ���ʱ���۾�ģ�����¸߶���һ������ֵ��
	double arcMS,arcES;
	arcMS = atan2(abs((double)(mr.y-ml.y)),abs((double)(mr.x-ml.x)));
	arcMS = arcMS*180/CV_PI; // ����ģ�ͶԽ���ԭʼ�Ƕ�
	arcES = atan2(abs((double)(er.y-el.y)),abs((double)(er.x-el.x)));
	arcES = arcES*180/CV_PI; // //�������۶Խ��߽Ƕ�
	
	//�����۾���ߵ㵽�Խ��ߵĸ߶�
	double l_e,l_ea,l_eb,l_ec;
	double l_eat = (el.x-er.x)*(el.x-er.x)+(el.y-er.y)*(el.y-er.y);
	l_ea=sqrt(l_eat);//����ģ��Խ��߳���
	cout << " (el.x-er.x):" << (el.x-er.x) << " (el.y-er.y):" << (el.y-er.y) << " l_ea:" << l_ea << endl;
	double l_ebt = (el.x-et.x)*(el.x-et.x)+(el.y-et.y)*(el.y-et.y);
	l_eb=sqrt(l_ebt);
	double l_ect = (er.x-et.x)*(er.x-et.x)+(er.y-et.y)*(er.y-et.y);
	l_ec=sqrt(l_ect);
	//CosC=(a^2+b^2-c^2)/2ab ;b*CosC=a1;((a^2+b^2-c^2)/2a)=a1;l_e=squr(b^2-((a^2+b^2-c^2)/2a)^2)
	double l_et = l_eb*l_eb-((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea))*((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea));
	l_e=sqrt(l_et);
	cout << " l_e" << l_e;
	//����ģ����ߵ㵽�Խ��ߵĸ߶�
	int l_m,l_ma,l_mb,l_mc;
	double l_mat = (ml.x-mr.x)*(ml.x-mr.x)+(ml.y-mr.y)*(ml.y-mr.y);
	l_ma=sqrt(l_mat);//�������۶Խ��߳���
	cout << " (ml.x-mr.x):" << (ml.x-mr.x) << " (ml.y-mr.y):" << (ml.y-mr.y) << " l_ma:" << l_ma << endl;
	double l_mbt = (ml.x-mt.x)*(ml.x-mt.x)+(ml.y-mt.y)*(ml.y-mt.y);
	l_mb=sqrt(l_mbt);
	double l_mct = (mr.x-mt.x)*(mr.x-mt.x)+(mr.y-mt.y)*(mr.y-mt.y);
	l_mc=sqrt(l_mct);
	double l_mt = l_mb*l_mb-((l_ma*l_ma+l_mb*l_mb-l_mc*l_mc)/(2*l_ma))*((l_ma*l_ma+l_mb*l_mb-l_mc*l_mc)/(2*l_ma));
	l_m=sqrt(l_mt);
	//����������ű���������һ������ֵ����������ģ������۶���תƽ���ٱ����ߵ㡣
	double _dh_em = (double)l_e/(double)l_m;//ȱ�����ˡ������򵥵��㷨���Ȳ��ϣ������档
	double _dl_em = (double)l_ea/(double)l_ma;
	cout << " l_eb:" << l_eb << " l_ec:" << l_ec << " l_mb:" << l_mb << " l_mc:" << l_mc << " " << endl;
	cout << " _dh_em:" << _dh_em << " (l_e:" << l_e << ")/(l_m:" << l_m << ")|   _dl_em:" << _dl_em << " (l_ea:" << l_ea << ")/(l_ma:" << l_ma << ")"<< endl;
	
	//�ж���ת����
	int directe,directm;
	if(el.y>er.y) directe = 1; // ��������
	if(el.y<=er.y) directe = 0; // ��������
	if(ml.y>mr.y) directm = 1; // ��������
	if(ml.y<=mr.y) directm = 0; // ��������
	//��ģ����ת��ˮƽ

	cout << "arcMs:" << arcMS << endl;

	if(directm)	
		CFace::rotate(modelImg,-1*arcMS);
	else 
		CFace::rotate(modelImg,arcMS); 
	//cout << modelImg.cols << " " << modelImg.rows << endl;
	//���ţ����۾������ҵ㳤�ȡ��߶ȡ�
	if(mode == 1){
		resize(modelImg,modelImg,Size(0,0),_dl_em,_dh_em);
		//��ת���۾���б�Ƕ�
		if(directe) CFace::rotate(modelImg,-1*arcES); else  CFace::rotate(modelImg,arcES);
		imwrite("MedianData//rotateEye.jpg",modelImg);
		return modelImg;
	}

			cout << " ��ת��ɣ�׼�����š�����" << endl;


	//cout << modelImg.cols << " " << modelImg.rows << endl;
	resize(modelImg,modelImg,Size(0,0),_dl_em,1);
	//��ת���۾���б�Ƕ�
	if(directe) CFace::rotate(modelImg,-1*arcES); else  CFace::rotate(modelImg,arcES);

	//int model_h = mt.y -mb.y;
	//int eye_h = et.y - eb.y;
	//_dh_em = (double)eye_h/(double)model_h;//ȱ�����ˡ������򵥵��㷨���Ȳ��ϡ�
	int model_h = mt.y -mb.y;
	int eye_h = et.y - eb.y;
	_dh_em = (double)eye_h/(double)model_h;//ȱ�����ˡ������򵥵��㷨���Ȳ��ϡ�
	cout << model_h << " " << eye_h << " " << _dh_em << endl;
	
	resize(modelImg,modelImg,Size(0,0),1,_dh_em);
	resizeEyeRate = _dh_em;
	imwrite("MedianData//rotateEye.jpg",modelImg);
	return modelImg;
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

