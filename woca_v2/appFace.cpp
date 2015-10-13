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
	//边缘小于全脸的1/8，就算是接着的。
	int w = (CFace::calLastColOfContour(imageFaceContourSM)-CFace::calFirstColOfContour(imageFaceContourSM))/8;
	int wt = 0;
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
	int _fl = CFace::calLastColOfContour(imageFaceContourSM);
	int _flr = CFace::calLastColOfContour_Row(imageFaceContourSM);
	//边缘小于全脸的1/8，就算是接着的。
	int w = (CFace::calLastColOfContour(imageFaceContourSM)-CFace::calFirstColOfContour(imageFaceContourSM))/8;
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

int appFace::calFirstRowOfContourHuman(){//改写为调用新方法，发现不太稳定。先改回去用。
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

vector<Rect> appFace::detectEyes(Mat _face){//在脸中检测眼睛
		//开始检测眼睛，
	Mat faceROI = _face;//frame_gray( faceDetectedRect );
	std::vector<Rect> eyes;
	int minSize = faceROI.rows / 5;
	for(int t=0;t<3;t++){
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize, minSize));
		if(eyes.size()!=2) continue;
		else t=3;
	}

	if(eyes.size()>2){	
		eyes.erase(eyes.begin()+2,eyes.end());//删除区间[2,结尾];区间从0开始
	}
	else if(eyes.size()>0)
		return eyes;
	else
		cout << " 没有检测到眼睛。 " << endl;

	return eyes;
}

void appFace::setEyesParameters(vector<Rect> __eyes){
	//	cout << " 检测到眼睛，开始设置参数。。。" << endl;
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
			cout << " 两眼检测框高度一样。高度分别为：" << eyeDetectedRects[0].height << ", " << eyeDetectedRects[1].height << endl;
			cout << "" << endl;
		} else {
			cout << " 两眼检测框高度不一样。高度分别为：" << eyeDetectedRects[0].y << ", " << eyeDetectedRects[0].height << " | " << eyeDetectedRects[1].y << ", " << eyeDetectedRects[1].height << endl;
			cout << "" << endl;
		}

		Mat roi ;
		imageOrigine(eyeDetectedRects[0]).copyTo(roi);
		//=======================================================================
		mask1 = CFace::createROI(roi,"eyeDetectedRects[0]1",1,2,3); // 中值算法 1/3半径
		mask2 = CFace::createROI(roi,"eyeDetectedRects[0]2",0,3,3); // 中值算法 1/3半径

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
		mask6 = CFace::createROI(roi,"eyeDetectedRects[0]6",0,2,3); // 中值算法 1/3半径
		mask7 = CFace::createROI(roi,"eyeDetectedRects[0]7",0,2,roi.rows/5); // 中值算法 5半径
		//=======================================================================
		Rect rBrow = Rect(mask1.cols/3,mask1.rows/3,mask1.cols/3,mask1.rows/3);
		Mat mask1tmp;mask1.copyTo(mask1tmp);
		CFace::removeBrow(mask1,rBrow);//MASK去掉眉毛部分
		mask1=mask1tmp-mask1;
		imwrite("MedianData//eyeDetectedRects[0]1-1.png",mask1);
		CFace::filterBlock(mask7,mask1,true); // 过滤细线MASK图
		imwrite("MedianData//eyeDetectedRects[0]7-1.png",mask7);
		CFace::filterBlock(mask6,mask1,true); // 过滤粗线MASK图
		imwrite("MedianData//eyeDetectedRects[0]6-1.png",mask7);

		imageOrigine(eyeDetectedRects[1]).copyTo(roi);
		//=======================================================================
		imwrite("MedianData//roi11.png",roi);
		mask11 = CFace::createROI(roi,"eyeDetectedRects[1]1",1,2,3); // 中值算法 1/3半径
		mask61 = CFace::createROI(roi,"eyeDetectedRects[1]6",0,2,3); // 中值算法 1/3半径
		mask71 = CFace::createROI(roi,"eyeDetectedRects[1]7",0,2,roi.rows/5); // 中值算法 5半径
		//=======================================================================
		rBrow = Rect(mask11.cols/3,mask11.rows/3,mask11.cols/3,mask11.rows/3);
		mask11.copyTo(mask1tmp);
		CFace::removeBrow(mask11,rBrow);//MASK去掉眉毛部分
		mask11=mask1tmp-mask11;
		CFace::filterBlock(mask71,mask11,true); // 过滤细线MASK图
		CFace::filterBlock(mask61,mask11,true); // 过滤粗线MASK图
		imwrite("MedianData//eyeDRects[0]6.png",mask6);
		imwrite("MedianData//eyeDRects[0]7.png",mask7);
		imwrite("MedianData//eyeDRects[0]1.png",mask1);
		imwrite("MedianData//eyeDRects[1]71.png",mask71);
		imwrite("MedianData//eyeDRects[1]61.png",mask61);
		imwrite("MedianData//eyeDRects[1]1.png",mask11);
		Mat m7,m71;mask7.copyTo(m7);mask71.copyTo(m71);
		getEyePoint(mask6,mask7,mask71,mask61,m7,m71); // 根据左右眼的ROI，计算点。

		//=========================这里取黑眼球==================================
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

		em11 = CFace::createROI(testMASK1,"ED11",1,2,3); // 中值算法 1/3半径
		em15 = CFace::createROI(testMASK1,"ED15",0,3,3); // 中值算法 1/3半径
		em12 = CFace::createROI(testMASK1,"ED12",0,2,3); // 中值算法 1/3半径
		em13 = CFace::createROI(testMASK1,"ED13",0,2,roi.rows/5); // 中值算法 1/3半径
		//----------------------------------------
		em14 = CFace::createROI(testMASK1,"ED14",1,1,128); // 中值算法 1/3半径cvAvg
		//接下来，估垂直积分，根据曲度变化，取出眼珠左右边界。
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
		mask11 = CFace::createROI(testMASK21,"ED21",1,2,3); // 中值算法 1/3半径
		mask11 = CFace::createROI(testMASK21,"ED22",0,2,3); // 中值算法 1/3半径
		mask11 = CFace::createROI(testMASK21,"ED23",0,2,roi.rows/5); // 中值算法 1/3半径
		mask11 = CFace::createROI(testMASK21,"ED24",1,1,128); // 中值算法 1/3半径
		*/
		//=========================这里需要取眉毛，并设置参数点==================================
		//先设定眉毛的检测区域
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
		bmask1 = CFace::createROI(broi,"browDetectedRects[0]1",1,2,3); // 中值算法 1/3半径
		//=======================================================================
		Rect lBrow = Rect(0,bmask1.rows/4,bmask1.cols*3/4,bmask1.rows/2);
		bmask1.copyTo(bmask2);
		CFace::removeBrow(bmask1,lBrow);//MASK去掉眉毛部分
		bmask1 = bmask2 - bmask1;
		imwrite("MedianData//browDetectedRects[0]71.png",bmask1);
		
		Mat broi1 ;
		imageOrigine(browDetectedRects[1]).copyTo(broi1);
		//=======================================================================
		bmask11 = CFace::createROI(broi1,"browDetectedRects[1]1",1,2,3); // 中值算法 1/3半径
		//=======================================================================
		rBrow = Rect(bmask11.cols/4,bmask11.rows/4,bmask11.cols*3/4,bmask11.rows/2);
		bmask11.copyTo(bmask21);
		CFace::removeBrow(bmask11,rBrow);//MASK去掉眉毛部分
		bmask11 = bmask21 - bmask11;
		imwrite("MedianData//browDetectedRects[1]71.png",bmask11);
		
		getBrowPoint(bmask1,bmask11); // 根据左右眼的ROI，计算点。
		cout << "" << endl;

		//==========================================================================

	for(int i=0;i<8;i++){
		Point ep = browPoint[i];
		cout << "第" << i << "个点，坐标：" << ep.x <<  "," << ep.y << endl;
	}

	} else 
	{
		for(int i=0;i<eyeNumber;i++){
			eyeDetectedRects[i] = Rect(faceDetectedRect.x+__eyes[i].x,faceDetectedRect.y+__eyes[i].y,__eyes[i].width,__eyes[i].height);
		}
	}
}
void appFace::setMouthsParameter(Vector<Rect> mouths){
		cout << " 检测到嘴，开始设置参数。。。" << endl;

	//======================================= 验证嘴是否正确 ===============================================
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
	//===================================================================================================
	//找到了一个嘴
	if(_ym>=0){
		this->mouthDetectedRect = Rect(this->faceDetectedRect.x+mouths[_ym].x,this->faceDetectedRect.y+mouths[_ym].y,mouths[_ym].width,mouths[_ym].height);


	} else {
		cout << " 没有找到正确的嘴 " << endl;
	}
}
void appFace::setMouthsParameter(Vector<Rect> mouths,Rect mouthRegion){
		cout << " 检测到嘴，开始设置参数。。。" << endl;
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


	/* 肤色模型
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
	bmask11 = CFace::createROI(broi1,"mouth1",1,2,3); // 中值算法 1/3半径
	//=======================================================================
	Rect rBrow = Rect(bmask11.cols/4,bmask11.rows/4,bmask11.cols*3/4,bmask11.rows/2);
	bmask11.copyTo(bmask21);
	CFace::removeBrow(bmask11,rBrow);//MASK去掉眉毛部分
	bmask11 = bmask21 - bmask11;
	imwrite("MedianData//mouth11.png",bmask11);


	getMouthPoint(bmask11); // 根据嘴的ROI，计算点。


}

void appFace::setNoseParameter(Vector<Rect> noses){

	//==========================================  检查鼻子是否正确  =====================================================
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
	//==========================================================================================================
	if(_yn>=0){
		this->noseDetectedRect = Rect(this->faceDetectedRect.x+noses[_yn].x,this->faceDetectedRect.y+noses[_yn].y,noses[_yn].width,noses[_yn].height);
	} else {
		cout << " 没有找到鼻子。 " << endl;
	}
}
void appFace::setNoseParameter(Vector<Rect> noses,Rect noseRegion){
		cout << " 检测到鼻子，开始设置参数。。。" << endl;

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
	else {
		cout << " 没有找到鼻子。 " << endl;
		return;
	}

		Mat roi;
		imageOrigine(noseDetectedRect).copyTo(roi);
		Mat mask7,mask1,mask2,mask3,mask4,mask5,mask6,mask8,mask9,mask10,mask11,mask12;
		//=======================================================================
		//bmask1 = createROI(broi,"browDetectedRects[0]1",1,2,3); // 中值算法 1/3半径
		//createROI(Mat m,string name,int pre,int mode,int range){ 
		// m源图。pre，预处理：0，无；1，直方图。mode：1，cvThreshold；2，cvAdaptiveThreshold；3，Canny边缘检测。
		//mask1 = CFace::createROI(roi,"nose1",1,1,3); // 中值算法 1/3半径
		mask2 = CFace::createROI(roi,"nose2",1,2,3); // 中值算法 1/3半径
		//mask3 = CFace::createROI(roi,"nose3",1,3,30); // 中值算法 5半径
		//mask4 = CFace::createROI(roi,"nose4",1,1,roi.rows/5); // 中值算法 1/3半径
		//mask5 = CFace::createROI(roi,"nose5",1,2,roi.rows/5); // 中值算法 1/3半径
		//mask6 = CFace::createROI(roi,"nose6",1,3,roi.rows/3); // 中值算法 5半径

		//=======================================================================
		/*
		Rect rBrow = Rect(0,0,mask1.cols,mask1.rows/4);
		imwrite("MedianData//eyeDetectedRects[0]10.png",mask1);
		removeBrow(mask1,rBrow);//MASK去掉眉毛部分
		filterBlock(mask7,mask1,true); // 过滤细线MASK图
		filterBlock(mask6,mask1,true); // 过滤粗线MASK图
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
		for(int t=0;t<3;t++){
			mouth_cascade.detectMultiScale( faceROI, mouths, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize1*3, minSize1));
			if(mouths.size()==1) t=3;
			else continue;
		}
		return mouths;
}

vector<Rect> appFace::detectNose(Mat _face){//在脸中检测鼻子

		//在脸里检测鼻子
		Mat faceROI = _face;//frame_gray( faceDetectedRect );
		std::vector<Rect> noses;
		//这个参数值影响巨大
		int minSize2 = faceROI.rows / 8;
		for(int t=0;t<3;t++){
			nose_cascade.detectMultiScale( faceROI, noses, 1.01, 10, 0 |CV_HAAR_SCALE_IMAGE, Size(minSize2, minSize2));
			if(noses.size()==1) t=3;
			else continue;
		}
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
	int angleDegree = (45)/angle;
	for(int i_angle=1;i_angle<angleDegree;i_angle++){ //从1度开始
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
		cout << " 旋转后重新检测脸一次。。。" << endl;
		Rect maskFaceRect;
		for(int t=0;t<3;t++){
			face_cascade.detectMultiScale( frame_gray, _faces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
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
				if(leftEyeMiddleY > rightEyeMiddleY && angle > 0 && i_angle == 1) angle = -1 * angle;
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

	//std::vector<Rect> faces; //做为全局变量了.
	Mat frame_gray;
	
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- 旋转并检测脸，取出双眼平行的图像。
	int faceCount  = rotateDetectFaces();
	cout << faceCount << endl;
	if(faceCount > 0 ){

		rotate();//旋转所有图像
		frame = CFace::rotate(frame,this->rotateAngle); // 旋转frame，准备在上面标记五官。
		imwrite("MedianData//simpleFaceDetection1.png",frame);
		frame.copyTo(debugFrame);//初始化debugFrame.

		setFaceParameters(frame);//设置脸的参数
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
				//createROI(roi,"nose");

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
	//为了检查错误先把复杂检测注释掉。
	simpleFaceDetection1();
	//simpleFaceDetection();
	//colorBasedFaceDetection();//这段实在不知道是什么用处，注释掉先。
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

//根据不同模式，处理脸模板。
void appFace::resizeFaceModel(int mode){
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
		fWidth = ((double)  faceWidth) / (double)faceModel.cols;
		fHeight = ((double) faceChangeRect.height) / (double)faceModel.rows;
		//如果计算出了脸的宽度，就修改faceChangeRect
		faceChangeRect = Rect(faceChangeRect.x+(faceChangeRect.width-faceWidth)/2,faceChangeRect.y,faceChangeRect.width,faceChangeRect.height);
	}
	else{ // TODO: 如果两侧挡脸，按黄金分割设置脸宽。
		faceWidth = faceModel.cols *  ((double)faceChangeRect.height / (double)faceModel.rows);
		fWidth = (double)  faceWidth / faceModel.cols;
		fHeight = (double) faceChangeRect.height / faceModel.rows;
		cout << "resizeFaceModel=> fWidth:" << fWidth << " fHeight:" << fHeight << endl;
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
	cout << "faceWidth:" << faceWidth << " fWidth:" << fWidth << " fHeight:" << fHeight << endl;
	faceSampleRect = Rect((faceMiddleRect.x-faceWidth/2),(faceChangeRect.y+faceChangeRect.height-faceSample.rows),(faceSample.cols),(faceSample.rows));
}
Mat appFace::replaceFace(Mat faceModel,Mat &resultImage,int mode){
	Mat ret;
	if(mode == REALMODEPLUS){
		ret = replaceFaceByREALMODEPLUS(faceModel,resultImage);
		return ret;
	}

	if(mode == DRAWMODE){ // 转手绘。

	}
	 return replaceFaceByDefault(faceModel,resultImage);
}


//在resultImage图中，按mode模式，将脸换成faceModel。
Mat appFace::replaceFaceByREALMODEPLUS(Mat faceModel,Mat &resultImage){
	imwrite("ResultData//OrigionFace_t1.png",resultImage);
	Mat frame;resultImage.copyTo(frame);
	//if(true) return frame;//为了测试虚化脸替代脸模板方案，这里直接返回了。
	imwrite("ResultData//OrigionFace_t2.png",frame);
	Vec4b *bgra_frame_data = frame.ptr<Vec4b>(0);

	//=======================================  底图效果处理二，平滑  ===================================
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

	//换脸上的显示数据
	uchar* mask_face_replace_data = maskFaceReplace.ptr<uchar>(0);
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);

	//因脸模板加宽了，不能从左到右匹配，而应该从中间对齐。需要将faceChangeRect向右移到中线。
	int middle = -1*(this->faceMiddleRect.x - faceChangeRect.x-faceChangeRect.width/2) +  faceSample.cols/2 - faceChangeRect.width/2;
	

	//先捏脸瘦脸
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
			//因脸模板加宽了，不能从左到右匹配，而应该从中间对齐。需要将faceChangeRect向右移到中线。
			int c = faceChangeRect.x - middle + _col;
			int index = r*frame.cols + c;
			//-- Get valid area of face model
			
			if (colDataBGRA[_col][3] == 0){ // 如果模板上没有数据就将脸这个地方设为0，相当于捏完脸了，但是还没有把背景拉伸。
				//mask_face_replace_data[index] = 0;//这里修改了原MASK数据
				bgra_frame_data[index] = Vec4b(0,0,0,0);
				continue;
			}
			
			//
			if (mask_face_replace_data[index] == 255)
			{
				//变化明暗度
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
				//if(vb_hsv[2]<200)//去掉过于明显的数据
				{
					vf_hsv[0] = vb_hsv[0];
					//加上下面的，脸上就显得乱了，很脏。
					vf_hsv[1] = vb_smooth_hsv[1];//+vb_hsv[1]*0.3; 
					vf_hsv[2] = vf_hsv[2];//+vb_hsv[2]*0.3;
				}
				vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
				//如果脸模板相同位置不透明，说明是有内容的
				if(colDataBGRA[_col][3]>5){
					double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
					//bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
					bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
				} else {
					//如果脸模板相同位置透明，说明是捏脸的部分，要被去掉

				}
				continue;
			}

			//FACEMASK这外，根据realFace处理
			if(mask_real_face_data[index] < 32){
				continue;
			}
			//如果脸部范围为白色，直接取脸色值。
			if (mask_real_face_data[index] > 223){
					double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
				bgra_frame_data[index] = Vec4b(colData[_col][0],colData[_col][1],colData[_col][2],(1-rate)*mask_real_face_data[index]);
			}
			// 这段是想把脸周边增加半透明处理，不做了。效果不好。
			else { //否则，要做透明化处理
				//变化明暗度
				Vec3b vf_BGR = Vec3b(colData[_col][0],colData[_col][1],colData[_col][2]);
				//这里取的是暗背景的点数据。对亮背景的，会有问题。
				Vec3b vb_BGR = Vec3b(bgra_frame_data[index][0],bgra_frame_data[index][1],bgra_frame_data[index][2]);
				Vec3b vf_hsv = CFace::kcvBGR2HSV(vf_BGR);
				Vec3b vb_hsv = CFace::kcvBGR2HSV(vb_BGR);
					
				vf_hsv[0] = vb_hsv[0];//只取色相值
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
	//这里需要处理
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

	//换脸上的显示数据
	uchar* mask_face_replace_data = maskFaceReplace.ptr<uchar>(0);
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);

	//因脸模板加宽了，不能从左到右匹配，而应该从中间对齐。需要将faceChangeRect向右移到中线。
	int middle = -1*(this->faceMiddleRect.x - faceChangeRect.x-faceChangeRect.width/2) +  faceSample.cols/2 - faceChangeRect.width/2;

	
	//亮度平衡处理
	IplImage* imageFace = &IplImage(faceSampleBGR);
	IplImage* imageBgr = &IplImage(_bgraFrameLight);
	IplImage* imageBgrSkin = &IplImage(_bgrFrameSkin);
	double gFace = CFace::get_avg_gray(imageFace);
	double gBgr = CFace::get_avg_gray(imageBgr);
	//为了调试捏脸，先注释掉。
	CFace::set_avg_gray(imageBgr,imageBgr,gFace*0.9);

	//肤色处理。
	CvSize imageSize = cvSize(imageBgrSkin->width, imageBgrSkin->height);
	IplImage *imageSkin = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
	
	CFace::cvSkinSegment(imageBgrSkin,imageSkin);
	//cvSkinYUV(imageBgrSkin,imageSkin);
	//cvSkinHSV(imageBgrSkin,imageSkin);
	Mat skinMat= Mat(imageSkin);

	imwrite("MedianData//skinTemp.png", skinMat);
	imwrite("MedianData//faceTemp.png", faceSampleBGR);
	//写调整亮度后的文件
	imwrite("MedianData//bgrLight.png", _bgraFrameLight);

	//捏脸瘦脸
	imwrite("MedianData//bgrLightBeforeChangeFaceNoTransparent.png", _bgraFrameLight);
	changeFace(_bgraFrameLight,mask_face_replace_data,faceSample);
	imwrite("MedianData//bgrLightAfterChangeFaceNoTransparent.png", _bgraFrameLight);
	// 这里开始加模式处理。
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
			
			if(true)
			{
				if (mask_face_replace_data[index] == 255)
				{
					//变化明暗度
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
					//去掉过于明显的数据
					//if(vb_hsv[2]<200)
					{
						vf_hsv[0] = vb_hsv[0];
						//加上下面的，脸上就显得乱了，很脏。
						//vf_hsv[1] = vf_hsv[1]+vb_hsv[1]*0.3;N

						//vf_hsv[2] = vf_hsv[2]+vb_hsv[2]*0.3;
					}
					vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
					//如果脸模板相同位置不透明，说明是有内容的
					if(colDataBGRA[_col][3]>5){
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255;
						bgra_frame_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
						bgra_frame_light_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2], (1-rate)*mask_real_face_data[index]);
					} else {
						//如果脸模板相同位置透明，说明是捏脸的部分，要被去掉

					}
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

	if(mode == DRAWMODE){//转手绘，根据MASK取原图值。
	}

	//if(mode == REALMODEPLUS) return face;

	//Mat noseModel = this->chm.currentHead.nose.model;
	//换完，先换鼻子。设定鼻子的位置：居中，靠上。
	Rect noseRect;
	if(this->noseDetectedRect.x>0){
		float noseHeight,noseWidth;

		if(mode == REALMODEPLUS){
		cout << " 写实增强版，开始缩放鼻子。。。" << endl;
		
		int noseHeight1 = (((float)(noseDetectedRect.y)+((float)(noseDetectedRect.height))/2) - (eyeDetectedRects[0].y))*1.1;//((float)(eyeDetectedRects[0].y)+((float)(eyeDetectedRects[0].height))/2))*1.1;
			int noseModelBridge = chm.currentHead.nose.points[0].y;
			//cout << noseModelBridge << " " << (noseModel.rows - noseModelBridge) << " " << (double)((double)noseHeight1/(double)noseModelBridge) << endl;
			noseHeight = noseHeight1 + (noseModel.rows - noseModelBridge)*(double)((double)noseHeight1/(double)noseModelBridge);
			//noseWidth = (rightEyeRect.x - (leftEyeRect.x+leftEyeRect.width))*1;//noseModel.cols * ((double)noseModel.rows/ (double)noseHeight );
			noseWidth = (eyeDetectedRects[1].x+eyesPoint[6].x - (eyeDetectedRects[0].x+eyesPoint[3].x))*1;//noseModel.cols * ((double)noseModel.rows/ (double)noseHeight );
		cout << " 写实增强版，缩放鼻子：" << rightEyeRect.x << " " << leftEyeRect.x << " " << leftEyeRect.width << endl;
		cout << " 写实增强版，缩放鼻子至宽高：" << noseWidth << " " << noseHeight << endl;
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
			eyeDetectedRects[0].y,//+eyeDetectedRects[0].height/2,
			noseWidth,
			noseHeight);
		//按比例放大缩小鼻子宽度
		//取均值的缩放比例，缩放模板	
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
	if(mode != REALMODEPLUS) { // 这段里把双眼检测框给平均了。这将导致超级手绘时，眼睛高度出错。
		eyeDetectedRects[0] = Rect(eyeDetectedRects[0].x + eyeDetectedRects[0].width/2 - mean_width/2, 
			eyeDetectedRects[0].y + eyeDetectedRects[0].height/2 - mean_height/2,
			mean_width,mean_height);
		eyeDetectedRects[1] = Rect(eyeDetectedRects[1].x + eyeDetectedRects[1].width/2 - mean_width/2, 
			eyeDetectedRects[1].y + eyeDetectedRects[1].height/2 - mean_height/2,
			mean_width,mean_height);
	}
	if (this->eyeNumber == 2){

		//按缩放比小的值，等比放缩眼睛
		double eHeight=0.0;
		double eWeith=0.0;
		resizeEyeRate = 0.0;

		//如果是增强写实风格8，按检测眼睛大小缩放
		if(mode == REALMODEPLUS){		//REALMODE，写实版。按实际五官大小、位置，乘以接近1的系数，放五官。
		cout << " 写实增强版，开始缩放眼睛。。。" << endl;
			//先取左眼模板3个点
			Point pEyes[2][4];//取眼睛模板的左、右点。lr:0取左眼白点，1取右眼白点，2取左眼边缘，3取右眼边缘，4取眼白上边缘，5取眼上边缘。
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
			cout << " 写实增强版，缩放左眼睛结束。。。" << endl;

			//rightEyeModel = resizeModel(rightEyeModel,pEyes[1][0],pEyes[1][1],pEyes[1][2],pEyes[1][3],eyesPoint[6],eyesPoint[7],eyesPoint[4],eyesPoint[5]);
			cout << " 写实增强版，缩放右眼睛开始。。。" << endl;
			if(rightEyeModel.cols) cout << "resizeModel，输入右眼模板正常。 " << endl; else  cout << "resizeModel，输入右眼模板图像为空。 " << endl;
			rightEyeModel = resizeModel(rightEyeModel,pEyes[1][0],pEyes[1][1],pEyes[1][2],pEyes[1][3],eyesPoint[6],eyesPoint[7],eyesPoint[4],eyesPoint[5],mode);
			rightEyeModel.copyTo(rightEyeSample);
			cout << " 写实增强版，缩放右眼睛结束。。。" << endl;
				imwrite("MedianData//leftEyeSample.png",leftEyeSample);

			int leftEyeModelWidth1 = leftEyeModel.cols;
			int leftEyeModelHeight1 = leftEyeModel.rows;
			int rightEyeModelWidth1 = rightEyeModel.cols;
			int rightEyeModelHeight1 = rightEyeModel.rows;

			//眼珠缩放按等比，不倾斜。
			//double resizeEyeRate = 0.0;
			double eHeight=0.0;
			double eWeith=0.0;
			//眼睛检测点的宽度与模板宽度比。
			
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
			//应该是左眼珠跟着左眼缩放
			resize(leftEyePupilModel,leftEyePupilSample,Size(leftEyePupilModel.cols*resizeEyeRate, leftEyePupilModel.rows*resizeEyeRate));
			resize(rightEyePupilModel,rightEyePupilSample,Size(rightEyePupilModel.cols*resizeEyeRate, rightEyePupilModel.rows*resizeEyeRate));
			//leftEyePupilModel.copyTo(leftEyePupilSample);

			// ============================================== 缩放眉毛 ===================================
			Point pBrows[2][4];//取眼睛模板的左、右点。lr:0取左眼白点，1取右眼白点，2取左眼边缘，3取右眼边缘，4取眼白上边缘，5取眼上边缘。
			// ===================================  这段是 抠眉毛方案 ================================
			pBrows[0][0] = chm.currentExpression.leftBrow.points[2];
			pBrows[0][1] = chm.currentExpression.leftBrow.points[3];
			pBrows[0][2] = chm.currentExpression.leftBrow.points[0];
			pBrows[0][3] = chm.currentExpression.leftBrow.points[1];
			pBrows[1][0] = chm.currentExpression.rightBrow.points[2];
			pBrows[1][1] = chm.currentExpression.rightBrow.points[3];
			pBrows[1][2] = chm.currentExpression.rightBrow.points[0];
			pBrows[1][3] = chm.currentExpression.rightBrow.points[1];

			cout << "写实增强版，缩放眉毛模板：" << endl;
			cout << " 缩放前左眉模板宽高：" << leftBrowModel.cols << " " << leftBrowModel.rows << endl;
			cout << " 左眉检测点：" << browPoint[2] << " " << browPoint[3] << " " << browPoint[0] << " " << browPoint[1] << endl;
			cout << " 左眉模板点：" << pBrows[0][0] << " " << pBrows[0][1] << " " << pBrows[0][2] << " " << pBrows[0][3] << endl;
			int browWidthLeft = 0;
			int browHeightLeft = 0;
			browWidthLeft =  browPoint[3].x - browPoint[2].x;
			browHeightLeft = leftBrowModel.rows * (double)( (double)browWidthLeft /(double)leftBrowModel.cols );
			resize(leftBrowModel,leftBrowSample,Size(browWidthLeft,browHeightLeft));
			cout << " 缩放后左眉模板宽高：" << leftBrowSample.cols << " " << leftBrowSample.rows << endl;
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
			// ============================================== 缩放眉毛 ===================================
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

Mat appFace::replaceEyesByREALMODEPLUS(Mat face,Rect &left,Rect &right){
	//移动眼睛
	//因为检测的是眼球，所以要根据中线，移动眼睛到相应位置。
	//如果眼睛eyeDetectedRects[0]在左半边脸
	//if (this->eyeDetectedRects[0].x < this->faceDetectedRect.x + 0.5 * this->faceDetectedRect.width)
	int leftEyeNum = 0,rightEyeNum = 1;
	if (this->eyeDetectedRects[0].x > faceMiddleRect.x){
		leftEyeNum = 1;
		rightEyeNum = 0;
	}

	cout << " 写实增强版，开始换眼睛。。。" << endl;

	//改为从中心线对齐。_dif真正眼睛的偏移量，也就是瞳孔在眼中心的偏移量，也是眼睛移动量。
	//==================================== 保持两眼间距 ===============================================
	//眼睛间距
	int _tj = (eyeDetectedRects[rightEyeNum].x - (eyeDetectedRects[leftEyeNum].x+eyeDetectedRects[leftEyeNum].width))/2;
	//向右偏为正，向左偏为负
	int _py = ((eyeDetectedRects[rightEyeNum].x-_tj) - faceMiddleRect.x);
		
	//=============================================== 眼睛框，没有按实际检测修改
	//眼睛，以瞳孔y轴中心为中心，上下居中；以中线为中心，增加眼间距。
	imwrite("MedianData//leftEyeSample.png",leftEyeSample);
	cout << eyesPoint[4].y << " " << this->chm.currentExpression.leftEye.points[4].y << " " << eyeDetectedRects[0].y << " " << eyeDetectedRects[1].y << endl;
	cout << eyeDetectedRects[0].y << " " << eyeDetectedRects[1].y << endl;

	// 写实类的，眼睛设置。上左右点要与实际检测点重合。如查有一侧超界了，那就是被头发挡了。

	cout << chm.currentExpression.leftEye.points[2].x << endl;
	this->leftEyeRect = Rect(
		eyeDetectedRects[0].x + eyesPoint[3].x-leftEyeSample.cols,//+chm.currentExpression.leftEye.points[2].x,//眼睛右点对齐
		eyeDetectedRects[0].y+eyeDetectedRects[0].height/2 - leftEyeSample.rows/2,//+eyesPoint[0].y,// - this->chm.currentExpression.leftEye.points[0].y, //眼睛上边缘对齐
		leftEyeSample.cols, 
		leftEyeSample.rows );
	cout << chm.currentExpression.rightEye.points[2].x << endl;
	cout << " 定位眼睛框0：" << eyeDetectedRects[1].x << " " <<  eyesPoint[6].x << " " << chm.currentExpression.rightEye.points[2].x << endl;
	this->rightEyeRect = Rect(
		eyeDetectedRects[1].x + eyesPoint[6].x,//-chm.currentExpression.rightEye.points[2].x,
		eyeDetectedRects[1].y + eyeDetectedRects[1].height/2 -rightEyeSample.rows/2,// eyesPoint[4].y,// - this->chm.currentExpression.rightEye.points[0].y,
		rightEyeSample.cols, 
		rightEyeSample.rows );
	cout << " 定位眼睛框：" << leftEyeRect.x << " " << leftEyeRect.y << " " << rightEyeRect.x << " " << rightEyeRect.y << endl;



	leftEyePupilRect = Rect(
		//左瞳孔X = 左眼中线 + 按缩放倍数偏移
		eyeDetectedRects[leftEyeNum].x + eyeDetectedRects[leftEyeNum].width/2 - leftEyePupilSample.cols/2,
		eyeDetectedRects[leftEyeNum].y+ (0.5*eyeDetectedRects[leftEyeNum].height  - 0.5*leftEyePupilSample.rows),
		leftEyePupilSample.cols, 
		leftEyePupilSample.rows );
	rightEyePupilRect = Rect(
		//右瞳孔X = 右眼中线 + 按缩放倍数偏移
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

			//-- Override face where mask > 0。这里有问题，全部都为0了。先去掉。
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
					//变化明暗度
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
					//if(vb_hsv[2]<200)//去掉过于明显的数据
					{
						//vf_hsv[0] = vb_hsv[0];
						//加上下面的，脸上就显得乱了，很脏。
						//vf_hsv[1] = vf_hsv[2];
						if(vf_hsv[1]>255) vf_hsv[1] = 255;
						vf_hsv[2] = grayTar.at<uchar>(_row,_col);
					}
					vf_BGR = CFace::kcvHSV2BGR(vf_hsv);

					face_data[index] = Vec4b(vf_BGR[0],vf_BGR[1],vf_BGR[2],face_data_s[index][3]);
				}
			}
		}
		//对三通通道做直方图，以突出特征。
		return imgT;
}

Mat appFace::replaceEyes(int mode,Mat face){

	if(mode == DRAWMODE){
	//根据眼MASK，取oSourceImage里的着色
	//Mat mask7,mask1,mask6;
	//Mat mask71,mask11,mask61;
	//Mat bmask2,bmask1;
	//Mat bmask21,bmask11;
		//缩放模板到眼睛大小

		//取模板数据

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
		{		//REALMODE，写实版。按实际五官大小、位置，乘以接近1的系数，放五官。
			return replaceEyesByREALMODEPLUS(face,leftEyeRect,rightEyeRect);
		}

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
	//========================== 保持两眼间距 ========================================
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

	//换鼻子
	//this->faceChangedLight = resizeNoseModel(mode,this->faceChangedLight);
	face = resizeNoseModel(mode,face);


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

			//-- Override face where mask > 0。这里有问题，全部都为0了。先去掉。
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
	//最后换嘴。设定嘴的位置：居中，靠上。
	Mat mouthSample;
	Rect mouthRect;
	if(this->mouthDetectedRect.x>0){
		double mouthResize = 0.0;
		int mouthWidth ;
		int mouthHeight ;


		cout << "缩放嘴之前，嘴的模型宽高：" << mouthModel.cols << " " << mouthModel.rows << endl;
		
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
		
		//如果是写实风格
		if(mode == REALMODE || mode == REALMODEPLUS ){

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
	//人脸数据
	if(body.channels()<4) cvtColor(body, body, CV_BGR2BGRA);
	body.copyTo(face);
	Vec4b* face_data = face.ptr<Vec4b>(0);
	//没有脸的背景数据
	body.copyTo(this->bodyWithoutFaceLight);
	Vec4b* bodyWithoutFace_data = this->bodyWithoutFaceLight.ptr<Vec4b>(0);

	//背景数据
	Vec4b* contour_data = body.ptr<Vec4b>(0);
	//切整齐的人脸
	uchar* mask_face_data = maskFaceReplace.ptr<uchar>(0);
	//真脸外边缘
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Mat imageContourSample;imageContourSM.copyTo(imageContourSample);
	//切整齐的真人外边缘
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
		//对没有脸的身体处理，第一个是脸外边透明部分高为空，第二个是与脸一样设透明，什么也不做，象现在，就是全不透明。
		//bodyWithoutFace_data[i] = Vec4b(0,0,0,0);
		//bodyWithoutFace_data[i][3] = 1.0 * (255 - mask_real_face_data[i]) /255;

	}
	cvSmooth(&IplImage(face),&IplImage(face));
	return face;
}

void appFace::replaceFaceAndEyes(Mat &resultImage,int mode){
	initCounter(); // 初始化
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

	// ================= 抠人像 ============
	imwrite("ResultData//BodyWithoutBackground1.png",_bgraFrame);
	bodyWithoutBackground = removeBackground(_bgraFrame,bodyWithoutBackground);
	imwrite("ResultData//BodyWithoutBackground.png",bodyWithoutBackground);

	//先捏脸瘦脸
	//changeFace(_bgraFrameLight,mask_face_replace_data,faceSample);


	// ==================色彩平衡 =================
	bodyWithoutBackgroundLight = CFace::lightBalanceFrame(bodyWithoutBackground,faceSample,bodyWithoutBackgroundLight);
	imwrite("ResultData//BodyWithoutBackgroundLight.png",bodyWithoutBackgroundLight);

	// ================= 抠脸 ========================
	//origionFace = removeFace(bodyWithoutBackground,origionFace,this->bodyWithoutFace);
	origionFaceLight = removeFace(bodyWithoutBackgroundLight,origionFaceLight,this->bodyWithoutFaceLight);
	imwrite("ResultData//OrigionFaceLight.png",origionFaceLight);
	imwrite("ResultData//BodyWithoutFaceLight.png",this->bodyWithoutFaceLight);
	imwrite("ResultData//OrigionFace.png",origionFace);
	imwrite("ResultData//BodyWithoutFace.png",this->bodyWithoutFace);

	// ==================肤色平衡================  还没有实现
	//skinBalance(bodyWithoutBackground,faceSample);
	//skinBalance(bodyWithoutBackgroundLight,faceSample);

	// ================= 换脸 ========================
	resizeFaceModel(mode);
	Mat faceModel = chm.currentHead.faceModel;
	//replaceFace(faceModel,resultImage,mode);
	this->faceChangedLight = replaceFace(faceModel,origionFaceLight,mode);
	imwrite("ResultData//faceChangedLight.png",this->faceChangedLight);

	// ================= 换鼻子 ========================
	// 放到换眼睛过程中间，因为要用到眼睛框参数。
	//this->faceChangedLight = resizeNoseModel(mode,this->faceChangedLight);
	//imwrite("ResultData//faceWithNose.png",this->faceChangedLight);

	// ================= 换眼睛 ========================

	//mode = DRAWMODE;// 测试，转手绘。
	resizeEyes(mode);
	this->faceChangedLight = replaceEyes(mode,this->faceChangedLight);
	imwrite("ResultData//faceWithEyes.png",this->faceChangedLight);

	// ================= 换嘴 ========================
	this->faceChangedLight = resizeMouth(mode,this->faceChangedLight);
	imwrite("ResultData//faceWithMouth.png",this->faceChangedLight);

	/*
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
	*/

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


//从背景中把人像抠出来
Mat appFace::removeBackground(Mat srcFrame,Mat body){
	//imwrite("ResultData//BodyWithoutBackground2.png",body);
	if(srcFrame.channels() < 4){
		cvtColor(srcFrame, srcFrame, CV_BGR2BGRA);
	} 
	cvtColor(srcFrame, body, CV_BGR2BGRA);
	//imwrite("ResultData//BodyWithoutBackground2.png",body);
	//srcFrame.copyTo(body);//现在身体与背景原图有一样的数据。接下来只需要去掉不要的，和做透明化处理
	//背景数据
	Vec4b* body_data = body.ptr<Vec4b>(0);
	int totalPixels = body.rows * body.cols;
	//真人外边缘
	Mat imageRealSample;
	imageRealContourSM.copyTo(imageRealSample);
	uchar* mask_real_contour_data = imageRealSample.ptr<uchar>(0);
	//切整齐的真人外边缘
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

	//切整齐的人脸
	uchar* mask_face_data = maskFaceReplace.ptr<uchar>(0);
	//背景数据
	Vec4b* contour_data = _contour.ptr<Vec4b>(0);
	//真人外边缘
	uchar* mask_real_contour_data = imageRealContourSM.ptr<uchar>(0);
	//切整齐的真人外边缘
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

	//切整齐的人脸
	uchar* mask_face_data = maskFaceReplace.ptr<uchar>(0);
	//背景数据
	Vec4b* contour_data = _contour.ptr<Vec4b>(0);
	//真人外边缘
	uchar* mask_real_contour_data = imageRealContourSM.ptr<uchar>(0);
	//切整齐的真人外边缘
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
	//亮度平衡处理
	IplImage* imageFace = &IplImage(_face);
	IplImage* imageBgr = &IplImage(_contour);
	IplImage* imageBgrSkin = &IplImage(_contour);
	double gFace = get_avg_gray(imageFace);
	double gBgr = get_avg_gray(imageBgr);
	set_avg_gray(imageBgr,imageBgr,gFace*0.7);
	*/


}




//捏脸。
//背景数据，要替换部分MASK，脸模板。简单的将脸下面移动上去，做好连接。
void appFace::changeFace(Mat &_bgraFrameLight,uchar *mask_face_replace_data,Mat faceSample){
	//	imwrite("MedianData//bgrTemp2.png", _bgraFrameLight);
	//Vec4b *bgra_frame_data = _bgraFrame.ptr<Vec4b>(0);
	Vec4b *bgra_frame_light_data = _bgraFrameLight.ptr<Vec4b>(0);

	int leftCol = CFace::calFirstColOfContour(faceSample);
	int rightCol = CFace::calLastColOfContour(faceSample);
	int topRow = CFace::calFirstRowOfContour(faceSample);
	int buttomRow = CFace::calLastRowOfContour(faceSample);

	//cout << "leftCol:" << leftCol <<" rightCol:"<< rightCol <<" topRow:"<< topRow <<" buttomRow:"<< buttomRow << endl;
	//从列开始数
	for(int _col = faceSampleRect.x;_col<faceSampleRect.x+faceSampleRect.width;_col++){
		int firstRow = 0,lastRow = 0;
		//从最后一行开始数
		for(int _row=faceSampleRect.y+faceSampleRect.height-1;_row>=faceSampleRect.y;_row--){
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
		//imwrite("MedianData//cartoonFilter//test.jpg", _bgraFrameLight);

}

/**
* frame 原图
* mouthModel 鼻子的模板图
* realSource
*/
Mat appFace::replaceNose(Mat face,Mat noseSample,Rect noseRect,Mat maskRealFace)
{
			
	//设定嘴的位置：居中，靠上。
	//Mat mouthSample;
//	mouthSample = mouthModel.
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *face_data = face.ptr<Vec4b>(0);

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
					int index = r*face.cols + c;//frame


					//-- Get valid area of nose model
					//为了解决白边问题，设定透明度为<250 的区域。鼻子不行，上边缘及两侧，需要渐变，与脸融合。
					if (colData[_col][3] == 0)
					{
						continue;
					}

					//鼻子，只取色相是不对了，还是黄。
					
					colData[_col][0] = superimposingTransparent(colData[_col][0],face_data[index][0],colData[_col][3],255);
					colData[_col][1] = superimposingTransparent(colData[_col][1],face_data[index][1],colData[_col][3],255);
					colData[_col][2] = superimposingTransparent(colData[_col][2],face_data[index][2],colData[_col][3],255);
					//colData[_col][3] = 255;
					
					//-- Override face where mask > 0
					if(true){
						//变化明暗度
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
						//去掉过于明显的数据
						//if(vb_hsv[2]<200)
						{
							vf_hsv[0] = vb_hsv[0];
							//vf_hsv[1] = vb_hsv[1];
							vf_hsv[2] = vb_hsv[2];
						}
						vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255; // mask_real_face_data
						//先改成不透明
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
* frame 原图
* mouthModel 嘴的模板图
* realSource
*/
Mat appFace::replaceMouth(Mat face,Mat mouthSample,Rect mouthRect,Mat maskRealFace)
{
	
	//设定嘴的位置：居中，靠上。
	//Mat mouthSample;
//	mouthSample = mouthModel.
	//Rect mouthRect;
	uchar* mask_real_face_data = maskRealFace.ptr<uchar>(0);
	Vec4b *face_data = face.ptr<Vec4b>(0);

	if(this->mouthDetectedRect.x>0){

	CFace::smoothRect(face(this->mouthDetectedRect),Rect(0,0,mouthDetectedRect.width,mouthDetectedRect.height),Point(0,0),Point(0,0),1);
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
					int index = r*face.cols + c;//frame

					//-- Override face where mask > 0
					if(true){
						//变化明暗度
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
						//去掉过于明显的数据
						//if(vb_hsv[2]<200)
						{
							//vf_hsv[0] = vb_hsv[0];//嘴唇，使用模板原色
							//vf_hsv[1] = vb_hsv[1];
							//vf_hsv[2] = vb_hsv[2];
						}
						vf_BGR = CFace::kcvHSV2BGR(vf_hsv);
						double rate = 1.0 * (255 - mask_real_face_data[index]) /255; // mask_real_face_data
						//先改成不透明
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




//  返回两个四个点的数组，每组上下左右，第一组左眉毛，第二组右眉毛。不能确定边缘的点，为（0，0）.
Point* appFace::getBrowPoint(Mat bmask1,Mat bmask11){
	//有一个有问题，就按以脸中线为准，左右对称设置。
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
	//逻辑判断：


	
	//1、眉毛要比眼睛宽
	if(browDetectedRects[0].x+llx>eyeDetectedRects[0].x+eyesPoint[2].x) llx = eyeDetectedRects[0].x+eyesPoint[2].x - browDetectedRects[0].x;
	if(browDetectedRects[0].x+lrx<eyeDetectedRects[0].x+eyesPoint[3].x) lrx = eyeDetectedRects[0].x+eyesPoint[3].x - browDetectedRects[0].x;

	if(browDetectedRects[1].x+rrx<eyeDetectedRects[1].x+eyesPoint[7].x) rrx = eyeDetectedRects[1].x+eyesPoint[7].x - browDetectedRects[1].x;
	if(browDetectedRects[1].x+rlx>eyeDetectedRects[1].x+eyesPoint[6].x) rlx = eyeDetectedRects[1].x+eyesPoint[6].x - browDetectedRects[1].x;
	
	/*
	//0、按中线做对称补充
	int llxt,lrxt,rlxt,rrxt,mx; // 点实际坐标
	llxt = browDetectedRects[0].x+llx;
	lrxt = browDetectedRects[0].x+lrx;
	rlxt = browDetectedRects[1].x+rlx;
	rrxt = browDetectedRects[1].x+rrx;
	mx = this->faceMiddleRect.x;
	if(mx-llxt> rrxt-mx){ //如果边点不一致
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
	//两眉毛一样宽，一样高，按大的来。
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
	//2、对称处理

	int middle = this->faceMiddleRect.x;

	browPoint[1] = Point(0,lby);//左眼下
	browPoint[5] = Point(0,rby);//右眼下

	browPoint[0] = Point(ltx,lty);//左眼上
	browPoint[4] = Point(rtx,rty);//右眼上
	cout << middle << endl;
	cout << " 上点= x:" << middle - ltx << "-" << rtx - middle << "  y:" << lty << "-" << rty << endl;
	cout << " 上点= x:" << middle - eyeDetectedRects[0].x - ltx << "-" << eyeDetectedRects[1].x+rtx - middle << "  y:" << eyeDetectedRects[0].y+lty << "-" << eyeDetectedRects[1].y+rty << endl;

	browPoint[2] = Point(llx,lly);//左眼左
	browPoint[7] = Point(rrx,rry);//右眼右
	cout << " 左点= x:" << middle - llx << "-" << rrx - middle << "  y:" << lly << "-" << rry << endl;

	browPoint[3] = Point(lrx,lry);//左眼右
	browPoint[6] = Point(rlx,rly); //右眼左
	cout << " 右点= x:" << middle - lrx << "-" << rlx - middle << "  y:" << lry << "-" << rly << endl;

	for(int i=0;i<8;i++){
		Point ep = browPoint[i];
		cout << "眉毛检测，第" << i << "个点，坐标：" << ep.x <<  "," << ep.y << endl;
	}



	return browPoint;
}
//  返回两个四个点的数组，每组上下左右，第一组左眼，第二组右眼。不能确定边缘的点，为（0，0）.
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
	rightEyeROI_lt.copyTo(rightEyeROI_l);//右眼的左角ROI备份到rightEyeROI_l里
	imwrite("MedianData//t111_30_1.png",rightEyeROI_lt);
	CFace::removeBrow(rightEyeROI_lt,rectr);//把右眼原图里指定矩形里的东西清空
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

	// 取眼睛中心
	lby = CFace::calLastRowOfContour(leftEyeROI(Rect(leftEyeROI_.cols*3/8,0,leftEyeROI_.cols/4,leftEyeROI_.rows-4)));
	rby = CFace::calLastRowOfContour(rightEyeROI(Rect(rightEyeROI_.cols*3/8,0,rightEyeROI_.cols/4,rightEyeROI_.rows-4)));

	cout << "检测眼睛坐标下端：" << lby << " " << rby << endl;

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


	
	//逻辑判断：
	//1、眼边缘不能与眼检测框边缘重合。
	int middletemp = (eyeDetectedRects[0].x+eyeDetectedRects[0].width/2 + eyeDetectedRects[1].x+eyeDetectedRects[1].width/2)/2;//临时中线，这个不对，
	if(llx<=0 && rrx>0) llx = middletemp - (eyeDetectedRects[1].x+rrx-middletemp)-eyeDetectedRects[0].x;//eyeDetectedRects[0].x
	if(rrx<=0 && llx>0) rrx = middletemp + (middletemp - (eyeDetectedRects[0].x+llx)) - eyeDetectedRects[1].x;
	if(lrx<=0 && rlx>0) lrx = middletemp - (eyeDetectedRects[1].x+rlx-middletemp)-eyeDetectedRects[0].x;
	if(rlx<=0 && lrx>0) rlx = middletemp + (middletemp - (eyeDetectedRects[0].x+lrx)) - eyeDetectedRects[1].x;

	cout << llx << " " << lrx << " " << rlx << " " << rrx << endl;
	//2、如果重合了，以脸中线做对称取值。
	//3、如果两侧都与检测框重合，就先设为框吧。以后改为设为-1，替换眼睛时，按模板比例设宽。
	
	//4、左眼右点，与眼左点，跟中线完全对称

	cout << (eyeDetectedRects[1].y + eyeDetectedRects[1].height/2) << " " << (eyeDetectedRects[0].y + eyeDetectedRects[0].height/2) << endl;
	cout << " 右点= x:" <<  lrx << "-" << rlx  << "  y:" << lry << "-" << rly << endl;
	
	//两眼宽度不一样，取大的，等宽。
	//因为取被挡住的那支眼有问题，所以这里先做对称处理，以达到临时好的效果，回头把它改掉，因为这样会导致其中一只眼的偏差，人的两只眼不是一样的。
	int _wLeftEye = lrx - llx;
	int _wRightEye = rrx -rlx;
	if(_wLeftEye > _wRightEye){
		//rly = (eyeDetectedRects[1].height/2) - ((eyeDetectedRects[0].height/2) - lry);//如果对称，原本这个应该是对的，但。。。
		//-----------------------
		rrx = rlx + _wLeftEye;
		//================== 下面是对称处理 ============================
		rry = rly - (lry - lly);
		rtx = rlx + (lrx -ltx);
		rty = rly - (lry - lty);//两眼高度一样，不是上点一样高。
		rby = rly - (lry - lby);
	}
	else {
		llx = lrx - _wRightEye;
		//================== 下面是对称处理 ============================
		lly = lry - (rly - rry);
		ltx = lrx - (rtx - rlx);
		lty = lry - (rly - rty);
		lby = lry - (rly -rby);
	}
	cout << " 右点= x:" <<  lrx << "-" << rlx  << "  y:" << (eyeDetectedRects[0].y )+lry << "-" << (eyeDetectedRects[1].y )+rly << endl;
	/*
	//两眼高度不一样，取大的，等高。
	int _hLeftEye = lry - lty;
	int _hRightEye = rty -rly;
	if(_hLeftEye > _hRightEye)	rty = rly - _hLeftEye ;
	else	lty = lry - _hRightEye;
	*/

	int middle = this->faceMiddleRect.x;

	eyesPoint[1] = Point(0,lby);//左眼下
	eyesPoint[5] = Point(0,rby);//右眼下

	eyesPoint[0] = Point(ltx,lty);//左眼上
	eyesPoint[4] = Point(rtx,rty);//右眼上
	cout << middle << endl;
	cout << " 上点= x:" << middle - ltx << "-" << rtx - middle << "  y:" << lty << "-" << rty << endl;
	cout << " 上点= x:" << middle - eyeDetectedRects[0].x - ltx << "-" << eyeDetectedRects[1].x+rtx - middle << "  y:" << eyeDetectedRects[0].y+lty << "-" << eyeDetectedRects[1].y+rty << endl;

	eyesPoint[2] = Point(llx,lly);//左眼左
	eyesPoint[7] = Point(rrx,rry);//右眼右
	cout << " 左点= x:" << middle - llx << "-" << rrx - middle << "  y:" << lly << "-" << rry << endl;

	eyesPoint[3] = Point(lrx,lry);//左眼右
	eyesPoint[6] = Point(rlx,rly); //右眼左
	cout << " 右点= x:" << middle - lrx << "-" << rlx - middle << "  y:" << lry << "-" << rly << endl;

	for(int i=0;i<8;i++){
		Point ep = eyesPoint[i];
		cout << "第" << i << "个点，坐标：" << ep.x <<  "," << ep.y << endl;
	}

	return eyes;
}

Point* appFace::getNosePoint(Mat noseROI){ //  返回两个点的数组，左右。不能确定边缘的点，为（0，0）.
	nosePoint[0];
	//定位鼻子中点，创建一个1/4高的矩形框

	//左向右扫描，找到到第一个白点，记录下来（最左外边，及它的内边），找到最左内边，
	return nosePoint;
}

Point* appFace::getMouthPoint(Mat mouthROI){ //  返回四个点的数组，上下左右。不能确定边缘的点，为（0，0）.
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

/*将模板按指定三点进行旋转和缩放。因为rotate(mat,angle)在一次旋转后，失去原图大小，不能保持原大小转回去，
所以这里需要先计算缩放，然后一次转向完成。先尝试多出来去掉纯黑色。
*/
Mat appFace::resizeModel2(Mat modelImg,Point ml,Point mr,Point mt,Point mb,Point el,Point er,Point et,Point eb,int mode){
	if(modelImg.cols) cout << "resizeModel，输入图像正常。 " << endl; else  cout << "resizeModel，输入图像为空。 " << endl;
	int xmlength = mr.x - ml.x; // TODO: mr.x值不对
	int ymlength = mr.y - ml.y; // 模型左右点Y值差
	int xelength = er.x - el.x;
	int yelength = er.y - el.y;

	if(mode == 9) {//不旋转，直接宽高缩放。下面的完全了。
		double dwidth = (double)xelength / (double)xmlength;
		double dheight = (double)yelength / (double)ymlength;
		resize(modelImg,modelImg,Size(0,0),dwidth,dheight);
		return modelImg;
	}


	//缩放处理方案二：旋转，再缩放。//这个方案碰到一个问题，需要改写rotate(mat,angle)旋转功能，否则报错。
	// TODO: 先旋模板转到水平，再缩放，再旋转到眼睛角度，再上下缩放，到模板顶点与眼睛顶点相同高度，这时的眼睛模板上下高度是一个近似值。
	double arcMS,arcES;
	arcMS = atan2(abs((double)(mr.y-ml.y)),abs((double)(mr.x-ml.x)));
	arcMS = arcMS*180/CV_PI; // 计算模型对角线原始角度
	arcES = atan2(abs((double)(er.y-el.y)),abs((double)(er.x-el.x)));
	arcES = arcES*180/CV_PI; // //计算人眼对角线角度
	
	//计算眼睛最高点到对角线的高度
	double l_e,l_ea,l_eb,l_ec;
	double l_eat = (el.x-er.x)*(el.x-er.x)+(el.y-er.y)*(el.y-er.y);
	l_ea=sqrt(l_eat);//计算模板对角线长度
	cout << " (el.x-er.x):" << (el.x-er.x) << " (el.y-er.y):" << (el.y-er.y) << " l_ea:" << l_ea << endl;
	double l_ebt = (el.x-et.x)*(el.x-et.x)+(el.y-et.y)*(el.y-et.y);
	l_eb=sqrt(l_ebt);
	double l_ect = (er.x-et.x)*(er.x-et.x)+(er.y-et.y)*(er.y-et.y);
	l_ec=sqrt(l_ect);
	//CosC=(a^2+b^2-c^2)/2ab ;b*CosC=a1;((a^2+b^2-c^2)/2a)=a1;l_e=squr(b^2-((a^2+b^2-c^2)/2a)^2)
	double l_et = l_eb*l_eb-((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea))*((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea));
	l_e=sqrt(l_et);
	cout << " l_e" << l_e;
	//计算模板最高点到对角线的高度
	int l_m,l_ma,l_mb,l_mc;
	double l_mat = (ml.x-mr.x)*(ml.x-mr.x)+(ml.y-mr.y)*(ml.y-mr.y);
	l_ma=sqrt(l_mat);//计算人眼对角线长度
	cout << " (ml.x-mr.x):" << (ml.x-mr.x) << " (ml.y-mr.y):" << (ml.y-mr.y) << " l_ma:" << l_ma << endl;
	double l_mbt = (ml.x-mt.x)*(ml.x-mt.x)+(ml.y-mt.y)*(ml.y-mt.y);
	l_mb=sqrt(l_mbt);
	double l_mct = (mr.x-mt.x)*(mr.x-mt.x)+(mr.y-mt.y)*(mr.y-mt.y);
	l_mc=sqrt(l_mct);
	double l_mt = l_mb*l_mb-((l_ma*l_ma+l_mb*l_mb-l_mc*l_mc)/(2*l_ma))*((l_ma*l_ma+l_mb*l_mb-l_mc*l_mc)/(2*l_ma));
	l_m=sqrt(l_mt);
	//相除计算缩放比例。这是一个近似值，最佳情况是模板和人眼都旋转平，再标记最高点。
	double _dh_em = (double)l_e/(double)l_m;//缺参数了。换个简单的算法，先补上，在下面。
	double _dl_em = (double)l_ea/(double)l_ma;
	cout << " l_eb:" << l_eb << " l_ec:" << l_ec << " l_mb:" << l_mb << " l_mc:" << l_mc << " " << endl;
	cout << " _dh_em:" << _dh_em << " (l_e:" << l_e << ")/(l_m:" << l_m << ")|   _dl_em:" << _dl_em << " (l_ea:" << l_ea << ")/(l_ma:" << l_ma << ")"<< endl;
	
	//判断旋转方向
	int directe,directm;
	if(el.y>er.y) directe = 1; // 左上右下
	if(el.y<=er.y) directe = 0; // 右上左下
	if(ml.y>mr.y) directm = 1; // 左上右下
	if(ml.y<=mr.y) directm = 0; // 右上左下
	//将模板旋转到水平

	cout << "arcMs:" << arcMS << endl;

	if(directm)	
		CFace::rotate(modelImg,-1*arcMS);
	else 
		CFace::rotate(modelImg,arcMS); 
	//cout << modelImg.cols << " " << modelImg.rows << endl;
	//缩放，到眼睛的左右点长度、高度。
	if(mode == 1){
		resize(modelImg,modelImg,Size(0,0),_dl_em,_dh_em);
		//旋转到眼睛倾斜角度
		if(directe) CFace::rotate(modelImg,-1*arcES); else  CFace::rotate(modelImg,arcES);
		imwrite("MedianData//rotateEye.jpg",modelImg);
		return modelImg;
	}

			cout << " 旋转完成，准备缩放。。。" << endl;


	//cout << modelImg.cols << " " << modelImg.rows << endl;
	resize(modelImg,modelImg,Size(0,0),_dl_em,1);
	//旋转到眼睛倾斜角度
	if(directe) CFace::rotate(modelImg,-1*arcES); else  CFace::rotate(modelImg,arcES);

	//int model_h = mt.y -mb.y;
	//int eye_h = et.y - eb.y;
	//_dh_em = (double)eye_h/(double)model_h;//缺参数了。换个简单的算法，先补上。
	int model_h = mt.y -mb.y;
	int eye_h = et.y - eb.y;
	_dh_em = (double)eye_h/(double)model_h;//缺参数了。换个简单的算法，先补上。
	cout << model_h << " " << eye_h << " " << _dh_em << endl;
	
	resize(modelImg,modelImg,Size(0,0),1,_dh_em);
	resizeEyeRate = _dh_em;
	imwrite("MedianData//rotateEye.jpg",modelImg);
	return modelImg;
}


/*将模板按指定三点进行旋转和缩放。因为rotate(mat,angle)在一次旋转后，失去原图大小，不能保持原大小转回去，
所以这里需要先计算缩放，然后一次转向完成。先尝试多出来去掉纯黑色。
*/
Mat appFace::resizeModel(Mat modelImg,Point ml,Point mr,Point mt,Point mb,Point el,Point er,Point et,Point eb,int mode){
	if(modelImg.cols) cout << "resizeModel，输入图像正常。 " << endl; else  cout << "resizeModel，输入图像为空。 " << endl;
	if(mode == 8) {//不旋转，直接宽高缩放。下面的完全了。
		int xmlength = mr.x - ml.x; // TODO: mr.x值不对
		int ymlength = mb.y - mt.y; // 模型左右点Y值差
		int xelength = er.x - el.x;
		int yelength = eb.y - et.y;

		double dwidth = (double)xelength / (double)xmlength;
		double dheight = (double)yelength / (double)ymlength;
		cout << dwidth << " " << dheight << endl;
		resize(modelImg,modelImg,Size(0,0),dwidth,dheight);
		return modelImg;
	}


	//缩放处理方案二：旋转，再缩放。//这个方案碰到一个问题，需要改写rotate(mat,angle)旋转功能，否则报错。
	// TODO: 先旋模板转到水平，再缩放，再旋转到眼睛角度，再上下缩放，到模板顶点与眼睛顶点相同高度，这时的眼睛模板上下高度是一个近似值。
	double arcMS,arcES;
	arcMS = atan2(abs((double)(mr.y-ml.y)),abs((double)(mr.x-ml.x)));
	arcMS = arcMS*180/CV_PI; // 计算模型对角线原始角度
	arcES = atan2(abs((double)(er.y-el.y)),abs((double)(er.x-el.x)));
	arcES = arcES*180/CV_PI; // //计算人眼对角线角度
	
	//计算眼睛最高点到对角线的高度
	double l_e,l_ea,l_eb,l_ec;
	double l_eat = (el.x-er.x)*(el.x-er.x)+(el.y-er.y)*(el.y-er.y);
	l_ea=sqrt(l_eat);//计算模板对角线长度
	cout << " (el.x-er.x):" << (el.x-er.x) << " (el.y-er.y):" << (el.y-er.y) << " l_ea:" << l_ea << endl;
	double l_ebt = (el.x-et.x)*(el.x-et.x)+(el.y-et.y)*(el.y-et.y);
	l_eb=sqrt(l_ebt);
	double l_ect = (er.x-et.x)*(er.x-et.x)+(er.y-et.y)*(er.y-et.y);
	l_ec=sqrt(l_ect);
	//CosC=(a^2+b^2-c^2)/2ab ;b*CosC=a1;((a^2+b^2-c^2)/2a)=a1;l_e=squr(b^2-((a^2+b^2-c^2)/2a)^2)
	double l_et = l_eb*l_eb-((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea))*((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea));
	l_e=sqrt(l_et);
	cout << " l_e" << l_e;
	//计算模板最高点到对角线的高度
	int l_m,l_ma,l_mb,l_mc;
	double l_mat = (ml.x-mr.x)*(ml.x-mr.x)+(ml.y-mr.y)*(ml.y-mr.y);
	l_ma=sqrt(l_mat);//计算人眼对角线长度
	cout << " (ml.x-mr.x):" << (ml.x-mr.x) << " (ml.y-mr.y):" << (ml.y-mr.y) << " l_ma:" << l_ma << endl;
	double l_mbt = (ml.x-mt.x)*(ml.x-mt.x)+(ml.y-mt.y)*(ml.y-mt.y);
	l_mb=sqrt(l_mbt);
	double l_mct = (mr.x-mt.x)*(mr.x-mt.x)+(mr.y-mt.y)*(mr.y-mt.y);
	l_mc=sqrt(l_mct);
	double l_mt = l_mb*l_mb-((l_ma*l_ma+l_mb*l_mb-l_mc*l_mc)/(2*l_ma))*((l_ma*l_ma+l_mb*l_mb-l_mc*l_mc)/(2*l_ma));
	l_m=sqrt(l_mt);
	//相除计算缩放比例。这是一个近似值，最佳情况是模板和人眼都旋转平，再标记最高点。
	double _dh_em = (double)l_e/(double)l_m;//缺参数了。换个简单的算法，先补上，在下面。
	double _dl_em = (double)l_ea/(double)l_ma;
	cout << " l_eb:" << l_eb << " l_ec:" << l_ec << " l_mb:" << l_mb << " l_mc:" << l_mc << " " << endl;
	cout << " _dh_em:" << _dh_em << " (l_e:" << l_e << ")/(l_m:" << l_m << ")|   _dl_em:" << _dl_em << " (l_ea:" << l_ea << ")/(l_ma:" << l_ma << ")"<< endl;
	
	//判断旋转方向
	int directe,directm;
	if(el.y>er.y) directe = 1; // 左上右下
	if(el.y<=er.y) directe = 0; // 右上左下
	if(ml.y>mr.y) directm = 1; // 左上右下
	if(ml.y<=mr.y) directm = 0; // 右上左下
	//将模板旋转到水平

	cout << "arcMs:" << arcMS << endl;

	if(directm)	
		CFace::rotate(modelImg,-1*arcMS);
	else 
		CFace::rotate(modelImg,arcMS); 
	//cout << modelImg.cols << " " << modelImg.rows << endl;
	//缩放，到眼睛的左右点长度、高度。
	if(mode == 1){
		resize(modelImg,modelImg,Size(0,0),_dl_em,_dh_em);
		//旋转到眼睛倾斜角度
		if(directe) CFace::rotate(modelImg,-1*arcES); else  CFace::rotate(modelImg,arcES);
		imwrite("MedianData//rotateEye.jpg",modelImg);
		return modelImg;
	}

			cout << " 旋转完成，准备缩放。。。" << endl;


	//cout << modelImg.cols << " " << modelImg.rows << endl;
	resize(modelImg,modelImg,Size(0,0),_dl_em,1);
	//旋转到眼睛倾斜角度
	if(directe) CFace::rotate(modelImg,-1*arcES); else  CFace::rotate(modelImg,arcES);

	//int model_h = mt.y -mb.y;
	//int eye_h = et.y - eb.y;
	//_dh_em = (double)eye_h/(double)model_h;//缺参数了。换个简单的算法，先补上。
	int model_h = mt.y -mb.y;
	int eye_h = et.y - eb.y;
	_dh_em = (double)eye_h/(double)model_h;//缺参数了。换个简单的算法，先补上。
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

