#include "face.h"
#include <iostream>
#include <stdio.h>  
#define M_PI       3.14159265358979323846

using namespace std;
CFace::CFace(){
}
CFace::~CFace(){
}


//����һ�������һ��ֱ������ƽ��ͶӰ����
double CFace::calParallelDistance(Point el,Point er,Point et){
	//et�����Ͼ���el�ľ��룬et1�����Ͼ���el�ľ���
	double etel_v = calVerticalDistance(el,er,et);
	double v_etel = (et.x-el.x)*(et.x-el.x) + (et.y-el.y)*(et.y-el.y);
	double etel = sqrt(v_etel);
	double etelp = sqrt(etel*etel - etel_v*etel_v);
	return etelp;
}
//�������������һ��ֱ�ߵ�ƽ�о���
double CFace::calParallelDistance(Point el,Point er,Point et,Point et1){
	double etelp = calParallelDistance(el,er,et);
	double et1elp = calParallelDistance(el,er,et1);
	return etelp - et1elp;
}

//����ָ���㵽ֱ�ߵĸ߶ȡ����ָ������ֱ�߷�����࣬�����Ϊ��������Ϊ����
double CFace::calVerticalDistance(Point el,Point er,Point et){
	double l_e,l_ea,l_eb,l_ec;
	double l_eat = (el.x-er.x)*(el.x-er.x)+(el.y-er.y)*(el.y-er.y);
	l_ea=sqrt(l_eat);//����ģ��Խ��߳���
	double l_ebt = (el.x-et.x)*(el.x-et.x)+(el.y-et.y)*(el.y-et.y);
	l_eb=sqrt(l_ebt);
	double l_ect = (er.x-et.x)*(er.x-et.x)+(er.y-et.y)*(er.y-et.y);
	l_ec=sqrt(l_ect);
	//CosC=(a^2+b^2-c^2)/2ab ;b*CosC=a1;((a^2+b^2-c^2)/2a)=a1;l_e=squr(b^2-((a^2+b^2-c^2)/2a)^2)
	double l_et = l_eb*l_eb-((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea))*((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea));
	l_e=sqrt(l_et);//�ߵ㵽�Խ��ߵĸ߶�

	//�Ƚ�etƽ���Ƶ�erһ��
	//��elΪԭ�㣬����ԭ����������ĽǶȣ���������࣬�������Ҳࡣ
	//���elΪԭ�㣬erΪX�������򣬼���et�������򣬾�֪���������ˡ�
	Point ert = Point(er.x-el.x,er.y-el.y);
	Point err = Point(et.x-el.x,et.y-el.y);
	int _2to1=1,_4to1=1,t2to1=1,t4to1=1;
	if(err.x<0) { err = Point(-1*err.x,err.y);ert = Point(-1*ert.x,ert.y);_2to1=-1;}//12����ת
	if(err.y<0) { err = Point(err.x,-1*err.y);ert = Point(ert.x,-1*ert.y);_4to1=-1;}
	if(ert.x<0) { ert = Point(-1*ert.x,ert.y);t2to1=-1;}
	if(ert.y<0) { ert = Point(ert.x,-1*ert.y);t4to1=-1;}

return 0;
}


//��ͼ��ָ���������š�sp,Դ�㣬tp����
void CFace::resizeMat(Mat src,Mat tar,Point sp,Point tp,double ratio){
	//��ָ��������һ�����Σ���סĿ������
	double rWidth,rHeight;//���ο��
	//�����ĸ����������ߵĴ�ֱ����
	double lt,rt,lb,rb; 
	Point ltp=Point(0,0) , rtp=Point(src.cols,0) ,lbp=Point(0,src.rows),rbp=Point(src.cols,src.rows);
	lt = calVerticalDistance(sp,tp,ltp);
	rt = calVerticalDistance(sp,tp,rtp);
	lb = calVerticalDistance(sp,tp,lbp);
	rb = calVerticalDistance(sp,tp,rbp);
	double sortT[4] = {lt,rt,lb,rb};
	Point sortP[4] = {ltp,rtp,lbp,rbp};
	Point maxPoint,minPoint;Point middlePoints[2];
	double max=0,min=-1;
	for(int i=0;i<4;i++){	if(max <sortT[i]) max = sortT[i];	}
	for(int i=0;i<4;i++){ 
		if(max=sortT[i] && sortP[i].x>=0 && sortP[i].y>=0) { // ȡ�����ֵ�����������
			maxPoint=Point(sortP[i].x,sortP[i].y);
			sortP[i]=Point(-1,-1);
			break;
		}
	}
	min = sortT[0];
	for(int i=0;i<4;i++){	if(min > sortT[i]) min = sortT[i];	}
	for(int i=0;i<4;i++){ 
		if(min=sortT[i] && sortP[i].x>=0 && sortP[i].y>=0) { // ȡ����Сֵ�����������
			minPoint=Point(sortP[i].x,sortP[i].y);
			sortP[i]=Point(-1,-1);
			break;
		}
	}
	int _middleP_t = 0;
	for(int i=0;i<4;i++){
		if(sortP[i].x>0){
			middlePoints[_middleP_t] = Point(sortP[i].x,sortP[i].y);
			_middleP_t++;
		}
	}

	//�����ԽǶ��㵽���ߵĴ�ֱ���룬Y���롣

	//����ͬ�򶥵������ߵ�ƽ�о��룬X����

	//����һ������

	//������ͼ���ӳ�䵽���ƽ�о�����

	//��������

	//����һ����סת��ԭ�Ƕȵľ���

	//�ٽ�ÿ�����Ӱ���ȥ

}

//���ͼ��eyeMask�У�ָ������r����ס�����ݣ��Լ���Щ���ݵ���ͨ����
void CFace::removeBrow(Mat eyeMask,Rect r){ // �趨�۾�ROI����1/6,һ�������üë�򲻰����۾���
	// cout << " ִ�� removeBrow " << r.x << " " << r.y << " " << r.width << " " << r.height << endl;
	absROI(eyeMask);
	// �����1/6�����е�һ����ͨ��
	int b=0,f=255;
	Mat tar;eyeMask.copyTo(tar); // ����һ��TAR����Ϊ��ʱ������������ղ�����ȡ�����Ķ�����

	Mat rt = eyeMask(r);//Mat rtt;rt.copyTo(rtt);rtt.setTo(Scalar(b));
	if(debug){
		imwrite("MedianData//BfremovedMASK.png",rt);
		imwrite("MedianData//BfremovedSource.png",tar);
	}

	for(int _rowrt=0;_rowrt<rt.rows;_rowrt++){
		for(int _colrt=0;_colrt<rt.cols;_colrt++){
			int _row = _rowrt + r.y;
			int _col = _colrt + r.x;
			int index = _row*eyeMask.cols+_col; // ��eyeMask�ж�λ�����ڵĵ�
			//int index = _col*eyeMask.cols+_row; // ��eyeMask�ж�λ�����ڵĵ�
			if(eyeMask.at<uchar>(_row,_col)){ //���eyeMask���������ֵ�������д���nineBox�Ѿ�����������Ҳ�ִ���ˡ������ﻹ���ж�����ֵ����������ֵ��
			//if((uchar)eyeMask.data[index] > 0){ //���eyeMask���������ֵ�������д���nineBox�Ѿ�����������Ҳ�ִ���ˡ������ﻹ���ж�����ֵ����������ֵ��
				//if(debug) cout << " removeBrow: found white point:(" << _col << "," << _row << ") " << "(" << _colrt << "," << _rowrt << ")" << endl;
				tar.setTo(Scalar(b));	// �����ʱ����TAR��
				nineBox(eyeMask,tar,_col,_row,true); // ��eyeMask���������ӵĿ��ó������ŵ�TAR�
				imwrite("MedianData//removed1.png",tar);
				imwrite("MedianData//removed2.png",eyeMask);
			}
		}
	}
	//imwrite("MedianData//eyeMask.png",eyeMask);
	//imwrite("MedianData//removed.png",eyeMask);

}
Mat CFace::createROI(Mat m1,string name,int pre,int mode,int range){ // mԴͼ��pre��Ԥ����0���ޣ�1��ֱ��ͼ��mode��1��cvThreshold��2��cvAdaptiveThreshold��3��Canny��Ե��⡣
	Mat m;
	if(m1.channels()>1){
		cvtColor( m1, m, CV_BGR2GRAY );
	} else
		m = m1;
	imwrite("MedianData//roi12.png",m);

		//�ٳ�Ŀ��
		Mat _gray;
		Mat mROI;
		//�ȵ���ԭͼƽ������
		IplImage* _imageBgr = &IplImage(m);
		Mat _tar ;
		m.copyTo(_tar);
		//imwrite("MedianData//"+name+"_tar.png", _tar);
		IplImage* _imageTar = &IplImage(_tar);
		set_avg_gray(_imageTar,_imageTar,(double)128.0);//����ƽ�⴦��
		imwrite("MedianData//"+name+"_128Light.jpg", _tar);
		// GaussianBlur(m,_tar,Size(5,5),0,0);
		//cvSmooth(_imageBgr,_imageTar);
		_tar.copyTo(_gray);
		Mat mt2 = Mat(_gray);
		IplImage* eye_gray_r = &IplImage(_gray);
		imwrite("MedianData//"+name+"_le1.jpg", mt2);

		int offSet = 0 ;
		if(m.cols > m.rows)
			offSet = ((float)(m.rows));
		else
			offSet = ((float)(m.cols));

		if(pre == 1){
			//�۾�ֱ��ͼ��Ч�����á���Ϊ�۾��ĺڰ��Ѿ��������ˡ����Ӻ������Ҫ.
			equalizeHist( _gray, _gray );
			imwrite("MedianData//"+name+"le2.jpg", mt2);
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
			while(1){
				if(range % 2 == 1 && range > 1){break;}
				else range++;
				if(range>100) return m;
			}

			if(mode == 2) // ��ֵ�㷨
				cvAdaptiveThreshold(graySource, grayDst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, range, range);

			if(mode == 3) {// ��˹�㷨
				//cvAdaptiveThreshold(graySource, grayDst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, range, range);
				cvAdaptiveThreshold(graySource, grayDst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, range, range);
			}
			
			reverseROI(grayDst); // ת�ɺڵװ���
			imwrite("MedianData//roi13.png", Mat(grayDst));
			filterBlock(grayDst,5,5,255); // ���˴�С��Ҫ�������
		}

		Mat mt = Mat(grayDst);
		absROI(mt);
		imwrite("MedianData//"+name+".jpg", mt);
		return mt;
}

void CFace::filterBlock(Mat frame,int w,int h,int v){ // ���˵�С��(w,h)�Ķ����顣stack overflow!!
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
				//// cout << " col: " << _col << " " << _row << endl;
				tar.setTo(Scalar(b)); // ��ɱ���ɫ��
				nineBox(src,tar,_col,_row,true);//��������nineBox
				//Mat reverse; tar.copyTo(reverse);reverse.setTo(Scalar(b));
				imwrite("MedianData//filterBlockReverse.png",tar);
				Rect r = getROIRect(tar);
				//reverseROI(tar);
				//��תSRC��ʹ�������Ͽ��������
				int width = r.width;//this->calLastColOfContour(reverse) - this->calFirstColOfContour(reverse);
				int height = r.height;//this->calLastRowOfContour(reverse) - this->calFirstRowOfContour(reverse);
				//// cout <<"width:"<< width << " height:" << height << " x:" << r.x << " y:" << r.y << " w:" << w << " h:" << h << endl;
				if(width >= w && height >= h) {
					copyROI(tar,frame); // ������ӵ�һ�������ָ����ߣ��͸��ƻ�ȥ��
					imwrite("MedianData//frameBlock.png",frame);
				}
			}
		}
	}
	//cvReleaseMat(tar);
	//reverseROI(frame);
	return;
}

//ʹ��smoothģʽ����P1P2Ϊ����
void CFace::smoothRect(Mat &img,Rect r,Point p1,Point p2,int deep){
	int l,s;
	double dx=0.0,dy=0.0;
	if(img.cols>img.rows) {
		l= img.cols;
		s=img.rows; 
		dy=1;dx=l/s;
	}else{ 
		l=img.rows;
		s=img.cols;
		dx=1;dy=l/s;
	}
	
	Mat mt;
	//for(int i=0;i<s/2;i++)
	if(r.width>1 && r.height>1 && r.width-2*deep*dx>0 && r.height-2*deep*dy>0 && r.x+deep*dx<r.width && r.y+deep*dy<r.height )
	{
		Rect newR = Rect(r.x+deep*dx,r.y+deep*dy,r.width-2*deep*dx,r.height-2*deep*dy);
		img(newR).copyTo(mt);

		IplImage* imgMs = &IplImage(img);
		for(int times=0;times<4;times++)
		{
			cvSmooth(imgMs,imgMs);
		}
		mt.copyTo(img(newR));
		smoothRect(img,newR,p1,p2,deep);
	}
}

//ʹ��smoothģʽ����P1P2Ϊ����
void CFace::rSmoothRect(Mat img,Rect r,Point p1,Point P2,int deep){
	//if(
}



//����MASK������ͨ���ˡ����frame�е�һ������MASK�ڣ���ô����MASK֮�����ͨ��Ҳ������(��MASK�ڵģ�����ͨ����)������ɾ����
void CFace::filterBlock(Mat frame,Mat mask,bool blackBackground){
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
					// cout << " remove " << endl;
				}
			}
		}
	}
	if(!blackBackground){reverseROI(ret);reverseROI(mask);	}
	ret.copyTo(frame);
	//frame = ret;
}

// �ж�t�еĵ��Ƿ�����mask��Χ�ڵġ�
bool CFace::joinMask(Mat t,Mat mask,bool blackBackground){ 
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
				//// cout << _tt.at<uchar>(_row,_col) << " , " << _row << " , " << _col << " , " << _maskt.at<uchar>(_row,_col) << endl;
				if( _maskt.at<uchar>(_row,_col)){//.data[index] < 50){
					//// cout << _tt.at<uchar>(_row,_col) << " . " << _row << " . " << _col << " . " << _maskt.at<uchar>(_row,_col) << endl;
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

int CFace::getROIWidth(Mat _roi,int mode){ // mode:0���Ǻڵף�1�ǰ׵�
	return 0;
}
int CFace::getROIHeight(Mat _roi,int mode){
	return 0;
}

// ��ROI��m�У���XY�����ӵĿ飬��m���ó������ŵ�t�С�ִ�к�m�У���û��������ֵ�ˡ�
Mat CFace::nineBox(Mat m,Mat t,int x,int y,bool blackBackground){
	int b=0,f=255; // �趨ǰ�󾰵�ɫֵ

	int left,right,top,buttom;//��õ������ӵ�9������������ҵ�����
	if(x-1>=0) left = x-1;else left = x;
	if(x+1<m.cols) right = x+1;else right = x;
	if(y-1>=0) top = y-1;else top = y;
	if(y+1<m.rows) buttom = y+1;else buttom = y;
	Rect r=Rect(left,top,right-left+1,buttom-top+1);// 9������ο򣬿����˱�Ե���⡣
	//���Ƴ�һ��9����鵽��ʱ����_submat�У����ڼ�¼ԭ9�����ĸ�����ֵ��������������Ψһһ��������Mat�ĵط���
	Mat submat = m(r);Mat _submat;submat.copyTo(_submat);
	if(!t.cols){	m.copyTo(t);	t.setTo(Scalar(b));	} //���Ŀ��Ϊ�գ��ͳ�ʼ��һ����
	Mat submatt = t(r);  //�ҳ�Ŀ������ͬλ�õ�9����飬ȡ����ͷ��
	//cvAdd(submatt,_submat,submatt);//дĿ��
	if(submatt.cols<1){
		cout << " " << endl;
	}
	if(submat.cols<1){
		cout << " " << endl;
	}
	submatt = submatt + submat;//дĿ�ꡣ����������Ļ���Ŀ����
	submat.setTo(Scalar(b));//���Դ�����ڻ������������⣬��Դ�������Ӧ�㣬û�б���գ�����ѭ�������֡�
	//cvSetZero(submat);//���Դ������ĺ���ûִ�У���������һ�Ρ�

	for(int _r=0;_r<r.height;_r++){//�жϵݹ�
		for(int _c=0;_c<r.width;_c++){
			if(_submat.at<uchar>(_r,_c)){ // 
				int xt = left + _c;
				int yt = top + _r;
				nineBox(m,t,xt,yt,true);
			}
		}
	}
	//_submat.release();
	return m;
}

//ȡSRC��ָ���㣨startX,startY������ͨMASK���������Ƴ���TAR�С�Ҫ����ʼʱ��ָ������ǰ��ֵ��ͼ��Ϊ�׵׺��֡�
Mat CFace::nineBox(Mat frameSrc,Mat frameTar,int startX,int startY){
	return nineBox(frameSrc,frameTar,startX,startY,true);
}

Rect CFace::getROIRect(Mat src){
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

Rect CFace::copyROI(Mat src,Mat tar){//��src�����0�ĵ㣬���Ƶ�tar��.
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
void CFace::absROI(Mat roi){
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
void CFace::reverseROI(Mat roi){
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
Vec3b CFace::kcvBGR2HSV(Vec3b img)  
{  
	Vec3b ret = Vec3b();
	//˳����ܴ���
	int b=img[0];  
	int g=img[1];  
	int r=img[2];  
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

Vec3b CFace::kcvHSV2BGR(Vec3b img)  
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
	//BGR
	ret[2]=b+m;  
	ret[1]=g+m;  
	ret[0]=r+m;  

	return ret;  
}  

Vec3b CFace::kcvBGR2HSL(Vec3b img) {

		Vec3b ret = Vec3b();

	int R=img[0];  
	int G=img[1];  
	int B=img[2];  
	
	int H,S,L;
	
    double Max,Min,del_R,del_G,del_B,del_Max;

    Min = min(R, min(G, B));    //Min. value of BGR
    Max = max(R, max(G, B));    //Max. value of BGR
    del_Max = Max - Min;        //Delta BGR value

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

Vec3b CFace::kcvHSL2BGR(Vec3b img){

    double R,G,B;
	double H,S,L;
	H = img[0];
	S = img[1];
	L = img[2];

    double var_1, var_2;
    if (S == 0)                       //HSL values = 0 �� 1
    {
        R = L * 255.0;                   //BGR results = 0 �� 255
        G = L * 255.0;
        B = L * 255.0;
    }
    else
    {
        if (L < 0.5) var_2 = L * (1 + S);
        else         var_2 = (L + S) - (S * L);

        var_1 = 2.0 * L - var_2;

        R = 255.0 * Hue2BGR(var_1, var_2, H + (1.0 / 3.0));
        G = 255.0 * Hue2BGR(var_1, var_2, H);
        B = 255.0 * Hue2BGR(var_1, var_2, H - (1.0 / 3.0));
    }

	Vec3d ret = Vec3d(R,G,B);
	return ret;
}

double CFace::Hue2BGR(double v1, double v2, double vH)
{
    if (vH < 0) vH += 1;
    if (vH > 1) vH -= 1;
    if (6.0 * vH < 1) return v1 + (v2 - v1) * 6.0 * vH;
    if (2.0 * vH < 1) return v2;
    if (3.0 * vH < 2) return v1 + (v2 - v1) * ((2.0 / 3.0) - vH) * 6.0;
    return (v1);
}

int CFace::averageLight(Mat faceSampleBGR){
	
	double count = 1.0;
	double lightV = 0.0;
	for(int i=0;i<faceSampleBGR.rows;i++){
		Vec4b *colDataBGRA = faceSampleBGR.ptr<Vec4b>(i);
		Vec3b *colDataBGR = faceSampleBGR.ptr<Vec3b>(i);
		for(int j=0;j<faceSampleBGR.cols;j++){
			int a = colDataBGRA[j][3];
			// cout << i << "  " << j << "  " << a << endl;
			if (a > 10)
			{
				//ת��������
				Vec3d p = kcvBGR2HSL(colDataBGR[j]);
				//�ۼ�
				count++;
				lightV = lightV + p[2];
			}
		}
	}
	int ret = (int)lightV / count;
	return ret;
}

void CFace::adjustLight(Mat faceSampleBGR,float _v){
	Vec3b *bgra_frame_data = faceSampleBGR.ptr<Vec3b>(0);
	for(int i=0;i<faceSampleBGR.rows;i++){
		Vec4b *colDataBGRA = faceSampleBGR.ptr<Vec4b>(i);
		Vec3b *colDataBGR = faceSampleBGR.ptr<Vec3b>(i);
		for(int j=0;j<faceSampleBGR.cols;j++){
			int a = colDataBGRA[j][3];
			if (a > 10)
			{
				//ת��������
				Vec3d p = kcvBGR2HSL(colDataBGR[j]);
				p[2] = (int)p[2]*_v;
				p = kcvHSL2BGR(p);
				bgra_frame_data[i*faceSampleBGR.rows+j] = p;
			}
		}
	}

}

double CFace::get_avg_gray(IplImage *img)
{
    IplImage *gray ;//= cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	if(img->nChannels>1){
		gray = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		cvCvtColor(img,gray,CV_BGR2GRAY);
		CvScalar scalar = cvAvg(gray);
		cvReleaseImage(&gray);
		return scalar.val[0];
	}
	else if(img->nChannels==1){
		//Mat mgray;
		//Mat(img).copyTo(mgray);
		if(img->depth == 8)
			gray = img;//&IplImage(mgray);
		else
			cout << " ʲô����� " << endl;
		imwrite("MedianData//tttt.png", Mat(gray));
		CvScalar scalar = cvAvg(gray);
		return scalar.val[0];
	}
}

void CFace::set_avg_gray(IplImage *img2,IplImage *out1,double avg_gray)
{
	double prev_avg_gray ;
	IplImage *img1,*img, *out;
	vector<Mat> h;
	Mat mimg1;
	Mat(img2).copyTo(mimg1);

	out = cvCreateImage( cvGetSize(img2),IPL_DEPTH_8U, 1 );  
	img1 = &IplImage(mimg1);

	Mat mimg,mout;
	if(img1->nChannels==3){//��Ϊ��BGR����
		kcvBGR2HSV(Mat(img1));//ת��HSV
		split(Mat(img1),h);
		img=&IplImage(h[2]);
	}
	if(img1->nChannels==1)//��Ϊ�ǻҶ�����
		img = img1;

	prev_avg_gray = get_avg_gray(img);
	if(prev_avg_gray>0 && avg_gray > 0){
		// cout << (double)(avg_gray/prev_avg_gray) << endl;
		cvConvertScale(img,out,(double)(prev_avg_gray/avg_gray));
	}
	else
		cout << "Div by zero!" << endl;

	if(img1->nChannels==3){//��Ϊ��BGR����
		merge(h,mout);
		kcvHSV2BGR(mout);//ת��BGR
		cvReleaseImage(&out);
		out1 = &IplImage(mout);
	}
	if(img1->nChannels==1) {//��Ϊ�ǻҶ�����
		cvCopy(out,out1,NULL);
		cvReleaseImage(&out);  
	}
	cout << "" << endl;
}

void CFace::cvSkinHSV(IplImage* src,IplImage* dst)    
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
			//// cout << h << "  " << w << endl;
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
    cvReleaseImage(&hsv);     
	
}    

void CFace::cvSkinSegment(IplImage* img, IplImage* mask){
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

void CFace::cvSkinYUV(IplImage* src,IplImage* dst)    
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
    cvReleaseImage(&ycrcb);     
}    
void CFace::filterMirrorBackground(Mat &resultImage){
	//------------------------
	//Add filter mirror here, this->imageContourSM is the contour of human, resultImage is the original image
	//�˾�������
	//this->imageContourSM;

	//------------------------
}

//����ͼ��
void CFace::brightnessContrast(Mat &resultImage, double alpha, double beta){
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

Mat CFace::changeChanel(Mat img,int _3To4){
	if(_3To4 ==0){
		cvtColor(img, img, CV_BGRA2BGR);
	}
	return img;
}

Mat CFace::lightBalanceFrame(Mat src,Mat model,Mat ret){

	//����ƽ�⴦��
	IplImage* imageModel = &IplImage(model);
	IplImage* imageSrc = &IplImage(src);
	src.copyTo(ret);
	IplImage* imageRet = &IplImage(ret);
	double gFace = get_avg_gray(imageModel);
	double gBgr = get_avg_gray(imageSrc);
	//Ϊ�˵�����������ע�͵���
	//set_avg_gray(imageRet,imageRet,gFace*0.7);
	//set_avg_gray(imageRet,imageRet,128);
	
	//�����Ƿ���͸������Ӱ��һ����ƽ��ֵ
	// cout << "gFace:" << gFace << " gBgr:" << gBgr << endl;

	return ret;
}

void skinBalance(Mat body,Mat face){
	/*
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
    cvReleaseImage(&imageSkin);     
	*/

}
//������ת��ͨ���Ŵ󻭲���������ԭͼƬ��С��������ת����ʧȥ��ԭͼƬ�Ĵ�С��ת��ȥʱ��ͼƬ���ˡ�Ҳ���ܿ��ǣ�ͨ�����㣬ֱ����ת���ŵ�Ŀ��ߴ硣
Mat CFace::rotate(Mat srcm,double angle){
	//	cout << " ��תģ�͡�����" << angle << "�� " <<  endl;

	//���Դ��4ͨ����ͼ��ת����ͨ���ȣ�����ת������ת����4ͨ����
	Mat src_temp;
	IplImage* srct =  &IplImage(srcm);//��MATתΪIplImage
	int chanelNum = srct->nChannels;
	IplImage* a_plane;
	//	cout << " ��תģ�͡�����1 " <<  endl;

	//if(srcm.cols)  cout << " ����ͼ��Ϊ�� " << endl;  else cout << " ����ԴͼΪ�� " << endl;

	if( chanelNum == 4){
		 a_plane = cvCreateImage(cvGetSize(srct),8,1);
		 cvCvtPixToPlane(srct,0,0,0,a_plane);      //��ȡ͸��ͨ��
		cvtColor(srcm, src_temp, CV_BGRA2BGR);
		//cvtColor(src , src_temp , CV_BGRA2BGR);
	} else {
		src_temp = srcm;
	}
	//	cout << " ��תģ�͡�����2 " <<  endl;

	IplImage* src =  &IplImage(src_temp);//��MATתΪIplImage
	//���Ƕ�ֵת��Ϊ��1������ֵ
	int angle1 = angle;
	angle = abs((int)angle % 180); 
	//// cout << " ang:" << angle << " " << angle1 << endl;
	if (angle > 90) { 
		angle = 90 - ((int)angle % 90); 
	} 
	if(angle<0) angle = 360+angle;

	//������ת��������ֵ
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

	//��������תֵ����ȥ������ֻ����+-90�ȵ�
	if(angle1!=0) angle = (abs(angle1)/angle1)*angle;


	//����һ���»����������Ϊ0���ٴ���һ���Խ�����ô��������Σ������Ϊ0.
	dst = cvCreateImage(cvSize(width, height), src->depth, src->nChannels); 
	cvZero(dst); 
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels); 
	cvZero(temp); 
	
	//���öԽ��߻����У����м�ԭͼ����ô������ΪROI������ԭͼ���Ƶ���������ڡ�
	cvSetImageROI(temp, cvRect(tempX, tempY, src->width, src->height)); 
	cvCopy(src, temp, NULL); 
	cvResetImageROI(temp); 

	//������ת����
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
	//// cout << "rotateMtr:" << m[0] << " " << m[1] << " " << m[3] << " " << m[4] << " " << m[2] << " " << m[5] << " | angle=" << angle << endl;
	// 
	CvMat M = cvMat(2, 3, CV_32F, m); 
	//ִ��ͼ����ת�����Խ��ߴ�ͼ��ת�����м�DST��ô��鸴�Ƶ�DST�С�
	cvGetQuadrangleSubPix(temp, dst, &M); 
	cvReleaseImage(&temp); 

	//cout << " ��ʼ�ָ���ͨ����ֵ " <<  endl;
	Mat ret = Mat(dst);

	if( chanelNum == 4){
		Mat a_m = Mat(a_plane);
		rotate(a_m,angle);
		cvtColor(ret, ret, CV_BGR2BGRA);

		for(int _row=0;_row<ret.rows;_row++){
			for(int _col=0;_col<ret.cols;_col++){
				int index = _row*ret.cols+_col;
				Vec4b _data = ret.data[index];
				//�����á��Ƚ�͸����ȥ�����һ���͸����ȫΪ0.
				_data[3] = a_m.data[index];
			}
		}
		//������ܴ����ڴ�й¶
		//cvReleaseImage(&a_plane); 
	} 

	cvReleaseImage(&temp); 
	return ret; 
}
Mat CFace::createROI(Mat m,string name){
		//�ٳ�Ŀ��
		Mat _gray;
		Mat mROI;
		//leftEyeROI.convertTo(leftEyeROI, leftEyeROI.type(), 1, 0); // ROI��ͼ��֮��ĸ���
		
		//�ȵ���ԭͼƽ������
		IplImage* _imageBgr = &IplImage(m);
		Mat _tar ;
		m.copyTo(_tar);
		IplImage* _imageTar = &IplImage(_tar);
		//set_avg_gray(_imageTar,_imageTar,(double)128.0);//����ƽ�⴦��
		imwrite("MedianData//"+name+"_128Light.jpg", _tar);
		
		cvSmooth(_imageTar,_imageTar);//ͼ��ƽ������

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
		// cout << "m.rows:" << m.rows << " - m.cols:" << m.cols <<  " . offSet: " << offSet << endl;
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
int CFace::calFirstRowOfContour(Mat countourSM){
	absROI(countourSM);
	for (int _row = 0; _row < countourSM.rows ; _row++)
	{
		uchar* rowData = countourSM.ptr<uchar>(_row);
		for (int _col = 0 ; _col < countourSM.cols ;_col++)
		{
			if (rowData[_col])
				return _row;
		}
	}
	return -1;
}
int CFace::calFirstRowOfContour_Col(Mat countourSM){
	absROI(countourSM);
	for (int _row = 0; _row < countourSM.rows ; _row++)
	{
		uchar* rowData = countourSM.ptr<uchar>(_row);
		for (int _col = 0 ; _col < countourSM.cols ;_col++)
		{
			if (rowData[_col])
				return _col;
		}
	}
	return -1;
}

int CFace::calLastRowOfContour(Mat countourSM){
	absROI(countourSM);
	for (int _row = countourSM.rows-1; _row >=0 ; _row--)
	{
		for (int _col = countourSM.cols-1 ; _col >=0 ;_col--)
		{
			int index = _row*countourSM.cols + _col;
			if(countourSM.at<uchar>(_row,_col))
				return _row;
		}
	}
	return -1;
}
int CFace::calLastRowOfContour_Col(Mat countourSM){
	absROI(countourSM);
	for (int _row = countourSM.rows-1; _row >=0 ; _row--)
	{
		for (int _col = countourSM.cols-1 ; _col >=0 ;_col--)
		{
			int index = _row*countourSM.cols + _col;
			if(countourSM.at<uchar>(_row,_col))
				return _col;
		}
	}
	return -1;
}

int CFace::calFirstColOfContour(Mat countourSM){
	absROI(countourSM);
	for (int _col = 0 ; _col < countourSM.cols ;_col++)
	{
		for (int _row = 0; _row < countourSM.rows ; _row++)
		{
			int index = _row*countourSM.cols+_col;
			//if (countourSM.data[index])
			if(countourSM.at<uchar>(_row,_col)){
				return _col;
			}
		}
	}
	return -1;
}

int CFace::calFirstColOfContour_Row(Mat countourSM){
	absROI(countourSM);
	for (int _col = 0 ; _col < countourSM.cols ;_col++)
	{
		for (int _row = 0; _row < countourSM.rows ; _row++)
		{
			int index = _row*countourSM.cols+_col;
			//if (countourSM.data[index])
			if(countourSM.at<uchar>(_row,_col)){
				return _row;
			}
		}
	}
	return -1;
}

int CFace::calLastColOfContour(Mat countourSM){
	absROI(countourSM);
	for (int _col = countourSM.cols -1 ; _col >=0 ;_col--)
	{
		for (int _row = countourSM.rows -1; _row >=0 ; _row--)
		{
			if(countourSM.at<uchar>(_row,_col)){
//				// cout <<  _row << "  -  " << _col << endl;
				return _col;
			}
		}
	}
	return -1;
}

int CFace::calLastColOfContour_Row(Mat countourSM){
	absROI(countourSM);
	for (int _col = countourSM.cols-1 ; _col >=0 ;_col--)
	{
		for (int _row = countourSM.rows-1; _row >=0 ; _row--)
		{
			if(countourSM.at<uchar>(_row,_col)){
//				// cout <<  _row << "  -  " << _col << endl;
				return _row;
			}
		}
	}
	return -1;
}
