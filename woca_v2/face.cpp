#include "face.h"
#include <iostream>
#include <stdio.h>  
#define M_PI       3.14159265358979323846

using namespace std;
CFace::CFace(){
}
CFace::~CFace(){
}


//计算一点相对于一条直线左点的平行投影距离
double CFace::calParallelDistance(Point el,Point er,Point et){
	//et在线上距离el的距离，et1在线上距离el的距离
	double etel_v = calVerticalDistance(el,er,et);
	double v_etel = (et.x-el.x)*(et.x-el.x) + (et.y-el.y)*(et.y-el.y);
	double etel = sqrt(v_etel);
	double etelp = sqrt(etel*etel - etel_v*etel_v);
	return etelp;
}
//计算两点相对于一条直线的平行距离
double CFace::calParallelDistance(Point el,Point er,Point et,Point et1){
	double etelp = calParallelDistance(el,er,et);
	double et1elp = calParallelDistance(el,er,et1);
	return etelp - et1elp;
}

//计算指定点到直线的高度。如果指定点在直线方向左侧，则距离为正，否则为负。
double CFace::calVerticalDistance(Point el,Point er,Point et){
	double l_e,l_ea,l_eb,l_ec;
	double l_eat = (el.x-er.x)*(el.x-er.x)+(el.y-er.y)*(el.y-er.y);
	l_ea=sqrt(l_eat);//计算模板对角线长度
	double l_ebt = (el.x-et.x)*(el.x-et.x)+(el.y-et.y)*(el.y-et.y);
	l_eb=sqrt(l_ebt);
	double l_ect = (er.x-et.x)*(er.x-et.x)+(er.y-et.y)*(er.y-et.y);
	l_ec=sqrt(l_ect);
	//CosC=(a^2+b^2-c^2)/2ab ;b*CosC=a1;((a^2+b^2-c^2)/2a)=a1;l_e=squr(b^2-((a^2+b^2-c^2)/2a)^2)
	double l_et = l_eb*l_eb-((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea))*((l_ea*l_ea+l_eb*l_eb-l_ec*l_ec)/(2*l_ea));
	l_e=sqrt(l_et);//高点到对角线的高度

	//先将et平行移到er一侧
	//以el为原点，计算原点与两个点的角度，就是在左侧，否则在右侧。
	//如果el为原点，er为X轴正方向，计算et所在区域，就知道是左右了。
	Point ert = Point(er.x-el.x,er.y-el.y);
	Point err = Point(et.x-el.x,et.y-el.y);
	int _2to1=1,_4to1=1,t2to1=1,t4to1=1;
	if(err.x<0) { err = Point(-1*err.x,err.y);ert = Point(-1*ert.x,ert.y);_2to1=-1;}//12象限转
	if(err.y<0) { err = Point(err.x,-1*err.y);ert = Point(ert.x,-1*ert.y);_4to1=-1;}
	if(ert.x<0) { ert = Point(-1*ert.x,ert.y);t2to1=-1;}
	if(ert.y<0) { ert = Point(ert.x,-1*ert.y);t4to1=-1;}

return 0;
}


//将图像按指定方向缩放。sp,源点，tp方向，
void CFace::resizeMat(Mat src,Mat tar,Point sp,Point tp,double ratio){
	//沿指定方向，做一个矩形，框住目标区域
	double rWidth,rHeight;//矩形宽高
	//计算四个点与这条线的垂直距离
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
		if(max=sortT[i] && sortP[i].x>=0 && sortP[i].y>=0) { // 取出最大值，并清除数组
			maxPoint=Point(sortP[i].x,sortP[i].y);
			sortP[i]=Point(-1,-1);
			break;
		}
	}
	min = sortT[0];
	for(int i=0;i<4;i++){	if(min > sortT[i]) min = sortT[i];	}
	for(int i=0;i<4;i++){ 
		if(min=sortT[i] && sortP[i].x>=0 && sortP[i].y>=0) { // 取出最小值，并清除数组
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

	//两个对角顶点到射线的垂直距离，Y距离。

	//两个同向顶点沿射线的平行距离，X距离

	//申请一个矩形

	//将所有图像点映射到这个平行矩形里

	//横向拉申

	//申请一个框住转回原角度的矩形

	//再将每个点的影射回去

}

//清除图像eyeMask中，指定矩形r所框住的内容，以及这些内容的联通区域。
void CFace::removeBrow(Mat eyeMask,Rect r){ // 设定眼睛ROI的上1/6,一定会包括眉毛或不包括眼睛。
	// cout << " 执行 removeBrow " << r.x << " " << r.y << " " << r.width << " " << r.height << endl;
	absROI(eyeMask);
	// 清除上1/6区域中的一切联通块
	int b=0,f=255;
	Mat tar;eyeMask.copyTo(tar); // 复制一个TAR，做为临时变量，用于清空并放提取出来的东西。

	Mat rt = eyeMask(r);//Mat rtt;rt.copyTo(rtt);rtt.setTo(Scalar(b));
	if(debug){
		imwrite("MedianData//BfremovedMASK.png",rt);
		imwrite("MedianData//BfremovedSource.png",tar);
	}

	for(int _rowrt=0;_rowrt<rt.rows;_rowrt++){
		for(int _colrt=0;_colrt<rt.cols;_colrt++){
			int _row = _rowrt + r.y;
			int _col = _colrt + r.x;
			int index = _row*eyeMask.cols+_col; // 在eyeMask中定位矩形内的点
			//int index = _col*eyeMask.cols+_row; // 在eyeMask中定位矩形内的点
			if(eyeMask.at<uchar>(_row,_col)){ //如果eyeMask里这个点有值。这里有错误，nineBox已经清除过，并且不执行了。但这里还会判定点有值，而且真有值。
			//if((uchar)eyeMask.data[index] > 0){ //如果eyeMask里这个点有值。这里有错误，nineBox已经清除过，并且不执行了。但这里还会判定点有值，而且真有值。
				//if(debug) cout << " removeBrow: found white point:(" << _col << "," << _row << ") " << "(" << _colrt << "," << _rowrt << ")" << endl;
				tar.setTo(Scalar(b));	// 清空临时变量TAR。
				nineBox(eyeMask,tar,_col,_row,true); // 把eyeMask里，这个点连接的块拿出来，放到TAR里。
				imwrite("MedianData//removed1.png",tar);
				imwrite("MedianData//removed2.png",eyeMask);
			}
		}
	}
	//imwrite("MedianData//eyeMask.png",eyeMask);
	//imwrite("MedianData//removed.png",eyeMask);

}
Mat CFace::createROI(Mat m1,string name,int pre,int mode,int range){ // m源图。pre，预处理：0，无；1，直方图。mode：1，cvThreshold；2，cvAdaptiveThreshold；3，Canny边缘检测。
	Mat m;
	if(m1.channels()>1){
		cvtColor( m1, m, CV_BGR2GRAY );
	} else
		m = m1;
	imwrite("MedianData//roi12.png",m);

		//抠出目标
		Mat _gray;
		Mat mROI;
		//先调整原图平均亮度
		IplImage* _imageBgr = &IplImage(m);
		Mat _tar ;
		m.copyTo(_tar);
		//imwrite("MedianData//"+name+"_tar.png", _tar);
		IplImage* _imageTar = &IplImage(_tar);
		set_avg_gray(_imageTar,_imageTar,(double)128.0);//亮度平衡处理
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
			//眼睛直方图后，效果不好。因为眼睛的黑白已经很明显了。鼻子和嘴就需要.
			equalizeHist( _gray, _gray );
			imwrite("MedianData//"+name+"le2.jpg", mt2);
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
			while(1){
				if(range % 2 == 1 && range > 1){break;}
				else range++;
				if(range>100) return m;
			}

			if(mode == 2) // 中值算法
				cvAdaptiveThreshold(graySource, grayDst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, range, range);

			if(mode == 3) {// 高斯算法
				//cvAdaptiveThreshold(graySource, grayDst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, range, range);
				cvAdaptiveThreshold(graySource, grayDst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, range, range);
			}
			
			reverseROI(grayDst); // 转成黑底白字
			imwrite("MedianData//roi13.png", Mat(grayDst));
			filterBlock(grayDst,5,5,255); // 过滤大小需要重新设计
		}

		Mat mt = Mat(grayDst);
		absROI(mt);
		imwrite("MedianData//"+name+".jpg", mt);
		return mt;
}

void CFace::filterBlock(Mat frame,int w,int h,int v){ // 过滤掉小于(w,h)的独立块。stack overflow!!
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
				//// cout << " col: " << _col << " " << _row << endl;
				tar.setTo(Scalar(b)); // 设成背景色。
				nineBox(src,tar,_col,_row,true);//生长法：nineBox
				//Mat reverse; tar.copyTo(reverse);reverse.setTo(Scalar(b));
				imwrite("MedianData//filterBlockReverse.png",tar);
				Rect r = getROIRect(tar);
				//reverseROI(tar);
				//旋转SRC，使它最贴合宽高最大比例
				int width = r.width;//this->calLastColOfContour(reverse) - this->calFirstColOfContour(reverse);
				int height = r.height;//this->calLastRowOfContour(reverse) - this->calFirstRowOfContour(reverse);
				//// cout <<"width:"<< width << " height:" << height << " x:" << r.x << " y:" << r.y << " w:" << w << " h:" << h << endl;
				if(width >= w && height >= h) {
					copyROI(tar,frame); // 如果连接的一个块大于指定宽高，就复制回去。
					imwrite("MedianData//frameBlock.png",frame);
				}
			}
		}
	}
	//cvReleaseMat(tar);
	//reverseROI(frame);
	return;
}

//使用smooth模式。按P1P2为中心
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

//使用smooth模式。按P1P2为中心
void CFace::rSmoothRect(Mat img,Rect r,Point p1,Point P2,int deep){
	//if(
}



//根据MASK，做联通过滤。如果frame中的一个点在MASK内，那么它在MASK之外的联通域也保留；(在MASK内的，做联通处理)。否则删除。
void CFace::filterBlock(Mat frame,Mat mask,bool blackBackground){
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
					// cout << " remove " << endl;
				}
			}
		}
	}
	if(!blackBackground){reverseROI(ret);reverseROI(mask);	}
	ret.copyTo(frame);
	//frame = ret;
}

// 判断t中的点是否有在mask范围内的。
bool CFace::joinMask(Mat t,Mat mask,bool blackBackground){ 
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
				//// cout << _tt.at<uchar>(_row,_col) << " , " << _row << " , " << _col << " , " << _maskt.at<uchar>(_row,_col) << endl;
				if( _maskt.at<uchar>(_row,_col)){//.data[index] < 50){
					//// cout << _tt.at<uchar>(_row,_col) << " . " << _row << " . " << _col << " . " << _maskt.at<uchar>(_row,_col) << endl;
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

int CFace::getROIWidth(Mat _roi,int mode){ // mode:0，是黑底，1是白底
	return 0;
}
int CFace::getROIHeight(Mat _roi,int mode){
	return 0;
}

// 把ROI：m中，与XY点连接的块，从m中拿出来，放到t中。执行后，m中，就没有这个块的值了。
Mat CFace::nineBox(Mat m,Mat t,int x,int y,bool blackBackground){
	int b=0,f=255; // 设定前后景的色值

	int left,right,top,buttom;//与该点相连接的9宫格的上下左右点坐标
	if(x-1>=0) left = x-1;else left = x;
	if(x+1<m.cols) right = x+1;else right = x;
	if(y-1>=0) top = y-1;else top = y;
	if(y+1<m.rows) buttom = y+1;else buttom = y;
	Rect r=Rect(left,top,right-left+1,buttom-top+1);// 9宫格矩形框，考虑了边缘问题。
	//复制出一个9宫格块到临时变量_submat中，用于记录原9格里哪个点有值。这是这个方法里，唯一一个创建新Mat的地方。
	Mat submat = m(r);Mat _submat;submat.copyTo(_submat);
	if(!t.cols){	m.copyTo(t);	t.setTo(Scalar(b));	} //如果目标为空，就初始化一个。
	Mat submatt = t(r);  //找出目标中相同位置的9宫格块，取引用头。
	//cvAdd(submatt,_submat,submatt);//写目标
	if(submatt.cols<1){
		cout << " " << endl;
	}
	if(submat.cols<1){
		cout << " " << endl;
	}
	submatt = submatt + submat;//写目标。这里有问题的话，目标会错。
	submat.setTo(Scalar(b));//清空源。现在怀疑这里有问题，因源里面的相应点，没有被清空，还在循环被发现。
	//cvSetZero(submat);//清空源。上面的好象没执行，这里再来一次。

	for(int _r=0;_r<r.height;_r++){//判断递归
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

//取SRC中指定点（startX,startY）的联通MASK，并将其移出至TAR中。要求，起始时，指定点有前景值；图像为白底黑字。
Mat CFace::nineBox(Mat frameSrc,Mat frameTar,int startX,int startY){
	return nineBox(frameSrc,frameTar,startX,startY,true);
}

Rect CFace::getROIRect(Mat src){
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

Rect CFace::copyROI(Mat src,Mat tar){//将src里大于0的点，复制到tar里.
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
	//顺序可能错了
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

Vec3b CFace::kcvHSL2BGR(Vec3b img){

    double R,G,B;
	double H,S,L;
	H = img[0];
	S = img[1];
	L = img[2];

    double var_1, var_2;
    if (S == 0)                       //HSL values = 0 ÷ 1
    {
        R = L * 255.0;                   //BGR results = 0 ÷ 255
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
				//转换成亮度
				Vec3d p = kcvBGR2HSL(colDataBGR[j]);
				//累加
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
				//转换成亮度
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
			cout << " 什么情况？ " << endl;
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
	if(img1->nChannels==3){//认为是BGR数据
		kcvBGR2HSV(Mat(img1));//转成HSV
		split(Mat(img1),h);
		img=&IplImage(h[2]);
	}
	if(img1->nChannels==1)//认为是灰度数据
		img = img1;

	prev_avg_gray = get_avg_gray(img);
	if(prev_avg_gray>0 && avg_gray > 0){
		// cout << (double)(avg_gray/prev_avg_gray) << endl;
		cvConvertScale(img,out,(double)(prev_avg_gray/avg_gray));
	}
	else
		cout << "Div by zero!" << endl;

	if(img1->nChannels==3){//认为是BGR数据
		merge(h,mout);
		kcvHSV2BGR(mout);//转成BGR
		cvReleaseImage(&out);
		out1 = &IplImage(mout);
	}
	if(img1->nChannels==1) {//认为是灰度数据
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
	//滤镜加在这
	//this->imageContourSM;

	//------------------------
}

//调亮图像。
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

	//亮度平衡处理
	IplImage* imageModel = &IplImage(model);
	IplImage* imageSrc = &IplImage(src);
	src.copyTo(ret);
	IplImage* imageRet = &IplImage(ret);
	double gFace = get_avg_gray(imageModel);
	double gBgr = get_avg_gray(imageSrc);
	//为了调试捏脸，先注释掉。
	//set_avg_gray(imageRet,imageRet,gFace*0.7);
	//set_avg_gray(imageRet,imageRet,128);
	
	//测试是否是透明部分影响一亮度平均值
	// cout << "gFace:" << gFace << " gBgr:" << gBgr << endl;

	return ret;
}

void skinBalance(Mat body,Mat face){
	/*
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
    cvReleaseImage(&imageSkin);     
	*/

}
//现在旋转后，通过放大画布，保持了原图片大小，但是旋转过后，失去了原图片的大小，转回去时，图片变了。也可能考虑，通过计算，直接旋转缩放到目标尺寸。
Mat CFace::rotate(Mat srcm,double angle){
	//	cout << " 旋转模型。。。" << angle << "度 " <<  endl;

	//如果源是4通道的图像，转成三通道先，等旋转过后，再转换回4通道。
	Mat src_temp;
	IplImage* srct =  &IplImage(srcm);//将MAT转为IplImage
	int chanelNum = srct->nChannels;
	IplImage* a_plane;
	//	cout << " 旋转模型。。。1 " <<  endl;

	//if(srcm.cols)  cout << " 输入图不为空 " << endl;  else cout << " 输入源图为空 " << endl;

	if( chanelNum == 4){
		 a_plane = cvCreateImage(cvGetSize(srct),8,1);
		 cvCvtPixToPlane(srct,0,0,0,a_plane);      //提取透明通道
		cvtColor(srcm, src_temp, CV_BGRA2BGR);
		//cvtColor(src , src_temp , CV_BGRA2BGR);
	} else {
		src_temp = srcm;
	}
	//	cout << " 旋转模型。。。2 " <<  endl;

	IplImage* src =  &IplImage(src_temp);//将MAT转为IplImage
	//将角度值转换为第1象限正值
	int angle1 = angle;
	angle = abs((int)angle % 180); 
	//// cout << " ang:" << angle << " " << angle1 << endl;
	if (angle > 90) { 
		angle = 90 - ((int)angle % 90); 
	} 
	if(angle<0) angle = 360+angle;

	//计算旋转后宽高缩放值
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

	//将正负旋转值付回去。这里只处理+-90度的
	if(angle1!=0) angle = (abs(angle1)/angle1)*angle;


	//创建一个新画布，并清空为0。再创建一个对角线那么宽的正方形，并清空为0.
	dst = cvCreateImage(cvSize(width, height), src->depth, src->nChannels); 
	cvZero(dst); 
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels); 
	cvZero(temp); 
	
	//设置对角线画布中，正中间原图像那么大区域为ROI区。将原图复制到这个区域内。
	cvSetImageROI(temp, cvRect(tempX, tempY, src->width, src->height)); 
	cvCopy(src, temp, NULL); 
	cvResetImageROI(temp); 

	//制作旋转参数
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
	//// cout << "rotateMtr:" << m[0] << " " << m[1] << " " << m[3] << " " << m[4] << " " << m[2] << " " << m[5] << " | angle=" << angle << endl;
	// 
	CvMat M = cvMat(2, 3, CV_32F, m); 
	//执行图像旋转。将对角线大图旋转，把中间DST那么大块复制到DST中。
	cvGetQuadrangleSubPix(temp, dst, &M); 
	cvReleaseImage(&temp); 

	//cout << " 开始恢复四通道数值 " <<  endl;
	Mat ret = Mat(dst);

	if( chanelNum == 4){
		Mat a_m = Mat(a_plane);
		rotate(a_m,angle);
		cvtColor(ret, ret, CV_BGR2BGRA);

		for(int _row=0;_row<ret.rows;_row++){
			for(int _col=0;_col<ret.cols;_col++){
				int index = _row*ret.cols+_col;
				Vec4b _data = ret.data[index];
				//调试用。先将透明度去掉，我怀疑透明度全为0.
				_data[3] = a_m.data[index];
			}
		}
		//这里可能存在内存泄露
		//cvReleaseImage(&a_plane); 
	} 

	cvReleaseImage(&temp); 
	return ret; 
}
Mat CFace::createROI(Mat m,string name){
		//抠出目标
		Mat _gray;
		Mat mROI;
		//leftEyeROI.convertTo(leftEyeROI, leftEyeROI.type(), 1, 0); // ROI子图像之间的复制
		
		//先调整原图平均亮度
		IplImage* _imageBgr = &IplImage(m);
		Mat _tar ;
		m.copyTo(_tar);
		IplImage* _imageTar = &IplImage(_tar);
		//set_avg_gray(_imageTar,_imageTar,(double)128.0);//亮度平衡处理
		imwrite("MedianData//"+name+"_128Light.jpg", _tar);
		
		cvSmooth(_imageTar,_imageTar);//图像平滑处理

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
		// cout << "m.rows:" << m.rows << " - m.cols:" << m.cols <<  " . offSet: " << offSet << endl;
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
