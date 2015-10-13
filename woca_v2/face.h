#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include <opencv2/opencv.hpp>


using namespace cv;

class CFace
{
public:
	CFace();
	virtual~ CFace();

	static const bool debug = true;

	//***************************************************************

	static Mat createROI(Mat m,string name,int pre,int mode,int range);
	static void filterBlock(Mat frame,int w,int h,int v);
	static void filterBlock(Mat frame,Mat mask,bool blackBackground);
	static Mat nineBox(Mat frameSrc,Mat frameTar,int startX,int startY);
	static Mat nineBox(Mat frameSrc,Mat frameTar,int startX,int startY,bool blackBackground);
	static void debugNineBox(Mat m,int x,int y);
	static bool joinMask(Mat t,Mat mask,bool blackBackground);
	static Rect copyROI(Mat src,Mat tar);
	static Rect getROIRect(Mat src);
	static void reverseROI(Mat roi);
	static void absROI(Mat roi);
	static int getROIWidth(Mat _roi,int mode);
	static int getROIHeight(Mat _roi,int mode);
	static void removeBrow(Mat eyeMask,Rect r);
	static void resizeMat(Mat src,Mat tar,Point sp,Point tp,double ratio);
	static double calVerticalDistance(Point s,Point t,Point c);
	static double calParallelDistance(Point el,Point er,Point et,Point et1);
	static double calParallelDistance(Point el,Point er,Point et);

	//取一个N通道图像中，指定ROI区域中的点的各通道平均值
	static vector<int> getAverageRoute(Mat img);
	//设置一个N通道图像，便之指定ROI区域内的各通道平均值为指定值
	static void setAverageRoute(Mat img,vector<int> vs);

	static Mat changeChanel(Mat img,int _3To4);

	static Mat createROI(Mat m,string name);
	static Mat rotate(Mat src,double angle);
	static void filterMirrorBackground(Mat &resultImage);
	static void brightnessContrast(Mat &resultImage, double alpha, double beta);
	static int calFirstRowOfContour(Mat countourSM);
	static int calFirstRowOfContour_Col(Mat countourSM);
	static int calLastRowOfContour(Mat countourSM);
	static int calLastRowOfContour_Col(Mat countourSM);
	static int calFirstColOfContour(Mat countourSM);
	static int calFirstColOfContour_Row(Mat countourSM);
	static int calLastColOfContour(Mat countourSM);
	static int calLastColOfContour_Row(Mat countourSM);

	static Vec3b kcvBGR2HSV(Vec3b img) ;
	static Vec3b kcvHSV2BGR(Vec3b img)  ;
	static Vec3b kcvBGR2HSL(Vec3b img) ;
	static Vec3b kcvHSL2BGR(Vec3b img)  ;
	static double Hue2BGR(double v1, double v2, double vH);

	static int averageLight(Mat faceSampleBGR);
	static void adjustLight(Mat faceSampleBGR,float _v);

	static Mat lightBalanceFrame(Mat src,Mat model,Mat ret);
	static void skinBalance(Mat body,Mat face);
	static double get_avg_gray(IplImage *img);
	static void set_avg_gray(IplImage *img,IplImage *out,double avg_gray);

	static void cvSkinHSV(IplImage* src,IplImage* dst);
	static void cvSkinSegment(IplImage* img, IplImage* mask);
	static void cvSkinYUV(IplImage* src,IplImage* dst) ;

	//指定区域，中心向四边虚化。保持中心清晰.deep是清除原色的程序
	static void smoothRect(Mat &img,Rect r,Point p1,Point P2,int deep);
	//指定区域，四周向中心虚化。保持四周清晰
	static void rSmoothRect(Mat img,Rect r,Point p1,Point P2,int deep);

	//***************************************************************

	static Point* getEyePoint(Mat leftEyeROI_l,Mat leftEyeROI_r,Mat rightEyeROI_r,Mat rightEyeROI_l,Mat leftEyeROI_,Mat rightEyeROI_);
	static Point* getBrowPoint(Mat bmask1,Mat bmask11); 
	static Point getEyeModelPoint(Mat model,int lr);
	//Mat resizeModel(Mat modelImg,Point ml,Point mr,Point mt,Point el,Point er,Point et);
	static Mat resizeModel(Mat modelImg,Point ml,Point mr,Point mt,Point mb,Point el,Point er,Point et,Point eb,int mode=0);
	static Point* getNosePoint(Mat noseROI);
	static Point* getMouthPoint(Mat mouthROI);


};
