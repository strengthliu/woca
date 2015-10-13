#include "cv.h"
#include "highgui.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

#define max_corners 10
#define max_face 10

#ifdef _EiC
#define WIN32
#endif
 
static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;
static CvHaarClassifierCascade* cascade2 = 0;
 
void detect_and_draw( IplImage* image );
 
const char* cascade_name =
    "haarcascade_frontalface_alt.xml";
const char* cascade_name2 =
    "haarcascade_eye_tree_eyeglasses.xml";
/*    "haarcascade_profileface.xml";*/
 
int main( int argc, char** argv )
{
    IplImage *pImg;

    if( argc == 2 && (pImg=cvLoadImage(argv[1],1))!= 0 )
    {
		cvNamedWindow("Src",1);
		cvShowImage("Src",pImg);
	}
    
    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
	cascade2 = (CvHaarClassifierCascade*)cvLoad( cascade_name2, 0, 0, 0 );

 
    if( !cascade )
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        fprintf( stderr,
        "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" );
        return -1;
    }
    storage = cvCreateMemStorage(0);
 
    cvNamedWindow( "result", 1 );
	detect_and_draw(pImg);
    cvWaitKey(0);
    cvReleaseImage( &pImg );
    cvDestroyWindow("result");
 
    return 0;
}
 
void detect_and_draw( IplImage* img )
{
	int cornerCount=max_corners;
	
    double scale = 1.3;
    IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );

    IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale),
                         cvRound (img->height/scale)),
                     8, 1 );
    int i,j;

    cvCvtColor( img, gray, CV_BGR2GRAY );
    cvResize( gray, small_img, CV_INTER_LINEAR );
    cvEqualizeHist( small_img, small_img );
    cvClearMemStorage( storage );
    
	CvRect eye_ROI[2*max_face];
	CvRect nose_ROI[max_face];
    CvRect mouth_ROI[max_face];
    CvPoint2D32f corners[max_corners];
    CvPoint face[2*max_face],eye[4*max_face],nose[2*max_face],mouth[2*max_face];
    int face_width[max_face],face_height[max_face];

    if( cascade )
    {
        double t = (double)cvGetTickCount();
        CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,
                                            1.1, 2, 0,cvSize(30, 30) );

		CvSeq* eyes = cvHaarDetectObjects( small_img, cascade2, storage,
                                            1.1, 2, 0,cvSize(30, 30) );

        t = (double)cvGetTickCount() - t;
        printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );

//人脸识别
        for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
            face[2*i].x=cvRound(r->x*scale);
			face[2*i].y=cvRound(r->y*scale);
			face[2*i+1].x=cvRound((r->x + r->width)*scale);
			face[2*i+1].y=cvRound((r->y + r->height)*scale);
            
 
			cvRectangle(img,face[2*i],face[2*i+1],CV_RGB(255,0,0),3,8,0);
        }

		int temp;            //face从大到小排序
		for(i=0;i<2*faces->total;i++)
		{
			for(j=0;j<2*faces->total-i;j++)
			{
				if(face[j].x<face[j+1].x)
				{
					temp=face[j].x;face[j].x=face[j+1].x;face[j+1].x=temp;

                    temp=face[j].y;face[j].y=face[j+1].y;face[j+1].y=temp;
				}
			}
		}
        for(i=0;i<faces->total;i++)
		{
			face_width[i]=face[2*i].x-face[2*i+1].x;
            face_height[i]=face[2*i].y-face[2*i+1].y;
		}



		

//眼睛检测与角点检测
        IplImage* corners1;
		IplImage* corners2;

		for( i = 0; i < (eyes ? eyes->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( eyes, i );
            

            eye[2*i].x=cvRound((r->x)*scale-face_width[cvRound(i/2-0.5)]/25);
			eye[2*i].y=cvRound((r->y)*scale+face_height[cvRound(i/2-0.5)]/15);
			eye[2*i+1].x=cvRound((r->x + r->width+face_width[cvRound(i/2-0.5)]/25)*scale);
			eye[2*i+1].y=cvRound((r->y + r->height)*scale-face_height[cvRound(i/2-0.5)]/25);
 
			cvRectangle(img,eye[2*i],eye[2*i+1],CV_RGB(255,0,0),3,8,0);
			
			eye_ROI[i].height=eye[2*i+1].y-eye[2*i].y;
			eye_ROI[i].width=eye[2*i+1].x-eye[2*i].x;
			eye_ROI[i].x=eye[2*i].x;
			eye_ROI[i].y=eye[2*i].y;
		}
		

		int ymin=0,ymax=0;
		for(i=0;i<eyes->total;i++)
		{
			cvSetImageROI(gray,eye_ROI[i]);
			cvSetImageROI(img,eye_ROI[i]);
			
			corners1= cvCreateImage(cvGetSize(gray), IPL_DEPTH_32F, 1);
			corners2= cvCreateImage(cvGetSize(gray),IPL_DEPTH_32F, 1);
			CvSize size;
            size=cvGetSize(gray);
			ymin=size.height/4;
			ymax=size.height*2/3;
			
			cvGoodFeaturesToTrack (gray, corners1, corners2, corners,
                                     &cornerCount, 0.15, 5, 0);
           
			CvPoint point1,point2;	
			point1.x=200;
			point1.y=200;
			point2.x=0;
			point2.y=0;
			for (int j=0; j<cornerCount; j++)
			{
				
				if(corners[j].y<ymax&&corners[j].y>ymin)
				{
					if(point1.x>corners[j].x) 
					{
						point1.x=corners[j].x;
						point1.y=corners[j].y;
					}
					if(point2.x<corners[j].x)
					{
						point2.x=corners[j].x;
						point2.y=corners[j].y;
					}
				}
			}
			cvCircle(img, cvPoint((int)(point1.x), (int)(point1.y)), 4,
               CV_RGB(255,0,0), 2, CV_AA, 0);
			cvCircle(img, cvPoint((int)(point2.x), (int)(point2.y)), 4,
               CV_RGB(255,0,0), 2, CV_AA, 0);

			img->roi=NULL;
			gray->roi=NULL;
		}         

		                          //eyes从大到小排序
		for(i=0;i<4*faces->total;i++)
		{
			for(j=0;j<4*faces->total-i;j++)
			{
				if(eye[j].x<eye[j+1].x)
				{
					temp=eye[j].x;eye[j].x=eye[j+1].x;eye[j+1].x=temp;

                    temp=eye[j].y;eye[j].y=eye[j+1].y;eye[j+1].y=temp;
				}
			}
		}
				
		
//鼻子检测与角点检测
		
		for(i=0;i<faces->total;i++)
		{
			nose[2*i].x=(eye[4*i].x+eye[4*i+1].x)/2-face_width[i]/18;
			nose[2*i].y=((eye[4*i].y>eye[4*i+2].y)?eye[4*i].y:eye[4*i+2].y);
			nose[2*i+1].x=(eye[4*i+2].x+eye[4*i+3].x)/2+face_width[i]/18;
            nose[2*i+1].y=nose[2*i].y+face_height[i]/3.56;
            
			nose_ROI[i].height=nose[2*i+1].y-nose[2*i].y;
			nose_ROI[i].width=nose[2*i].x-nose[2*i+1].x;
			nose_ROI[i].x=nose[2*i+1].x;
			nose_ROI[i].y=nose[2*i].y;
                       
			cvRectangle(img,nose[2*i],nose[2*i+1],CV_RGB(255,255,255),2,CV_AA,0);
		}

        CvPoint point3,point4;	
		for(i=0;i<faces->total;i++)
		{
			cvSetImageROI(gray,nose_ROI[i]);
			cvSetImageROI(img,nose_ROI[i]);
			
            CvSize size;
			size=cvGetSize(gray);
			corners1= cvCreateImage(cvGetSize(gray), IPL_DEPTH_32F, 1);
			corners2= cvCreateImage(cvGetSize(gray),IPL_DEPTH_32F, 1);

			
			cvGoodFeaturesToTrack (gray, corners1, corners2, corners,
                                     &cornerCount, 0.2, 5, 0);
           

			point3.x=size.width/2;
			point3.y=0;
			point4.x=size.width/2;
			point4.y=0;
			for (int j=0; j<cornerCount; j++)
			{
				if(point3.y<corners[j].y&&point3.x>corners[j].x) 
				{
					point3.x=corners[j].x;
					point3.y=corners[j].y;
				}
				if(point4.y<corners[j].x&&point4.x<corners[j].x)
				{
					point4.x=corners[j].x;
					point4.y=corners[j].y;
				}
			}
			cvCircle(img, cvPoint((int)(point3.x), (int)(point3.y)), 6,
               CV_RGB(255,0,0), 2, CV_AA, 0);
		    cvCircle(img, cvPoint((int)(point4.x), (int)(point4.y)), 6,
               CV_RGB(255,0,0), 2, CV_AA, 0);

			img->roi=NULL;
		} 
        


//嘴检测
      
        for(i=0;i<faces->total;i++)
		{
			mouth[2*i].x=(eye[4*i].x+eye[4*i+1].x)/2;
			mouth[2*i].y=nose[2*i+1].y;
			mouth[2*i+1].x=(eye[4*i+2].x+eye[4*i+3].x)/2;
			mouth[2*i+1].y=nose[2*i+1].y+nose_ROI[i].height/1.9;
            
			mouth_ROI[i].height=mouth[2*i+1].y-mouth[2*i].y;
			mouth_ROI[i].width=mouth[2*i].x-mouth[2*i+1].x;
			mouth_ROI[i].x=mouth[2*i+1].x;
		    mouth_ROI[i].y=mouth[2*i].y;

            cvRectangle(img,mouth[2*i],mouth[2*i+1],CV_RGB(255,255,255),2,CV_AA,0);
		}
        
        for(i=0;i<faces->total;i++)
		{
			cvSetImageROI(gray,mouth_ROI[i]);
			cvSetImageROI(img,mouth_ROI[i]);
			
            CvSize size;
			size=cvGetSize(gray);
			corners1= cvCreateImage(cvGetSize(gray), IPL_DEPTH_32F, 1);
			corners2= cvCreateImage(cvGetSize(gray),IPL_DEPTH_32F, 1);

			
			cvGoodFeaturesToTrack (gray, corners1, corners2, corners,
                                     &cornerCount, 0.1, 5, 0);
           
			CvPoint point5,point6;	
			point5.x=0;
			point5.y=0;
			point6.x=100;
			point6.y=0;
			for (int j=0; j<cornerCount; j++)
			{

				if(point5.x<corners[j].x) 
				{
					point5.x=corners[j].x;
					point5.y=corners[j].y;
				}
				if(point6.x>corners[j].x)
				{
					point6.x=corners[j].x;
					point6.y=corners[j].y;
				}
			}

			cvCircle(img, cvPoint((int)(point5.x), (int)(point5.y)), 4,
               CV_RGB(255,0,0), 2, CV_AA, 0);
		    cvCircle(img, cvPoint((int)(point6.x), (int)(point6.y)), 4,
               CV_RGB(255,0,0), 2, CV_AA, 0);

			img->roi=NULL;
		} 

	}
        	    
    cvShowImage( "result", img);
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );
}
