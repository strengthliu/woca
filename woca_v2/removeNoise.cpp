// 图像去噪.cpp : 定义控制台应用程序的入口点。

//#include "stdafx.h"
/*中值滤波、均值滤波与加权(高斯)滤波*/
#include<iostream>
using namespace std; 
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>

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

void removeNoise(IplImage* pImg)//----------------------------------------------------------------------------主函数-----------------------------------
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
	PrintStar();
	cvNamedWindow( "Original Image", CV_WINDOW_AUTOSIZE );//创建窗口
	cvMoveWindow( "Original Image", 500, 200 );//设置图像输出的窗口位置
	cvShowImage( "Original Image", pImg );//显示原始图像

	cvNamedWindow( "Median Filter Image", CV_WINDOW_AUTOSIZE );//创建窗口
	cvMoveWindow( "Median Filter Image", 900, 200 );//设置图像输出的窗口位置
	cvShowImage( "Median Filter Image", Img_Median );//显示中值滤波后图像

	cvNamedWindow( "Average Filter Image", CV_WINDOW_AUTOSIZE );//创建窗口
	cvMoveWindow( "Average Filter Image", 500, 500 );//设置图像输出的窗口位置
	cvShowImage( "Average Filter Image", Img_Average );//显示均值滤波后图像

	cvNamedWindow( "Weighted Filter Image", CV_WINDOW_AUTOSIZE );//创建窗口
	cvMoveWindow( "Weighted Filter Image", 900, 500 );//设置图像输出的窗口位置
	cvShowImage( "Weighted Filter Image", Img_Weighted );//显示加权(高斯)滤波后图像

	cvWaitKey(0); //等待按键

	cvDestroyWindow( "Original Image" );//销毁原始图像窗口
	cvReleaseImage( &pImg ); //释放图像

	cvDestroyWindow( "Median Filtering Image" );//销毁中值滤波窗口
	cvReleaseImage( &Img_Median ); //释放图像

	cvDestroyWindow( "Average Filtering Image" );//销毁均值滤波窗口
	cvReleaseImage( &Img_Average ); //释放图像

	cvDestroyWindow( "Weighted Filtering Image" );//销毁加权(高斯)滤波窗口
	cvReleaseImage( &Img_Weighted ); //释放图像
}
//--------------------------------------------------------------------------------------------------------------------------------
