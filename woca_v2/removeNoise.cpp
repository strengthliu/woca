// ͼ��ȥ��.cpp : �������̨Ӧ�ó������ڵ㡣

//#include "stdafx.h"
/*��ֵ�˲�����ֵ�˲����Ȩ(��˹)�˲�*/
#include<iostream>
using namespace std; 
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>

//#define picture_name "004.jpg"//ͼƬ���� ���޸�ͼƬ���ƣ�����ֻ���޸���һ�����
#define template_size 3 //ģ���С ���޸�ģ���С������ֻ���޸���һ�����
int m,n,height,width;//����ͼ���С
int p,q;//����ģ���С��Ϊ��̺��޸ķ��㣬��ֱ����template_size
int result_median,result_average,result_weighted,logo;//����ÿ������Ľ�� �㷨��־λ
int a[template_size*template_size],b[template_size*template_size];//a��������ð������ b�������ڴ�ģ��ϵ��
int c[template_size][template_size];//c[][]���ڴ洢��Ȩ(��˹)ģ��ϵ��
//--------------------------------------------------------------------------------------------------------------------------------
void PrintStar()//-----------------------------------------------------------------------��ӡ�ָ���-------------------------------
{
	for(int i=0;i<80;i++)
		printf("*");
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Weighted()//------------------------------------------------------------------����ģ��ϵ�������ֵ���浽����median��--
{
	//�������ȸ���Ȩ(��˹)ģ��ϵ�� 
	int t,i,j,*y;
	for (t=1;t<=(p+1)/2;t++)//Ϊģ�������м��е�q��Ԫ�ظ�ֵ
	{
		c[(p-1)/2][(q-1)/2-(t-1)]=pow(2.0,(p-t));
		c[(p-1)/2][(q-1)/2+(t-1)]=pow(2.0,(p-t));
	}
	for (t=1;t<=(p-1)/2;t++)//Ϊģ����������ÿ�е�q��Ԫ�ظ�ֵ
	{
		for (j=0;j<p;j++)
		{
			c[(p-1)/2-((p+1)/2-t)][j]=c[(p-1)/2][j]/pow(2.0,((p+1)/2-t));
			c[(p-1)/2+((p+1)/2-t)][j]=c[(p-1)/2][j]/pow(2.0,((p+1)/2-t));
		}
	}
	y=&b[0];                        //ָ�����ָ��һά�����׵�ַ���Դ���һ�δ���
	for (i=0;i<p;i++)
	{
		for (j=0;j<q;j++)
		{
			*y=c[i][j];             //�Ѹ�˹ģ��ϵ���浽һά����b[]
			y++;
		}
	}
	//�������ȸ���Ȩ(��˹)ģ��ϵ��
	int sum=0,sum_template=0;
	for (i=0;i<p*q;i++)
	{
		sum+=a[i]*b[i];              //��Ȩ���
		sum_template+=b[i];          //��ģ��ϵ����
	}
	result_weighted=sum/sum_template;//�ѽ���ݴ浽result_weighted�У���׼��ͨ��Img_Weightedָ����ͼ�������и�ֵ
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Average()//------------------------------------------------------------------����ģ��ϵ�������ֵ���浽����median��---
{
	//�������ȸ���ֵģ��ϵ��
	for (int k=0;k<p*q;k++)
	{
		b[k]=1;//��ֵ�˲�ģ��ϵ����Ϊ1                          
	}
	//�������ȸ���ֵģ��ϵ��
	int i,sum=0,sum_template=0;
	for (i=0;i<p*q;i++)
	{
		sum+=a[i]*b[i];              //��Ȩ���
		sum_template+=b[i];          //��ģ��ϵ����
	}
	result_average=sum/sum_template;//�ѽ���ݴ浽result_average�У���׼��ͨ��Img_Averageָ����ͼ�������и�ֵ
}
//--------------------------------------------------------------------------------------------------------------------------------
void Select_Median()//-------------------------------------------------------------------ȡ����ֵ���浽����median��---------------
{
	if ((p*q)%2==1)                         //ģ�峤��˻�Ϊ����
	{
		result_median=a[(p*q-1)/2];//�ѽ���ݴ浽result_median�У���׼��ͨ��Img_Medianָ����ͼ�������и�ֵ
	} 
	else                                    //ģ�峤��˻�Ϊż��  ����else����ȥ���ˣ���Ϊ�Ѿ��涨ģ��Ϊ3*3 �˻�������ż��
	{
		result_median=(a[p*q/2]+a[p*q/2-1])/2;
	}
}
//--------------------------------------------------------------------------------------------------------------------------------
void Sort()//----------------------------------------------------------------------------����-------------------------------------
{
	int i,j,k;
	for(k=0;k<p*q-1;k++)                    //ð������
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
	//cout<<"�����"<<a[0]<<" "<<a[1]<<" "<<a[2]<<" "<<a[3]<<" "<<a[4]<<" "<<a[5]<<" "<<a[6]<<" "<<a[7]<<" "<<a[8]<<" "<<endl;
	Select_Median();
}
//--------------------------------------------------------------------------------------------------------------------------------
void Judge_Method()//--------------------------------------------------------------------�ж������ַ����˲�-----------------------
{
	switch(logo)//����3����������У����Լ�break���ѡ��
	{
	case 0: Sort();//logo=0Ϊ��ֵ�˲� �ȵ�Sort����ȥ�����ٵ�Select_Median����ѡ��ֵ
	case 1: Select_Average();//logo=1Ϊ��ֵ�˲�
	case 2: Select_Weighted();//logo=2Ϊ��Ȩ�˲�
	}
}
//--------------------------------------------------------------------------------------------------------------------------------
double Input()//-------------------------------------------------------------------------��ʼ��һЩ��Ҫ����-----------------------
{
	PrintStar();
	printf("ģ��Ĵ�СΪ%d*%d\n",template_size,template_size);
	printf("���ڱ��⣺Original Image-------------------ԭʼͼ��\n");
	printf("���ڱ��⣺Median Filter Image--------------��ֵ�˲����ͼ��\n");
	printf("���ڱ��⣺Average Filter Image-------------��ֵ�˲����ͼ��\n");
	printf("���ڱ��⣺Weighted(Gaussian) Filter Image--��Ȩ(��˹)�˲����ͼ��\n");
	PrintStar();
	p=template_size;//ģ�����
	q=template_size;//ģ�����
	logo=0;
	return 0;
}

//--------------------------------------------------------------------------------------------------------------------------------

void removeNoise(IplImage* pImg)//----------------------------------------------------------------------------������-----------------------------------
{
	; //����IplImageָ��
    //pImg = cvLoadImage("E:\\30%.bmp", 0);//����ͼ��

	IplImage* Img_Median;
	IplImage* Img_Average;
	IplImage* Img_Weighted;

	Img_Median=cvCloneImage(pImg);
	Img_Average=cvCloneImage(pImg);
	Img_Weighted=cvCloneImage(pImg);

	int height;//���������׳�����ֱ��дint height=pImg->height;�������⣬���Զ���͸�ֵ�ֿ�д
	int width;

	height=pImg->height;    //height������ǰ�浥������
	width=pImg->width;
	Input();
	m=height;
	n=width;
	int i,j,g,h,*z;
	for (i=0;i<m-p+1;i++)                   //ģ�����Ͻ���ͼ�������������������ƶ�
	{
		for (j=0;j<n-q+1;j++)
		{
			z=&a[0];                        //ָ�����ָ��һά�����׵�ַ���Դ���һ�δ��������������׳���
			for (g=i;g<i+p;g++)
			{
				for (h=j;h<j+q;h++)
				{
					*z=((uchar *)(pImg->imageData+g*pImg->widthStep))[h];//��ģ���е�����ֵ�浽һ��һά������������׳���
					z++;                    //�޸�ָ���ַ
				}
			}
			Judge_Method();
			((uchar *)(Img_Median->imageData+(i+(p-1)/2)*Img_Median->widthStep))[j+(q-1)/2]=result_median; //��ģ���м�λ��Ԫ���滻Ϊ���
			((uchar *)(Img_Average->imageData+(i+(p-1)/2)*Img_Average->widthStep))[j+(q-1)/2]=result_average; //��ģ���м�λ��Ԫ���滻Ϊ���
			((uchar *)(Img_Weighted->imageData+(i+(p-1)/2)*Img_Weighted->widthStep))[j+(q-1)/2]=result_weighted; //��ģ���м�λ��Ԫ���滻Ϊ���
		}
	}
	printf("��Ȩ(��˹)�˲���ģ��Ϊ��\n");
	for (i=0;i<p;i++)
	{
		for (j=0;j<q;j++)
		{
			printf("c[%d][%d]=%2d ",i,j,c[i][j]);
		}
		printf("\n");
	}
	PrintStar();
	cvNamedWindow( "Original Image", CV_WINDOW_AUTOSIZE );//��������
	cvMoveWindow( "Original Image", 500, 200 );//����ͼ������Ĵ���λ��
	cvShowImage( "Original Image", pImg );//��ʾԭʼͼ��

	cvNamedWindow( "Median Filter Image", CV_WINDOW_AUTOSIZE );//��������
	cvMoveWindow( "Median Filter Image", 900, 200 );//����ͼ������Ĵ���λ��
	cvShowImage( "Median Filter Image", Img_Median );//��ʾ��ֵ�˲���ͼ��

	cvNamedWindow( "Average Filter Image", CV_WINDOW_AUTOSIZE );//��������
	cvMoveWindow( "Average Filter Image", 500, 500 );//����ͼ������Ĵ���λ��
	cvShowImage( "Average Filter Image", Img_Average );//��ʾ��ֵ�˲���ͼ��

	cvNamedWindow( "Weighted Filter Image", CV_WINDOW_AUTOSIZE );//��������
	cvMoveWindow( "Weighted Filter Image", 900, 500 );//����ͼ������Ĵ���λ��
	cvShowImage( "Weighted Filter Image", Img_Weighted );//��ʾ��Ȩ(��˹)�˲���ͼ��

	cvWaitKey(0); //�ȴ�����

	cvDestroyWindow( "Original Image" );//����ԭʼͼ�񴰿�
	cvReleaseImage( &pImg ); //�ͷ�ͼ��

	cvDestroyWindow( "Median Filtering Image" );//������ֵ�˲�����
	cvReleaseImage( &Img_Median ); //�ͷ�ͼ��

	cvDestroyWindow( "Average Filtering Image" );//���پ�ֵ�˲�����
	cvReleaseImage( &Img_Average ); //�ͷ�ͼ��

	cvDestroyWindow( "Weighted Filtering Image" );//���ټ�Ȩ(��˹)�˲�����
	cvReleaseImage( &Img_Weighted ); //�ͷ�ͼ��
}
//--------------------------------------------------------------------------------------------------------------------------------
