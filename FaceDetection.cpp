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
 
#ifdef _EiC
#define WIN32
#endif
 
static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

void detect_and_draw( IplImage* image );
 
const char* cascade_name ="haarcascade_frontalface_alt2.xml";

 
int main( int argc, char** argv )
{
	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
	storage = cvCreateMemStorage(0);
	cvNamedWindow( "result", 1 );

    IplImage* image = cvLoadImage( "01.jpg", 1 );
	detect_and_draw( image );
    cvWaitKey(0);
    cvReleaseImage( &image );
        
    return 0;
}
 
void detect_and_draw( IplImage* img )
{
    static CvScalar colors[] = 
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}}
    };
 
    double scale = 1.3;
    IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
    IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale),
                          cvRound (img->height/scale)), 8, 1 );
    int i;
 
    cvCvtColor( img, gray, CV_BGR2GRAY );
    cvResize( gray, small_img, CV_INTER_LINEAR );
    cvEqualizeHist( small_img, small_img );
    cvClearMemStorage( storage );
 
    if( cascade )
    {
        double t = (double)cvGetTickCount();
        CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,
                                            1.3, 2, 0, cvSize(10, 10) );
        t = (double)cvGetTickCount() - t;
        printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
		
		for( i = 0; i < (faces ? faces->total : 0); i++ )
		{
			CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
			CvPoint pt1;
			CvPoint pt2;
			pt1.x = cvRound(r->x*scale); 
			pt2.x = cvRound((r->x+r->width)*scale); 
			pt1.y = cvRound(r->y*scale);
			pt2.y = cvRound((r->y+r->height)*scale);
			
			cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 ); 
		}
	}

    cvShowImage( "result", img );
	cvSaveImage("result.jpg", img);
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );
}