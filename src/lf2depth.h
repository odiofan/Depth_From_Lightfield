/*  \lf2depth.h 
**  Extact depth from light field.
**  Copyright (C) 2012-2016 xxx @ National University of xxx.

**  This program is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
    
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**  GNU General Public License for more details.
   
**  You should have received a copy of the GNU General Public License
**  along with this program in the file "LICENSE".
**  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef _DEPTH_FROM_LIGHTFIELD
#define _DEPTH_FROM_LIGHTFIELD
#include <limits>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "WMF/JointWMF.h"
#include "gco/GCoptimization.h"
#include "light_field.h"
#include "lf2depth_mrf.h"
#include "volume_filtering.h"
#include "misc.h"

#define DEBUG

using namespace std;
using namespace cv;
float* d;

/**
    final result filtering using fast Weighted Median Filter(WMF)
    @lf_ptr          light field strutue pointer 
*/
//======output the depth mapped filtered with the central view
void depth_filtering(LF* lf_ptr){
    Mat img_grey;
    cvtColor(lf_ptr->imgc, img_grey, CV_RGB2GRAY);
    JointWMF wmf;
    lf_ptr->depth_f=wmf.filter(lf_ptr->depth, img_grey, 5, 25.5, 64, 256, 1, "exp");//cos
    //medianBlur ( lf.depth, lf.depth_f, 5 );
}

/**
    Calculate the disparity cost per pixel
    @img          EPI slice as input
    @depth        depth cost as output
    @depthc       depth confidence as output
    @lf_ptr       light field structure pointer 
*/
bool disparity_cost( const Mat& img,  float *depth, float *depthc,  LF* lf_ptr){

    //medianBlur ( img, img, 5);
    
    //scan the slice
	float data_new[3]; 	  
	Vec3f *data_ptr = (Vec3f*)(img.data); 	 
	 
	for (int i=3;i<(img.cols-3); i++){

	    Vec3f     data  = img.at<Vec3f>((lf_ptr->U-1)/2,i);//Vec3b
     
	    float* depth_addr  = depth  + i*lf_ptr->nlabels;
	    float* depthc_addr = depthc + i*lf_ptr->nlabels;

             //coarse search 
		     for (int k=0; k<lf_ptr->nlabels; k++){ //-64 64
  
		          float err0=0;
                  float avg =0;
                  for (int t=-3; t<=3; t++){
					  if (abs(t)>0){
					  //if (t==1){
				      float xnew  = (i + d[k] * float(t));//(i + d * float(t))*8;	
				      	 
                      float idy =((lf_ptr->U-1)/2+t);
                      int idx = int(idy)*img.cols+int(xnew);
					  //bilinear interpolation
                      float a,b;
					  a = 1-(xnew- int(xnew));
					  b = 1 -a;		
		              data_new[0] = data_ptr[idx][0]*a + data_ptr[idx+1][0]*b ;  
		              data_new[1] = data_ptr[idx][1]*a + data_ptr[idx+1][1]*b ; 
		              data_new[2] = data_ptr[idx][2]*a + data_ptr[idx+1][2]*b ;                                               
		              float tmp[3];
		              tmp[0] = data[0] -data_new[0];//norm(data -data_new); 
		              tmp[1] = data[1] -data_new[1];//norm(data -data_new);  
		        	  tmp[2] = data[2] -data_new[2];//norm(data -data_new);                                          
		        	  avg    = avg    + data_new[0] + data_new[1] + data_new[2];
		              err0 += tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2];
		              //err0+=tmp[2];// 
		              //err0= data_new[0];	                              		                            		 
					  }
				  }
				  	  
 			*depth_addr++   =  err0;
 			*depthc_addr++  =  avg;//(avg[0]+avg[1]+avg[2])/3.0f;
			}   		 			 
	}
    return true;
}

/**
    Build the cost volume.
    @epi_h         Horizontal EPI slices as input
    @epi_v         Vertical   EPI slices as input
    @depth_x       Horizontal cost volume
    @depth_y       Vertical   cost volume
    @depth_cx      Horizontal confidence volume
    @depth_cy      Vertical   confidence volume
    @lf_ptr        light field structure pointer         
*/
void cost_volume(     vector<Mat>& epi_h, 
                	  vector<Mat>& epi_v,  
                      float* depth_x,
			          float* depth_y,
                      float* depth_cx,
			          float* depth_cy,			          
                      LF* lf_ptr){
    


    //discrete depth value
    d  = new float[lf_ptr->nlabels+1];   
    
    float dmin=lf_ptr->d_min;
    float dmax=lf_ptr->d_max;   
  	for (int k=0; k<=lf_ptr->nlabels; k++){
		d[k]= dmin+float(k)*(dmax-dmin)/float(lf_ptr->nlabels);
	    //cout<<d[k]<<endl;	
    }

    //=============Horizontal==================
    #pragma omp parallel for
	for (int j = 0; j < lf_ptr->H; j++) {
        //cout<<j<<endl;
        //allocate memeory inside the loop to enable multi threads
        float *depth_array           =  (float*) calloc (lf_ptr->nlabels*lf_ptr->W, sizeof(float)); 
	    float *con_array             =  (float*) calloc (lf_ptr->nlabels*lf_ptr->W, sizeof(float)); 

		disparity_cost( epi_h[j], depth_array, con_array, lf_ptr);  	 
		memcpy(depth_x + lf_ptr->nlabels*j*lf_ptr->W, depth_array,  lf_ptr->W*lf_ptr->nlabels * sizeof(float));
		memcpy(depth_cx+ lf_ptr->nlabels*j*lf_ptr->W, con_array,    lf_ptr->W*lf_ptr->nlabels * sizeof(float));		  

	    delete[] depth_array;
        delete[] con_array;	
	}
	
	cout<<" Extracting Horizontal EPI Slices Done"<<endl;

    //==============Vertical=====================
    #pragma omp parallel for
	for (int i = 0; i < lf_ptr->W; i++) {
	    //cout<<"\rVertical Slice   "<<cnt<<"/"<<lf_ptr->W;
		
        float *depth_array           =  (float*) calloc (lf_ptr->nlabels*lf_ptr->W, sizeof(float)); 
	    float *con_array             =  (float*) calloc (lf_ptr->nlabels*lf_ptr->W, sizeof(float)); 	         
            
        disparity_cost( epi_v[i], depth_array, con_array, lf_ptr);
			
		for (int j = 0; j < lf_ptr->H; j++){	
		int idx = j*lf_ptr->W+i;			    
	    memcpy(depth_y  + lf_ptr->nlabels*(idx), depth_array+ j*lf_ptr->nlabels, lf_ptr->nlabels * sizeof(float));
	    memcpy(depth_cy + lf_ptr->nlabels*(idx), con_array  + j*lf_ptr->nlabels, lf_ptr->nlabels * sizeof(float));		    
        }

        delete[] depth_array;
        delete[] con_array;
	}
	cout<<" Extracting Vertical   EPI Slices Done"<<endl;
    
}

/**
    Brute-force searching the optimal depth per pixel to produce the disparity map.
    @data          A cost volume as input
    @data_best     The disparity map as output
    @lf_ptr        light field structure pointer         
*/
bool depth_optimal( float* data, uchar* data_best, LF* lf_ptr){

	for (int i=0; i<lf_ptr->W*lf_ptr->H; i++){	
					
	    float value=10000000;
	    int offset = i*lf_ptr->nlabels;
        int idx =0;

	    for (int k=0; k<lf_ptr->nlabels; k++){
	        float d =  data[offset+k];
	        
	        //if (i==406*640+27) cout<<d<<endl;
	        if (value>d){
	           value=d;
	           idx = k;
	        }
	    }
	    data_best[i]=idx;	  
	    //cout<<idx<<endl;  
	}
    //cout<<"===================================="<<endl;
    return true;
}

/*
    Brute-force searching the optimal slope (depth) for a pixel.
    @data         A single stack for one pixel
    @num          The number of layer for the stack (volume)
    @idx          The optimal slope index
    @score        The score for the optimal slope           
*/
bool depth_optimal_pixel(float* data, int num, int& idx, float& score){

idx = 0;
float avg=0;
float value_min=100000;
float value_max=0;
//float diff=0;

int idx_min;
int idx_max;

for (int k=0; k<num; k++){

    avg = avg + data[k];

    if (value_min>data[k]){
        value_min=data[k];
        idx = k;
        idx_min=k;   
    }
    if (value_max<data[k]){
        value_max=data[k]; 
        idx_max=k; 
    }   
}

//find the total number o pixels that larger than average value
int cnt = 0;
avg = avg/num;
for (int k=0; k<num; k++){
    if (data[k]>avg)
    cnt++;
}

score = float(cnt) / float(num);
//cout<<avg<<" "<<value_min<<" "<<score<<endl;

return 0;//
}

/**
    Find the optimal disparity(depth) per pixel to produce the disaprity map
    based on the horizontal and vertical volume stacks.
    @data1   the horizontal volume (variance)
    @data2   the vertical   volume (variance)
    @conf1   the horizontal volume (average)
    @conf2   the vertical   volume (average)   
    @cf1     the horizontal image (average) 
    @cf2     the vertical   image (average)       
    @data_best  the 2D disaprity
    @lf_ptr  the light field structure pointer
*/
bool compute_slope_xy( float* data1, float* data2,
                 float* conf1, float* conf2, 
                 float* cf1,   float* cf2,  
                 uchar* data_best,                                                  
                 LF* lf_ptr){

    int height =  lf_ptr->H;
    int width  =  lf_ptr->W;
    int labels =  lf_ptr->nlabels;

    int cnt=0;
    for (int j=0; j< height; j++){
	    for (int i=0; i< width; i++){
	    	
	    	if ((j>0)&&(i>0)&&(j<(height-1))&&(i<(width-1))){    					   
                float score[2]={0,0};
                int idx[2]    ={0,0};                   
                
                depth_optimal_pixel(&data1[cnt*labels], labels, idx[0], score[0]);
                depth_optimal_pixel(&data2[cnt*labels], labels, idx[1], score[1]); 
                
                int offsetx0  = cnt*labels +idx[0];
                int offsety0  = cnt*labels +idx[1];            
                int offsety1 = ((j-1)*width+i)*labels   +idx[1];
                int offsetx1 = (    j*width+i-1)*labels +idx[0];  
                int offsety2 = ((j+1)*width+i)*labels   +idx[1];
                int offsetx2 = (    j*width+i+1)*labels +idx[0];  
                             
                  
                float cx1 = fabs(conf1[offsetx0]-conf1[offsetx1]);
                float cy1 = fabs(conf2[offsety0]-conf2[offsety1]);           
                float cx2 = fabs(conf1[offsetx2]-conf1[offsetx1]);
                float cy2 = fabs(conf2[offsety2]-conf2[offsety1]); 
                                
                data_best[cnt]=  cx2>cy2? idx[0]:idx[1];  //
                if (score[0]>0.4) 
                    cf1[cnt]      =  cx1/18.0f;
                else  
                    cf1[cnt]      =  0;
                if (score[1]>0.4) 
                    cf2[cnt]      =  cy2/18.0f;
                else  
                    cf2[cnt]      =  0;
               
                //cout<<score[0]<<" "<<score[1]<<endl;
            }
            else{
                data_best[cnt]=  0;  
                cf1[cnt]      =  0;
                cf2[cnt]      =  0;
            }
            cnt++;   
            
            //if ((j==381)&&(i==400)){
            //    mem2hdf5  ("error_debugx.h5", "data", H5T_NATIVE_FLOAT, 64, 1, data1+((381)*width+400)*labels);
            //    mem2hdf5  ("error_debugy.h5", "data", H5T_NATIVE_FLOAT, 64, 1, data2+((381)*width+400)*labels);            
            //}               
        }                     
	}
	return true;
}

/**
    Extact the depth from horizontal and vertical EPI slices
    @epi_h    horizontal EPI slices as input
    @epi_v    vertical   EPI slices as input
    @lf_ptr   the light field structure pointer
    @depth    the final result
*/
bool lf2depth(LF* lf_ptr){

    int width  = lf_ptr->W;
    int height = lf_ptr->H;
    int num_pixels = width*height;
    int num_labels = lf_ptr->nlabels;

    if (lf_ptr->type==0){ //HCI
        lf_ptr->d_min=lf_ptr->dt_min;
        lf_ptr->d_max=lf_ptr->dt_max;
    }
    if (fabs(lf_ptr->d_min-lf_ptr->d_max)>2)
        lf_ptr->nlabels=64;
    else
        lf_ptr->nlabels=64;
    num_labels = lf_ptr->nlabels;


    float *depth_x      = new float[num_pixels*num_labels];
    float *depth_y      = new float[num_pixels*num_labels];
    float *depth_cx     = new float[num_pixels*num_labels];
    float *depth_cy     = new float[num_pixels*num_labels];
    float *confidence_x = (float*) calloc (num_pixels, sizeof(float)); 
    float *confidence_y = (float*) calloc (num_pixels, sizeof(float)); 
    uchar *depth_best_x = (uchar*) calloc (num_pixels, sizeof(uchar)); 
    uchar *depth_best_y = (uchar*) calloc (num_pixels, sizeof(uchar)); 
    uchar *depth_best_xy  = new uchar[num_pixels];
    float *cost         = new float[num_pixels*num_labels];
    float* cost_filter;
    float* cost2        = new float[num_pixels*num_labels];

    int64 t0, t1;
    t0 = cv::getTickCount();
    cost_volume(lf_ptr->epi_h, lf_ptr->epi_v, depth_x, depth_y, depth_cx, depth_cy, lf_ptr); //build the cost volume
    compute_slope_xy    (depth_x, depth_y, depth_cx, depth_cy, confidence_x, confidence_y, depth_best_xy, lf_ptr);//===xy estimate
    
    t1 = cv::getTickCount();   
    cout<<"Time spent "<<(t1-t0)/cv::getTickFrequency()<<" Seconds"<<endl;  
    vector<Mat>().swap(lf_ptr->epi_h); 
    vector<Mat>().swap(lf_ptr->epi_v);
    
    //===================================================================
    #ifdef DEBUG
    Mat rrr = Mat(height, width, CV_8U, depth_best_xy);
    color_map_confidence(lf_ptr, rrr, "./debug/data/depth0.png", 0, confidence_x, confidence_y, 1); 
    color_map_confidence(lf_ptr, rrr, "./debug/data/depth1.png", 2, confidence_x, confidence_y, 1);      
    color_map_confidence(lf_ptr, rrr, "./debug/data/depth2.png", 4, confidence_x, confidence_y, 1);
    color_map_confidence(lf_ptr, rrr, "./debug/data/depth4.png", 6, confidence_x, confidence_y, 1);
    depth_optimal(depth_x, depth_best_x,  lf_ptr);  
    depth_optimal(depth_y, depth_best_y,  lf_ptr); 
    Mat testx( height, width, CV_8U, depth_best_x);  
    Mat testy( height, width, CV_8U, depth_best_y);  
    imwrite("./debug/data/testx.png", testx);
    imwrite("./debug/data/testy.png", testy);
    Mat ccx (  height, width, CV_32F, confidence_x);
    Mat ccy (  height, width, CV_32F, confidence_y);
    Mat ddx (  height, width, CV_8U, depth_best_x);
    Mat ddy (  height, width, CV_8U, depth_best_y);
    

        
    Mat cvx (  height*width*lf_ptr->nlabels, 1, CV_32F, depth_cx);
    Mat cvy (  height*width*lf_ptr->nlabels, 1, CV_32F, depth_cy); 
    Mat dvx (  height*width*lf_ptr->nlabels, 1, CV_32F, depth_x);
    Mat dvy (  height*width*lf_ptr->nlabels, 1, CV_32F, depth_y);          
    mat2hdf5("./debug/data/cvx.h5", "data", H5T_NATIVE_FLOAT, float(), cvx);                     
    mat2hdf5("./debug/data/cvy.h5", "data", H5T_NATIVE_FLOAT, float(), cvy);
    mat2hdf5("./debug/data/dvx.h5", "data", H5T_NATIVE_FLOAT, float(), dvx);                     
    mat2hdf5("./debug/data/dvy.h5", "data", H5T_NATIVE_FLOAT, float(), dvy);          
    mat2hdf5("./debug/data/cx.h5", "data", H5T_NATIVE_FLOAT, float(), ccx);                     
    mat2hdf5("./debug/data/cy.h5", "data", H5T_NATIVE_FLOAT, float(), ccy);
    mat2hdf5("./debug/data/dxu.h5","data", H5T_NATIVE_UCHAR, uchar(), ddx);                     
    mat2hdf5("./debug/data/dyu.h5","data", H5T_NATIVE_UCHAR, uchar(), ddy);
    cout<<int(testx.at<uchar>(406,27))<<endl;
    cout<<int(testy.at<uchar>(406,27))<<endl;
    cout<<int(ddx.at<uchar>(406,27))<<endl;
    cout<<int(ddy.at<uchar>(406,27))<<endl;    
   
    Mat dddx( height, width, CV_32F);
    Mat dddy( height, width, CV_32F);     
    Mat dddd( height, width, CV_32F);            
    label2depth( ddx, dddx, lf_ptr);
    label2depth( ddy, dddy, lf_ptr);
    label2depth( rrr, dddd, lf_ptr); 
    mat2hdf5("./debug/data/dx.h5","data", H5T_NATIVE_FLOAT, float(), dddx);                     
    mat2hdf5("./debug/data/dy.h5","data", H5T_NATIVE_FLOAT, float(), dddy);
    mat2hdf5("./debug/data/d.h5" ,"data", H5T_NATIVE_FLOAT, float(), dddd);     
    mat2hdf5("./debug/data/gt.h5","data", H5T_NATIVE_FLOAT, float(), lf_ptr->disparity_gt);
       
    color_map(ddx(Rect(20,20,width-40,height-40)),  "./debug/data/depth_x2.png",       0);
    color_map(ddy(Rect(20,20,width-40,height-40)),  "./debug/data/depth_y2.png",       0);
    if (lf_ptr->type==0){
        cout<<" ======== Horizontal Result     ======>>>>>>>"<<endl;
        evaluate_depth         (depth_best_x,  "./debug/data/depth_x.png",   lf_ptr);//===x
        cout<<" ======== Vertical   Result     ======>>>>>>>"<<endl;
        evaluate_depth         (depth_best_y,  "./debug/data/depth_y.png",   lf_ptr);//===y    
        cout<<" ======== Our Merge  Result     ======>>>>>>>"<<endl;
        evaluate_depth         (depth_best_xy, "./debug/data/depth_xy_real.png",   lf_ptr);
        /*
        cout<<" ======== Best Merge Result     ======>>>>>>>"<<endl;
        depth_merge_gt(depth_best_x,depth_best_y,depth_best_xy, lf_ptr);
        evaluate_depth         (depth_best_xy, "depth_xy_ideal.png",  lf_ptr);//===xy best
        */
    }
    #endif
    //=================================================================== 
    
    if (lf_ptr->type==1){ //Refine the depth result for Lytro data
        lf2depth_mrf(depth_x, depth_y, confidence_x, confidence_y, depth_best_xy, lf_ptr);
    }
    else {//Just copy
        Mat result = Mat(height, width, CV_8U, depth_best_xy);
        result.convertTo(lf_ptr->depth, CV_32F);
    }
    depth_filtering(lf_ptr);//post filtering
    color_map(lf_ptr->depth(Rect(20,20,width-40,height-40)),  lf_ptr->depth_filename.c_str(),       0);
    color_map(lf_ptr->depth_f(Rect(20,20,width-40,height-40)),lf_ptr->depth_filter_filename.c_str(),0);
    grey_map (lf_ptr->depth_f(Rect(20,20,width-40,height-40))*4,"./debug/data/r.png",0);
 
	delete[] depth_x;
	delete[] depth_y;
	delete[] depth_cx;
	delete[] depth_cy;	
	delete[] confidence_x;
	delete[] confidence_y;
	delete[] depth_best_x;
	delete[] depth_best_y;
	delete[] depth_best_xy;
    delete[] d;
	 
	return true;
}

#endif
