/*  \lf2depth_mrf.h 
**  Refine the disparity map from the cost volume using multi-label optimization.
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


#ifndef _LF2DEPTH_MRF
#define _LF2DEPTH_MRF

#include <limits>
#include <opencv2/opencv.hpp>
#include "gco/GCoptimization.h"
#include "volume_filtering.h"
#include "light_field.h"
#include "misc.h"
using namespace std;
using namespace cv;

/**
    Volume filtering wrapper
    @cost            input  3D cost volume stack
    @cost_filter     output filtered 3D cost volume stack
    @lf_ptr          light field structure pointer 
*/
bool cost_filtering( float* cost, float* cost_filter, LF* lf_ptr){

    int w = lf_ptr->W;
    int h = lf_ptr->H;
    int l = lf_ptr->nlabels;

    volume_filtering(cost, cost_filter, w, h, l);

    return true;
}

/**
    Merge the vertical and horizontal volumes to a single volume.
    @depth_x  the horizontal volume as input
    @depth_y  the vertical   volume as input
    @cost     the merged volume as output
    @lf_ptr   the light field structure pointer
*/
void volume_merge(  float* depth_x, 
                    float* depth_y,                
                    float* cost, 
                    LF* lf_ptr){
                   
	for (int j = 1; j < (lf_ptr->H-1); j++)
		for (int i = 1; i < (lf_ptr->W-1); i++) 

		    for (int k = 0; k < lf_ptr->nlabels; k++){

		        int idx  = lf_ptr->nlabels*(j*lf_ptr->W+i)+k;  
		        //cost[lf_ptr->nlabels*(j*lf_ptr->W+i)+k]= depth_x[idx]>depth_y[idx]? depth_x[idx]:depth_y[idx];              
                cost[lf_ptr->nlabels*(j*lf_ptr->W+i)+k]= (depth_x[idx]+depth_y[idx])/2;            
            }                                 
}

/**
    Using Makov Random Field (Multi-label optimization) to refine the disparity map.
    @depth_x  the horizontal volume as input
    @depth_y  the vertical   volume as input
    @confidence_x   the horizontal confidence map
    @confidence_y   the vertical   confidence map 
    @depth_best_xy  the first estimate disparity map   
    @lf_ptr   the light field structure pointer
*/

bool lf2depth_mrf(  float *depth_x,
                    float *depth_y,
                    float *confidence_x,
                    float *confidence_y,
                    uchar *depth_best_xy,                    
                    LF* lf_ptr){

    int width  = lf_ptr->W;
    int height = lf_ptr->H;
    int num_pixels = width*height;
    int num_labels = lf_ptr->nlabels;
    float *cost         = new float[num_pixels*num_labels];
    float* cost_filter;
    float* cost2        = new float[num_pixels*num_labels];
    cout<<" ======== MRF Refinement Result ======>>>>>>>"<<endl;
    volume_merge(depth_x, depth_y, cost, lf_ptr);

    //need rearrange the cost volume
    if (lf_ptr->type==1)
     cost_filter  = new float[num_pixels*num_labels];   

        for (int l=0; l<num_labels; l++ )
	        for (int j = 0; j<height; j++)
		        for (int i = 0; i<width; i++){		    
		            cost2[l*num_pixels+j*width+i]=cost[num_labels*(j*width+i)+l];
		            if (lf_ptr->type==1)
		                cost_filter[l*num_pixels+j*width+i]=cost[num_labels*(j*width+i)+l];
                }
          
    if (lf_ptr->type==1)//3D denosing for nosiy data
        cost_filtering(cost2, cost_filter, lf_ptr);
    else 
        cost_filter =  cost2;  
        
    int *smooth = new int[num_labels*num_labels];
	    for (int l1 = 0; l1 < num_labels; l1++)
		    for (int l2 = 0; l2 < num_labels; l2++)
			    smooth[l1 + l2*num_labels] = abs(l1 - l2);//abs(l1 - l2);//*(l1 - l2);//abs(l1 - l2);//l1 norm

    try{

	    GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels, num_labels);

        if (lf_ptr->lambda==0) lf_ptr->lambda=1;
	    //add data costs individually
	    for (int j = 0; j<height; j++)
		    for (int i = 0; i<width; i++){

		        int idx = j*width+i;
		        for (int k = 0; k<num_labels; k++){
		        
		        	if ((j>4)&&(i>4)&&(j<(width-4))&&(i<(height-4))&&lf_ptr->disparity_mask.at<float>(j,i)){
		               		            
		                if ((confidence_x[idx]<lf_ptr->threshold)&&(confidence_y[idx]<lf_ptr->threshold))
		                    gc->setDataCost(j*width+i, k, 0);				            
		                else
		                    gc->setDataCost(j*width+i, k, (lf_ptr->lambda*cost_filter[k*(height*width)+j*width+i])); 			 
			        }
			        else 
			            gc->setDataCost(j*width+i, k, 0); 
			    }
		    }

		    // now set up a grid neighborhood system
		    // first set up horizontal neighbors
		    for (int y = 0; y < height; y++ )
			    for (int  x = 1; x < width; x++ ){
				    int p1  = x-1+y*width;
				    int p2  = x+  y*width;													
				    int  weight =  confidence_x[p2]>lf_ptr->threshold ? 0: 1;						    
				    gc->setNeighbors(p1,p2,lf_ptr->disparity_mask.at<float>(y,x)*weight);	
			    }

		    // next set up vertical neighbors
		    for (int y = 1; y < height; y++ )
			    for (int  x = 0; x < width; x++ ){
				    int p1 = x+(y-1)*width;
				    int p2 = x+y*width;								
				    int  weight =  confidence_y[p2]>lf_ptr->threshold ? 0: 1;						
				    gc->setNeighbors(p1,p2,lf_ptr->disparity_mask.at<float>(y,x)*weight);
			    }

	    gc->setSmoothCost(smooth);
    	
	    //set initial guess
	    /*
	    for (int j = 0; j<height; j++)
		    for (int i = 0; i<width; i++){	
		        float grad_x = fabs(confidence_x[j*width+i+1]  -confidence_x[j*width+i-1]);
		        float grad_y = fabs(confidence_y[(j+1)*width+i]-confidence_y[(j-1)*width+i]);
		
	                gc->setLabel(j*width+i,depth_best_xy[j*width+i]);//
	          
	    }*/ 
    	
        gc->setVerbosity(1);
	    printf("Before optimization energy is %lld\n", gc->compute_energy());
	    gc->expansion(1);
        printf("After optimization energy is %lld\n", gc->compute_energy());
	    for (int j = 0; j<height; j++)
		    for (int i = 0; i<width; i++)
			    lf_ptr->depth.at<float>(j,i) = gc->whatLabel(j*width+i);
      
	    delete gc;
    }

    catch (GCException e){
	    e.Report();
    }
        
    delete[] cost;
    delete[] cost2;
    if (lf_ptr->type==1)
	    delete[] cost_filter;
	delete[] smooth;	    	    
        
        
}
#endif

