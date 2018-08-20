//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef _CPUTOOLS_H_
#define _CPUTOOLS_H_

double cpuE(int *hlat, int *hH, double h,  int width, int height, int length){
		double E=0.0;
		int idx, idy, idz;
		for(idx=0; idx<width; idx++){
			for(idy=0; idy<height; idy++){			
				for(idz=0; idz<length; idz++){					
					int down = idz+1;
					int right = idx + 1;
				  	int front = idy + 1;
					// periodic boundary, check limits
					if(idx == width-1) right =0;
					if(idy == height-1) front=0;
					if(idz == length-1) down =0;
					int i=idz*width*length+idy*width+idx;
										
					E += 	(double)hlat[i]*hlat[width*length*idz+width*idy+right] + 
							(double)hlat[i]*hlat[width*length*idz+width*front+idx]+
					   		(double)hlat[i]*hlat[width*length*down+width*idy+idx];     
					
					

				}
			}
		}
		for(int i=0; i<width*length*height;i++)
			E += (double)h*hH[i]*hlat[i];
		return -E;
}

double cpuF(int *hlat, int width, int height, int length){
		double F = 0.0;
		double mfx1=0.0;
		double mfx2=0.0;
		double mfy1=0.0;
		double mfy2=0.0;
		double mfz1=0.0;
		double mfz2=0.0;
		double k = 2.0*PI/(double)width;
		int idx, idy, idz;
		for(idx=0; idx<width; idx++){
			for(idy=0; idy<height; idy++){			
				for(idz=0; idz<length; idz++){					
					
						int i=idz*width*length+idy*width+idx;
						mfx1 += hlat[i] * cos(k * (double)idx);					
					    mfx2 += hlat[i] * sin(k * (double)idx);					
					
						mfy1 += hlat[i] * cos(k * (double)idy);					
					    mfy2 += hlat[i] * sin(k * (double)idy);					

						mfz1 += hlat[i] * cos(k * (double)idz);					
					    mfz2 += hlat[i] * sin(k * (double)idz);					

				}
			}
		}
		//printf("mfx1=%f mfx2=%f\nmfy1=%f mfy2=%f\nmfz1=%f mfz2=%f\n", mfx1, mfx2, mfy1, mfy2, mfz1, mfz2);
		//getchar();

		mfx1 *= mfx1;
		mfx2 *= mfx2;

		mfy1 *= mfy1;
		mfy2 *= mfy2;

		mfz1 *= mfz1;
		mfz2 *= mfz2;
		F = (1.0/(double)(3*width*length*height)) * ( mfx1 + mfx2 + mfy1 + mfy2 + mfz1 + mfz2 );
		return F;
}

int cpuM(int *hM, int width, int height, int length){
	int M = 0;
	for(int i=0; i<width*height*length; i++)
		M += hM[i];

	return M;
}

#endif
