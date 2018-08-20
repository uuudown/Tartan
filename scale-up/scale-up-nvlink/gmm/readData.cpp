/*
 * readData.cpp
 *
 *  Created on: Nov 4, 2008
 *      Author: Doug Roberts
 *      Modified by: Andrew Pangborn
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "gaussian.h"

using namespace std;

extern "C"
float* readData(char* f, int* ndims, int* nevents);
float* readBIN(char* f, int* ndims, int* nevents);
float* readCSV(char* f, int* ndims, int* nevents);

float* readData(char* f, int* ndims, int* nevents) {
    int length = strlen(f);
    printf("File Extension: %s\n",f+length-3);
    if(strcmp(f+length-3,"bin") == 0) {
        return readBIN(f,ndims,nevents);
    } else {
        return readCSV(f,ndims,nevents);
    }
}

float* readBIN(char* f, int* ndims, int* nevents) {
    FILE* fin = fopen(f,"rb");

    fread(nevents,4,1,fin);
    fread(ndims,4,1,fin);
    printf("Number of elements removed for memory alignment: %d\n",*nevents % (16 * 2));
    *nevents -= *nevents % (16 * 2) ; // 2 gpus
    int num_elements = (*ndims)*(*nevents);
    printf("Number of rows: %d\n",*nevents);
    printf("Number of cols: %d\n",*ndims);
    float* data = (float*) malloc(sizeof(float)*num_elements);
    fread(data,sizeof(float),num_elements,fin);
    fclose(fin);
    return data;
}

float* readCSV(char* f, int* ndims, int* nevents) {
    string line1;
    ifstream file(f);
    vector<string> lines;
    int num_dims = 0;
    char* temp;
    float* data;

    if (file.is_open()) {
        while(!file.eof()) {
            getline(file, line1);

            if (!line1.empty()) {
                lines.push_back(line1);
            }
        }

        file.close();
    }
    else {
        cout << "Unable to read the file " << f << endl;
        return NULL;
    }
   
    if(lines.size() > 0) {
        line1 = lines[0];
        string line2 (line1.begin(), line1.end());

        temp = strtok((char*)line1.c_str(), ",");

        while(temp != NULL) {
            num_dims++;
            temp = strtok(NULL, ",");
        }

        lines.erase(lines.begin()); // Remove first line, assumed to be header
        int num_events = (int)lines.size();
        
        #if TRUNCATE == 1
            printf("Number of events removed to ensure memory alignment %d\n",num_events % (16 * 2));
            num_events -= num_events % (16 * 2);
        #endif

        // Allocate space for all the FCS data
        data = (float*)malloc(sizeof(float) * num_dims * (num_events));
        if(!data){
            printf("Cannot allocate enough memory for FCS data.\n");
            return NULL;
        }

        for (int i = 0; i < num_events; i++) {
            temp = strtok((char*)lines[i].c_str(), ",");

            for (int j = 0; j < num_dims; j++) {
                if(temp == NULL) {
                    free(data);
                    return NULL;
                }
                data[i * num_dims + j] = atof(temp);
                temp = strtok(NULL, ",");
            }
        }

        *ndims = num_dims;
        *nevents = num_events;

        return data;    
    } else {
        return NULL;
    }
    
    
}
