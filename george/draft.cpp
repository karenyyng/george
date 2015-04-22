/*
 * =====================================================================================
 *
 *       Filename:  draft.cpp
 *
 *    Description:  draft for deriv kernel class   
 *
 *        Version:  1.0
 *        Created:  04/22/2015 11:25:43
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h> 

#include <cmath> 
#include <cfloat>
#include <vector>
#include <stdio.h>
#include <iostream>

using std::vector;  // avoid having to write std:: all the time

int** get_termB_ixes(){
    unsigned int r, c;
    const int rows = 6, cols = 4;
    int arr[rows][cols] = {{0, 1, 2, 3}, 
                           {0, 2, 1, 3},
                           {0, 3, 1, 2},
                           {2, 3, 0, 1},
                           {1, 3, 0, 2},
                           {1, 2, 0, 3}};

    int** ix;
    ix = new int* [rows];

    for (r = 0; r < rows; r++){
        ix[r] = new int [cols];
        for(c = 0; c < cols; c++){
            ix[r][c] = arr[r][c];
            std::cout << ix[r][c] << " "; 
        }
        std::cout << std::endl;
    }
    return ix;
}

void del_termB_ixes(int** ix){ 
    unsigned int r;
    const int rows = 6;

    for (r=0; r < rows; r++){
        delete [] ix[r];
    }
    delete [] ix;
}

int main(){
    int** ptr = get_termB_ixes();

    for (r = 0; r < rows; r++){
        ix[r] = new int [cols];
        for(c = 0; c < cols; c++){
            ix[r][c] = arr[r][c];
            std::cout << ix[r][c] << " "; 
        }
        std::cout << std::endl;
    }
    

    del_termB_ixes(ptr);
    ptr = NULL;
    
    return 0;
}
