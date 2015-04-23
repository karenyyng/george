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

vector< vector<int> > get_termB_ixes(){
    unsigned int r, c;
    vector<vector<int> > v2d;
    vector<int> rowvector;
    const int rows = 6, cols = 4;

    int arr[rows][cols] = {{0, 1, 2, 3}, 
                           {0, 2, 1, 3},
                           {0, 3, 1, 2},
                           {2, 3, 0, 1},
                           {1, 3, 0, 2},
                           {1, 2, 0, 3}};

    for (r = 0; r < rows; r++) {
        rowvector.clear();
        for (c = 0; c < cols; c++ ) {
            rowvector.push_back(arr[r][c]);
            std::cout << arr[r][c];
        } 
        std::cout << std::endl;
        v2d.push_back(rowvector);
    }

    return v2d;
}


int main(){
    const vector< vector<int> > B_ix  = get_termB_ixes();
    std::cout << B_ix.size() << std::endl; 
    std::cout << B_ix[0].size() << std::endl; 
    return 0;
}
