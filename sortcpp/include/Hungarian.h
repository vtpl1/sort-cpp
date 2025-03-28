#pragma once
#ifndef Hungarian_h
#define Hungarian_h

///////////////////////////////////////////////////////////////////////////////
// Hungarian.h: Header file for Class HungarianAlgorithm.
//
// This is a C++ wrapper with slight modification of a hungarian algorithm implementation by Markus Buehren.
// The original implementation is a few mex-functions for use in MATLAB, found here:
// http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem
//
// Both this code and the orignal code are published under the BSD license.
// by Cong Ma, 2016
//

#include <iostream>
#include <vector>
#include "sortcpp_export.h"
class SORTCPP_EXPORT HungarianAlgorithm
{
  public:
    HungarianAlgorithm() = default;
    ~HungarianAlgorithm() = default;
    static double Solve(std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment);

  private:
    static void assignmentoptimal(int* assignment, double* cost, double* distMatrix, int nOfRows, int nOfColumns);
    static void buildassignmentvector(int* assignment, const unsigned char* starMatrix, int nOfRows, int nOfColumns);
    static void computeassignmentcost(const int* assignment, double* cost, const double* distMatrix, int nOfRows);
    static void step2a(int* assignment, double* distMatrix, unsigned char* starMatrix, unsigned char* newStarMatrix,
                unsigned char* primeMatrix, unsigned char* coveredColumns, unsigned char* coveredRows, int nOfRows,
                int nOfColumns, int minDim);
    static void step2b(int* assignment, double* distMatrix, unsigned char* starMatrix, unsigned char* newStarMatrix,
                unsigned char* primeMatrix, unsigned char* coveredColumns, unsigned char* coveredRows, int nOfRows,
                int nOfColumns, int minDim);
    static void step3(int* assignment, double* distMatrix, unsigned char* starMatrix, unsigned char* newStarMatrix,
               unsigned char* primeMatrix, unsigned char* coveredColumns, unsigned char* coveredRows, int nOfRows,
               int nOfColumns, int minDim);
    static void step4(int* assignment, double* distMatrix, unsigned char* starMatrix, unsigned char* newStarMatrix,
               unsigned char* primeMatrix, unsigned char* coveredColumns, unsigned char* coveredRows, int nOfRows,
               int nOfColumns, int minDim, int row, int col);
    static void step5(int* assignment, double* distMatrix, unsigned char* starMatrix, unsigned char* newStarMatrix,
               unsigned char* primeMatrix, unsigned char* coveredColumns, unsigned char* coveredRows, int nOfRows,
               int nOfColumns, int minDim);
};

#endif	// Hungarian_h
