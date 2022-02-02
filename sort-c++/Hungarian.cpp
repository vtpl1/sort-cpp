///////////////////////////////////////////////////////////////////////////////
// Hungarian.cpp: Implementation file for Class HungarianAlgorithm.
//
// This is a C++ wrapper with slight modification of a hungarian algorithm implementation by Markus Buehren.
// The original implementation is a few mex-functions for use in MATLAB, found here:
// http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem
//
// Both this code and the orignal code are published under the BSD license.
// by Cong Ma, 2016
//

#include <cfloat>
#include <cmath>

#include "Hungarian.h"
//********************************************************//
// A single function wrapper for solving assignment problem.
//********************************************************//
double HungarianAlgorithm::Solve(std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment)
{
    int nRows = static_cast<int>(DistMatrix.size());
    int nCols = static_cast<int>(DistMatrix[0].size());

    // double* distMatrixIn = new double[nRows * nCols];
    // int* assignment = new int[nRows];
    std::vector<double> distMatrixIn(static_cast<size_t>(nRows * nCols), 0.0);
    std::vector<int> assignment(nRows, 0);
    double cost = 0.0;

    // Fill in the distMatrixIn. Mind the index is "i + nRows * j".
    // Here the cost matrix of size MxN is defined as a double precision array of N*M elements.
    // In the solving functions matrices are seen to be saved MATLAB-internally in row-order.
    // (i.e. the matrix [1 2; 3 4] will be stored as a vector [1 3 2 4], NOT [1 2 3 4]).
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            distMatrixIn[i + nRows * j] = DistMatrix[i][j];
        }
    }

    // call solving function
    assignmentoptimal(assignment.empty() ? nullptr : assignment.data(), &cost,
                      distMatrixIn.empty() ? nullptr : distMatrixIn.data(), nRows, nCols);

    Assignment.clear();
    for (unsigned int r = 0; r < nRows; r++) {
        Assignment.push_back(assignment[r]);
    }

    // delete[] distMatrixIn;
    // delete[] assignment;
    return cost;
}

//********************************************************//
// Solve optimal solution for assignment problem using Munkres algorithm, also known as Hungarian Algorithm.
//********************************************************//
void HungarianAlgorithm::assignmentoptimal(int* assignment, double* cost, double* distMatrixIn, int nOfRows,
                                           int nOfColumns)
{

    // double* distMatrix;
    double* distMatrixTemp = nullptr;
    double* distMatrixEnd = nullptr;
    double* columnEnd = nullptr;
    double value = 0;
    double minValue = 0;
    // bool* coveredColumns;
    // bool* coveredRows;
    // bool* starMatrix;
    // bool* newStarMatrix;
    // bool* primeMatrix;
    int nOfElements = 0;
    int minDim = 0;
    int row = 0;
    int col = 0;

    /* initialization */
    *cost = 0;
    for (row = 0; row < nOfRows; row++) {
        if (assignment == nullptr) { // TODO(vtpl1): introduced
            break;
        }
        assignment[row] = -1;
    }

    /* generate working copy of distance Matrix */
    /* check if all matrix elements are positive */
    nOfElements = nOfRows * nOfColumns;
    std::vector<double> distMatrix(nOfElements);
    // distMatrix = (double*)malloc(nOfElements * sizeof(double));
    distMatrixEnd = distMatrix.empty() ? nullptr : &distMatrix[0] + nOfElements;

    for (row = 0; row < nOfElements; row++) {
        if ((distMatrixIn == nullptr) || distMatrix.empty()) { // TODO(vtpl1): introduced
            break;
        }
        value = distMatrixIn[row];
        if (value < 0) {
            std::cerr << "All matrix elements have to be non-negative." << std::endl;
        }
        distMatrix[row] = value;
    }

    /* memory allocation */
    // coveredColumns = (bool*)calloc(nOfColumns, sizeof(bool));
    // coveredRows = (bool*)calloc(nOfRows, sizeof(bool));
    // starMatrix = (bool*)calloc(nOfElements, sizeof(bool));
    // primeMatrix = (bool*)calloc(nOfElements, sizeof(bool));
    // newStarMatrix = (bool*)calloc(nOfElements, sizeof(bool)); /* used in step4 */

    std::vector<unsigned char> coveredColumns(nOfColumns, 0);
    std::vector<unsigned char> coveredRows(nOfRows, 0);
    std::vector<unsigned char> starMatrix(nOfElements, 0);
    std::vector<unsigned char> primeMatrix(nOfElements, 0);
    std::vector<unsigned char> newStarMatrix(nOfElements, 0);

    /* preliminary steps */
    if (nOfRows <= nOfColumns) {
        minDim = nOfRows;

        for (row = 0; row < nOfRows; row++) {
            /* find the smallest element in the row */
            distMatrixTemp = distMatrix.empty() ? nullptr : &distMatrix[0] + row;
            if ((distMatrixTemp == nullptr) || (distMatrixEnd == nullptr)) { // TODO(vtpl1): introduced
                break;
            }

            minValue = *distMatrixTemp;
            distMatrixTemp += nOfRows;
            while (distMatrixTemp < distMatrixEnd) {
                value = *distMatrixTemp;
                if (value < minValue) {
                    minValue = value;
                }
                distMatrixTemp += nOfRows;
            }

            /* subtract the smallest element from each element of the row */
            distMatrixTemp = distMatrix.empty() ? nullptr : &distMatrix[0] + row;
            if ((distMatrixTemp == nullptr) || (distMatrixEnd == nullptr)) { // TODO(vtpl1): introduced
                break;
            }
            while (distMatrixTemp < distMatrixEnd) {
                *distMatrixTemp -= minValue;
                distMatrixTemp += nOfRows;
            }
        }

        /* Steps 1 and 2a */
        for (row = 0; row < nOfRows; row++) {
            for (col = 0; col < nOfColumns; col++) {
                if (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON) {
                    if (coveredColumns[col] == 0) {
                        starMatrix[row + nOfRows * col] = 1;
                        coveredColumns[col] = 1;
                        break;
                    }
                }
            }
        }
    } else /* if(nOfRows > nOfColumns) */
    {
        minDim = nOfColumns;

        for (col = 0; col < nOfColumns; col++) {
            /* find the smallest element in the column */
            distMatrixTemp = distMatrix.empty() ? nullptr : &distMatrix[0] + static_cast<ptrdiff_t>(nOfRows * col);
            if (distMatrixTemp == nullptr) { // TODO(vtpl1): introduced
                break;
            }
            columnEnd = distMatrixTemp + nOfRows;

            minValue = *distMatrixTemp++;
            while (distMatrixTemp < columnEnd) {
                value = *distMatrixTemp++;
                if (value < minValue) {
                    minValue = value;
                }
            }

            /* subtract the smallest element from each element of the column */
            distMatrixTemp = distMatrix.empty() ? nullptr : &distMatrix[0] + static_cast<ptrdiff_t>(nOfRows * col);
            if (distMatrixTemp == nullptr) { // TODO(vtpl1): introduced
                break;
            }
            while (distMatrixTemp < columnEnd) {
                *distMatrixTemp++ -= minValue;
            }
        }

        /* Steps 1 and 2a */
        for (col = 0; col < nOfColumns; col++) {
            for (row = 0; row < nOfRows; row++) {
                if (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON) {
                    if (coveredRows[row] == 0) {
                        starMatrix[row + nOfRows * col] = 1;
                        coveredColumns[col] = 1;
                        coveredRows[row] = 1;
                        break;
                    }
                }
            }
        }
        for (row = 0; row < nOfRows; row++) {
            coveredRows[row] = 0;
        }
    }

    /* move to step 2b */
    step2b(assignment, (distMatrix.empty() ? nullptr : distMatrix.data()),
           starMatrix.empty() ? nullptr : &starMatrix[0], newStarMatrix.empty() ? nullptr : &newStarMatrix[0],
           primeMatrix.empty() ? nullptr : &primeMatrix[0], coveredColumns.empty() ? nullptr : &coveredColumns[0],
           coveredRows.empty() ? nullptr : &coveredRows[0], nOfRows, nOfColumns, minDim);

    /* compute cost and remove invalid assignments */
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);

    /* free allocated memory */
    // free(distMatrix);
    // free(coveredColumns);
    // free(coveredRows);
    // free(starMatrix);
    // free(primeMatrix);
    // free(newStarMatrix);
}

/********************************************************/
void HungarianAlgorithm::buildassignmentvector(int* assignment, const unsigned char* starMatrix, int nOfRows,
                                               int nOfColumns)
{
    int row = 0;
    int col = 0;

    for (row = 0; row < nOfRows; row++) {
        for (col = 0; col < nOfColumns; col++) {
            if (starMatrix[row + nOfRows * col] == 1) {
#ifdef ONE_INDEXING
                assignment[row] = col + 1; /* MATLAB-Indexing */
#else
                assignment[row] = col;
#endif
                break;
            }
        }
    }
}

/********************************************************/
void HungarianAlgorithm::computeassignmentcost(const int* assignment, double* cost, const double* distMatrix,
                                               int nOfRows)
{
    int row = 0;
    int col = 0;

    for (row = 0; row < nOfRows; row++) {
        col = assignment[row];
        if (col >= 0) {
            *cost += distMatrix[row + nOfRows * col];
        }
    }
}

/********************************************************/
void HungarianAlgorithm::step2a(int* assignment, double* distMatrix, unsigned char* starMatrix,
                                unsigned char* newStarMatrix, unsigned char* primeMatrix, unsigned char* coveredColumns,
                                unsigned char* coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    unsigned char* starMatrixTemp = nullptr;
    unsigned char* columnEnd = nullptr;
    int col = 0;

    /* cover every column containing a starred zero */
    for (col = 0; col < nOfColumns; col++) {
        starMatrixTemp = starMatrix + static_cast<ptrdiff_t>(nOfRows * col);
        columnEnd = starMatrixTemp + nOfRows;
        while (starMatrixTemp < columnEnd) {
            if (*starMatrixTemp++ == 1) {
                coveredColumns[col] = 1;
                break;
            }
        }
    }

    /* move to step 3 */
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
           nOfColumns, minDim);
}

/********************************************************/
void HungarianAlgorithm::step2b(int* assignment, double* distMatrix, unsigned char* starMatrix,
                                unsigned char* newStarMatrix, unsigned char* primeMatrix, unsigned char* coveredColumns,
                                unsigned char* coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    int col = 0;
    int nOfCoveredColumns = 0;

    /* count covered columns */
    nOfCoveredColumns = 0;
    for (col = 0; col < nOfColumns; col++) {
        if (coveredColumns == nullptr) { // TODO(vtpl1): introduced
            break;
        }
        if (coveredColumns[col] == 1) {
            nOfCoveredColumns++;
        }
    }

    if (nOfCoveredColumns == minDim) {
        /* algorithm finished */
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    } else {
        /* move to step 3 */
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
              nOfColumns, minDim);
    }
}

/********************************************************/
void HungarianAlgorithm::step3(int* assignment, double* distMatrix, unsigned char* starMatrix,
                               unsigned char* newStarMatrix, unsigned char* primeMatrix, unsigned char* coveredColumns,
                               unsigned char* coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    unsigned char zerosFound = 0;
    int row = 0;
    int col = 0;
    int starCol = 0;

    zerosFound = 1;
    while (zerosFound == 1) {
        zerosFound = 0;
        for (col = 0; col < nOfColumns; col++) {
            if (coveredColumns[col] == 0) {
                for (row = 0; row < nOfRows; row++) {
                    if ((coveredRows[row] == 0) && (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON)) {
                        /* prime zero */
                        primeMatrix[row + nOfRows * col] = 1;

                        /* find starred zero in current row */
                        for (starCol = 0; starCol < nOfColumns; starCol++) {
                            if (starMatrix[row + nOfRows * starCol] == 1) {
                                break;
                            }
                        }

                        if (starCol == nOfColumns) /* no starred zero found */
                        {
                            /* move to step 4 */
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns,
                                  coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                        coveredRows[row] = 1;
                        coveredColumns[starCol] = 0;
                        zerosFound = 1;
                        break;
                    }
                }
            }
        }
    }

    /* move to step 5 */
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
          nOfColumns, minDim);
}

/********************************************************/
void HungarianAlgorithm::step4(int* assignment, double* distMatrix, unsigned char* starMatrix,
                               unsigned char* newStarMatrix, unsigned char* primeMatrix, unsigned char* coveredColumns,
                               unsigned char* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
    int n = 0;
    int starRow = 0;
    int starCol = 0;
    int primeRow = 0;
    int primeCol = 0;
    int nOfElements = nOfRows * nOfColumns;

    /* generate temporary copy of starMatrix */
    for (n = 0; n < nOfElements; n++) {
        newStarMatrix[n] = starMatrix[n];
    }

    /* star current zero */
    newStarMatrix[row + nOfRows * col] = 1;

    /* find starred zero in current column */
    starCol = col;
    for (starRow = 0; starRow < nOfRows; starRow++) {
        if (starMatrix[starRow + nOfRows * starCol] == 1) {
            break;
        }
    }

    while (starRow < nOfRows) {
        /* unstar the starred zero */
        newStarMatrix[starRow + nOfRows * starCol] = 0;

        /* find primed zero in current row */
        primeRow = starRow;
        for (primeCol = 0; primeCol < nOfColumns; primeCol++) {
            if (primeMatrix[primeRow + nOfRows * primeCol] == 1) {
                break;
            }
        }

        /* star the primed zero */
        newStarMatrix[primeRow + nOfRows * primeCol] = 1;

        /* find starred zero in current column */
        starCol = primeCol;
        for (starRow = 0; starRow < nOfRows; starRow++) {
            if (starMatrix[starRow + nOfRows * starCol] == 1) {
                break;
            }
        }
    }

    /* use temporary copy as new starMatrix */
    /* delete all primes, uncover all rows */
    for (n = 0; n < nOfElements; n++) {
        primeMatrix[n] = 0;
        starMatrix[n] = newStarMatrix[n];
    }
    for (n = 0; n < nOfRows; n++) {
        coveredRows[n] = 0;
    }

    /* move to step 2a */
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
           nOfColumns, minDim);
}

/********************************************************/
void HungarianAlgorithm::step5(int* assignment, double* distMatrix, unsigned char* starMatrix,
                               unsigned char* newStarMatrix, unsigned char* primeMatrix, unsigned char* coveredColumns,
                               unsigned char* coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    double h = 0;
    double value = 0;
    int row = 0;
    int col = 0;

    /* find smallest uncovered element h */
    h = DBL_MAX;
    for (row = 0; row < nOfRows; row++) {
        if (coveredRows[row] == 0) {
            for (col = 0; col < nOfColumns; col++) {
                if (coveredColumns[col] == 0) {
                    value = distMatrix[row + nOfRows * col];
                    if (value < h) {
                        h = value;
                    }
                }
            }
        }
    }

    /* add h to each covered row */
    for (row = 0; row < nOfRows; row++) {
        if (coveredRows[row] == 1) {
            for (col = 0; col < nOfColumns; col++) {
                distMatrix[row + nOfRows * col] += h;
            }
        }
    }

    /* subtract h from each uncovered column */
    for (col = 0; col < nOfColumns; col++) {
        if (coveredColumns[col] == 0) {
            for (row = 0; row < nOfRows; row++) {
                distMatrix[row + nOfRows * col] -= h;
            }
        }
    }

    /* move to step 3 */
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
          nOfColumns, minDim);
}