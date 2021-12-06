///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <iomanip> // to format image names using setw() and setfill()
#include <iostream>
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

typedef struct TrackingBox {
    int frame;
    int id;
    cv::Rect_<float> box;
} TrackingBox;

// Computes IOU between two bounding boxes
double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

// global variables for counting
#define CNUM 20
int total_frames = 0;
double total_time = 0.0;

void TestSORT(std::string seqName, bool display);

int main()
{
    std::vector<std::string> sequences = {"PETS09-S2L1",  "TUD-Campus",    "TUD-Stadtmitte", "ETH-Bahnhof",
                                          "ETH-Sunnyday", "ETH-Pedcross2", "KITTI-13",       "KITTI-17",
                                          "ADL-Rundle-6", "ADL-Rundle-8",  "Venice-2"};
    for (auto seq : sequences)
        TestSORT(seq, false);
    // TestSORT("PETS09-S2L1", true);

    // Note: time counted here is of tracking procedure, while the running speed bottleneck is opening and parsing
    // detectionFile.
    std::cout << "Total Tracking took: " << total_time << " for " << total_frames << " frames or "
              << ((double)total_frames / (double)total_time) << " FPS" << std::endl;

    return 0;
}

void TestSORT(std::string seqName, bool display)
{
    std::cout << "Processing " << seqName << "..." << std::endl;

    // 0. randomly generate colors, only for display
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
        rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);

    std::string imgPath = "D:/Data/Track/2DMOT2015/train/" + seqName + "/img1/";

    if (display)
        if (_access(imgPath.c_str(), 0) == -1) {
            std::cerr << "Image path not found!" << std::endl;
            display = false;
        }

    // 1. read detection file
    std::ifstream detectionFile;
    std::string detFileName = "data/" + seqName + "/det.txt";
    detectionFile.open(detFileName);

    if (!detectionFile.is_open()) {
        std::cerr << "Error: can not find file " << detFileName << std::endl;
        return;
    }

    std::string detLine;
    std::istringstream ss;
    std::vector<TrackingBox> detData;
    char ch;
    float tpx, tpy, tpw, tph;

    while (getline(detectionFile, detLine)) {
        TrackingBox tb;

        ss.str(detLine);
        ss >> tb.frame >> ch >> tb.id >> ch;
        ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
        ss.str("");

        tb.box = cv::Rect_<float>(cv::Point_<float>(tpx, tpy), cv::Point_<float>(tpx + tpw, tpy + tph));
        detData.push_back(tb);
    }
    detectionFile.close();

    // 2. group detData by frame
    int maxFrame = 0;
    for (auto tb : detData) // find max frame number
    {
        if (maxFrame < tb.frame)
            maxFrame = tb.frame;
    }

    std::vector<std::vector<TrackingBox>> detFrameData;
    std::vector<TrackingBox> tempVec;
    for (int fi = 0; fi < maxFrame; fi++) {
        for (auto tb : detData)
            if (tb.frame == fi + 1) // frame num starts from 1
                tempVec.push_back(tb);
        detFrameData.push_back(tempVec);
        tempVec.clear();
    }

    // 3. update across frames
    int frame_count = 0;
    int max_age = 1;
    int min_hits = 3;
    double iouThreshold = 0.3;
    std::vector<KalmanTracker> trackers;
    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

    // variables used in the for-loop
    std::vector<cv::Rect_<float>> predictedBoxes;
    std::vector<std::vector<double>> iouMatrix;
    std::vector<int> assignment;
    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs;
    std::vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    double cycle_time = 0.0;
    int64 start_time = 0;

    // prepare result file.
    std::ofstream resultsFile;
    std::string resFileName = "output/" + seqName + ".txt";
    resultsFile.open(resFileName);

    if (!resultsFile.is_open()) {
        std::cerr << "Error: can not create file " << resFileName << std::endl;
        return;
    }

    //////////////////////////////////////////////
    // main loop
    for (int fi = 0; fi < maxFrame; fi++) {
        total_frames++;
        frame_count++;
        // cout << frame_count << endl;

        // I used to count running time using clock(), but found it seems to conflict with cv::cvWaitkey(),
        // when they both exists, clock() can not get right result. Now I use cv::getTickCount() instead.
        start_time = cv::getTickCount();

        if (trackers.size() == 0) // the first frame met
        {
            // initialize kalman trackers using first detections.
            for (unsigned int i = 0; i < detFrameData[fi].size(); i++) {
                KalmanTracker trk = KalmanTracker(detFrameData[fi][i].box);
                trackers.push_back(trk);
            }
            // output the first frame detections
            for (unsigned int id = 0; id < detFrameData[fi].size(); id++) {
                TrackingBox tb = detFrameData[fi][id];
                resultsFile << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width
                            << "," << tb.box.height << ",1,-1,-1,-1" << std::endl;
            }
            continue;
        }

        ///////////////////////////////////////
        // 3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end();) {
            cv::Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0) {
                predictedBoxes.push_back(pBox);
                it++;
            } else {
                it = trackers.erase(it);
                // cerr << "Box invalid at frame: " << frame_count << endl;
            }
        }

        ///////////////////////////////////////
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
        // dets : detFrameData[fi]
        trkNum = predictedBoxes.size();
        detNum = detFrameData[fi].size();

        iouMatrix.clear();
        iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < detNum; j++) {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(iouMatrix, assignment);

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //	there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(),
                           std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        } else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        } else
            ;

        // filter out matched with low IOU
        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i) {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            } else
                matchedPairs.push_back(cv::Point(i, assignment[i]));
        }

        ///////////////////////////////////////
        // 3.3. updating trackers

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a tracker
        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); i++) {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detFrameData[fi][detIdx].box);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections) {
            KalmanTracker tracker = KalmanTracker(detFrameData[fi][umd].box);
            trackers.push_back(tracker);
        }

        // get trackers' output
        frameTrackingResult.clear();
        for (auto it = trackers.begin(); it != trackers.end();) {
            if (((*it).m_time_since_update < 1) && ((*it).m_hit_streak >= min_hits || frame_count <= min_hits)) {
                TrackingBox res;
                res.box = (*it).get_state();
                res.id = (*it).m_id + 1;
                res.frame = frame_count;
                frameTrackingResult.push_back(res);
                it++;
            } else
                it++;

            // remove dead tracklet
            if (it != trackers.end() && (*it).m_time_since_update > max_age)
                it = trackers.erase(it);
        }

        cycle_time = (double)(cv::getTickCount() - start_time);
        total_time += cycle_time / cv::getTickFrequency();

        for (auto tb : frameTrackingResult)
            resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << ","
                        << tb.box.height << ",1,-1,-1,-1" << std::endl;

        if (display) // read image, draw results and show them
        {
            std::ostringstream oss;
            oss << imgPath << std::setw(6) << std::setfill('0') << fi + 1;
            cv::Mat img = cv::imread(oss.str() + ".jpg");
            if (img.empty())
                continue;

            for (auto tb : frameTrackingResult)
                cv::rectangle(img, tb.box, randColor[tb.id % CNUM], 2, 8, 0);
            cv::imshow(seqName, img);
            cv::waitKey(40);
        }
    }

    resultsFile.close();

    if (display)
        cv::destroyAllWindows();
}
