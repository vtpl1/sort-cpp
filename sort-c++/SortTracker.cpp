#include "SortTracker.h"
#include "Hungarian.h"
namespace vtpl
{

// Computes IOU between two bounding boxes
double getIOU(const cv::Rect_<float>& bb_test, const cv::Rect_<float>& bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON) {
        return 0.0;
    }

    return static_cast<double>(in / un);
}

// Computes modified IOU between two bounding boxes
double getModIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt, float rc_ext, int height, int width)
{
    constexpr float rc_ext_divisor = 2.0;
    int col_ext_bb_gt = int(bb_gt.width * (rc_ext / rc_ext_divisor));
    int row_ext_bb_gt = int(bb_gt.height * (2 * rc_ext));

    cv::Rect_<float> temp_bb_gt;
    temp_bb_gt.x = MAX(0, bb_gt.x - col_ext_bb_gt);
    temp_bb_gt.y = MAX(0, bb_gt.y - row_ext_bb_gt);
    int t_right = MIN(width - 1, bb_gt.x + bb_gt.width);
    int t_bottom = MIN(height - 1, bb_gt.y + bb_gt.height);
    t_right = MIN(width, t_right + col_ext_bb_gt);
    t_bottom = MIN(height, t_bottom + row_ext_bb_gt);
    temp_bb_gt.width = MAX(0, t_right - temp_bb_gt.x - 1);
    temp_bb_gt.height = MAX(0, t_bottom - temp_bb_gt.y - 1);

    int col_ext_bb_test = int(bb_test.width * (rc_ext / rc_ext_divisor));
    int row_ext_bb_test = int(bb_test.height * (2 * rc_ext));

    cv::Rect_<float> temp_bb_test;
    temp_bb_test.x = MAX(0, bb_test.x - col_ext_bb_test);
    temp_bb_test.y = MAX(0, bb_test.y - row_ext_bb_test);
    int t_right1 = MIN(width - 1, bb_test.x + bb_test.width);
    int t_bottom1 = MIN(height - 1, bb_test.y + bb_test.height);
    t_right1 = MIN(width, t_right1 + col_ext_bb_test);
    t_bottom1 = MIN(height, t_bottom1 + row_ext_bb_test);
    temp_bb_test.width = MAX(0, t_right1 - temp_bb_test.x - 1);
    temp_bb_test.height = MAX(0, t_bottom1 - temp_bb_test.y - 1);

    float in = (temp_bb_test & temp_bb_gt).area();
    float un = temp_bb_test.area() + temp_bb_gt.area() - in;

    if (un < DBL_EPSILON) {
        return 0.0;
    }

    return static_cast<double>(in / un);
}

SortTracker::SortTracker(int max_age, int min_hits, double iou_threshold, bool show_msg)
    : _max_age(max_age), _min_hits(min_hits), _iou_threshold(iou_threshold), _show_msg(show_msg), _frame_count(0)
{
}

SortTracker::~SortTracker() { _trackers.clear(); }

std::vector<vtpl::TrackingBox> SortTracker::getResult(const std::vector<vtpl::TrackingBox>& tracking_box_vec,
                                                      float rc_ext, int height, int width, bool iou_mod)
{
    if (!_show_msg) {
        std::cout << "_max_age :: " << _max_age << "; _min_hits :: " << _min_hits
                  << "; _iou_threshold :: " << _iou_threshold << std::endl;
        std::cout << "Sort_tracker getResult()..." << std::endl;
    }

    if (_trackers.empty()) // the first frame met
    {
        // initialize kalman trackers using first detections.
        int loop_cnt = 0;
        for (auto&& i : tracking_box_vec) {
            KalmanTracker trk = KalmanTracker((i).box);
            _trackers.emplace_back(trk);
            if (_show_msg) {
                std::cout << (i).frame << "," << loop_cnt++ << "," << (i).box.x << "," << (i).box.y << ","
                          << (i).box.width << "," << (i).box.height << std::endl;
            }
        }

        return std::vector<vtpl::TrackingBox>();
    }

    if (_show_msg) {
        for (auto&& i : tracking_box_vec) {
            std::cout << (i).frame << "," << (i).box.x << "," << (i).box.y << "," << (i).box.width << ","
                      << (i).box.height << std::endl;
        }
    }

    // variables used in the for-loop

    // get predicted locations from existing trackers.
    std::vector<cv::Rect_<float>> predictedBoxes;

    for (auto it = _trackers.begin(); it != _trackers.end();) {
        cv::Rect_<float> predicted_box = (*it).predict();
        if (predicted_box.x >= 0 && predicted_box.y >= 0) {
            predictedBoxes.emplace_back(predicted_box);
            it++;
        } else {
            it = _trackers.erase(it);
            if (_show_msg) {
                std::cerr << "Box invalid at frame: " << _frame_count << std::endl;
            }
        }
    }

    // associate detections to tracked object (both represented as bounding boxes)
    // dets : detFrameData[fi]
    int trkNum = predictedBoxes.size();
    int detNum = tracking_box_vec.size();

    std::vector<std::vector<double>> iouMatrix;
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            if (iou_mod) {
                iouMatrix[i][j] = 1 - getModIOU(predictedBoxes[i], tracking_box_vec.at(j).box, rc_ext, height, width);
            } else {
                iouMatrix[i][j] = 1 - getIOU(predictedBoxes[i], tracking_box_vec.at(j).box);
            }
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;
    if (!iouMatrix.empty()) {
        HungAlgo.Solve(iouMatrix, assignment);
    }

    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs;
    if (detNum > trkNum) //	there are unmatched detections
    {
        if (!_show_msg) {
            std::cout << "New Track Object ID ::" << trkNum + 1 << std::endl;
        }
        for (unsigned int n = 0; n < detNum; n++) {
            allItems.insert(n);
        }

        for (unsigned int i = 0; i < trkNum; ++i) {
            matchedItems.insert(assignment[i]);
        }

        set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(),
                       std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        if (_show_msg) {
            std::cout << "HERE 2" << std::endl;
        }
        for (unsigned int i = 0; i < trkNum; ++i) {
            if (assignment[i] == -1) { // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
            }
        }
    } else {
        ;
    }

    // filter out matched with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i) {
        if (assignment[i] == -1) { // pass over invalid values
            continue;
        }
        if (1 - iouMatrix[i][assignment[i]] < _iou_threshold) {
            if (_show_msg) {
                std::cout << "****LOW IOU :: " << _iou_threshold << " :: " << (1 - iouMatrix[i][assignment[i]])
                          << std::endl;
            }
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        } else {
            matchedPairs.emplace_back(cv::Point(i, assignment[i]));
        }
    }

    // updating trackers
    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    for (auto& matchedPair : matchedPairs) {
        int trkIdx = matchedPair.x;
        int detIdx = matchedPair.y;
        _trackers[trkIdx].update(tracking_box_vec.at(detIdx).box);
    }

    // create and initialise new trackers for unmatched detections
    for (const auto& umd : unmatchedDetections) {
        KalmanTracker tracker = KalmanTracker(tracking_box_vec.at(umd).box);
        _trackers.emplace_back(tracker);
    }

    // get trackers' output
    std::vector<vtpl::TrackingBox> frameTrackingResult;
    for (auto it = _trackers.begin(); it != _trackers.end();) {
        if (((*it).m_time_since_update < 1) && ((*it).m_hit_streak >= _min_hits || _frame_count <= _min_hits)) {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.frame = _frame_count;
            frameTrackingResult.emplace_back(res);
            it++;
        } else {
            it++;
        }

        // remove dead tracklet
        if (it != _trackers.end() && (*it).m_time_since_update > _max_age) {
            if (!_show_msg) {
				std::cout << "Deleting the trackid::::" << (*it).m_id << std::endl;
            }
            it = _trackers.erase(it);
        }
    }

    return frameTrackingResult;
}
} // namespace vtpl