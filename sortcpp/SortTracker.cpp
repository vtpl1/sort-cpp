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
double getModIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt, float rc_ext, int height, int width, int method,
                 float width_multiplier)
{
    // std::cout << "************************************************method: " << method << std::endl;
    // std::cout << "************************************************width_multiplier: " << width_multiplier << std::endl;
    if (method == 0) { // original method
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
    } else if (method == 1) { // New method 1 = extending width-wise 0.5 and height-wise accroding to rc_ext
        constexpr float rc_ext_divisor = 2.0;
        int col_ext_bb_gt = int(bb_gt.width * 0.5);
        int row_ext_bb_gt = int(bb_gt.height * rc_ext);

        cv::Rect_<float> temp_bb_gt;
        temp_bb_gt.x = MAX(0, bb_gt.x - col_ext_bb_gt);
        temp_bb_gt.y = MAX(0, bb_gt.y - row_ext_bb_gt);
        int t_right = MIN(width - 1, bb_gt.x + bb_gt.width);
        int t_bottom = MIN(height - 1, bb_gt.y + bb_gt.height);
        t_right = MIN(width, t_right + col_ext_bb_gt);
        t_bottom = MIN(height, t_bottom + row_ext_bb_gt);
        temp_bb_gt.width = MAX(0, t_right - temp_bb_gt.x - 1);
        temp_bb_gt.height = MAX(0, t_bottom - temp_bb_gt.y - 1);

        int col_ext_bb_test = int(bb_test.width * 0.5);
        int row_ext_bb_test = int(bb_test.height * rc_ext);

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
    } else { // New method 2 = method 1 +
        constexpr float rc_ext_divisor = 2.0;
        int col_ext_bb_gt = int(bb_gt.width * width_multiplier);
        int row_ext_bb_gt = int(bb_gt.height * rc_ext);

        cv::Rect_<float> temp_bb_gt;
        int fixed_extension = 5000;

        temp_bb_gt.x = bb_gt.x - col_ext_bb_gt;
        temp_bb_gt.y = bb_gt.y - row_ext_bb_gt;
        int t_right = bb_gt.x + bb_gt.width;
        int t_bottom = bb_gt.y + bb_gt.height;

        t_right = t_right + col_ext_bb_gt;
        t_bottom = t_bottom + row_ext_bb_gt;

        temp_bb_gt.x = temp_bb_gt.x + fixed_extension;
        temp_bb_gt.y = temp_bb_gt.y + fixed_extension;

        t_right = t_right + fixed_extension;
        t_bottom = t_bottom + fixed_extension;

        temp_bb_gt.width = t_right - temp_bb_gt.x - 1;
        temp_bb_gt.height = t_bottom - temp_bb_gt.y - 1;

        int col_ext_bb_test = int(bb_test.width * width_multiplier);
        int row_ext_bb_test = int(bb_test.height * rc_ext);

        cv::Rect_<float> temp_bb_test;
        temp_bb_test.x = bb_test.x - col_ext_bb_test;
        temp_bb_test.y = bb_test.y - row_ext_bb_test;
        int t_right1 = bb_test.x + bb_test.width;
        int t_bottom1 = bb_test.y + bb_test.height;
        t_right1 = t_right1 + col_ext_bb_test;
        t_bottom1 = t_bottom1 + row_ext_bb_test;

        temp_bb_test.x = temp_bb_test.x + fixed_extension;
        temp_bb_test.y = temp_bb_test.y + fixed_extension;
        t_right1 = t_right1 + fixed_extension;
        t_bottom1 = t_bottom1 + fixed_extension;

        temp_bb_test.width = MAX(0, t_right1 - temp_bb_test.x - 1);
        temp_bb_test.height = MAX(0, t_bottom1 - temp_bb_test.y - 1);

        float in = (temp_bb_test & temp_bb_gt).area();
        float un = temp_bb_test.area() + temp_bb_gt.area() - in;
        if (un < DBL_EPSILON) {
            return 0.0;
        }
        return static_cast<double>(in / un);
    }
}

SortTracker::SortTracker(int max_age, int min_hits, double iou_threshold, bool show_msg)
    : _max_age(max_age), _min_hits(min_hits), _iou_threshold(iou_threshold), _show_msg(show_msg), _frame_count(0)
{
}

SortTracker::~SortTracker() { _trackers.clear(); }

std::vector<vtpl::TrackingBox> SortTracker::getResult(const std::vector<vtpl::TrackingBox>& tracking_box_vec,
                                                      int height, int width)
{
    if (_show_msg) {
        std::cout << "_max_age :: " << _max_age << "; _min_hits :: " << _min_hits
                  << "; _iou_threshold :: " << _iou_threshold << std::endl;
        std::cout << "Sort_tracker getResult()..." << std::endl;
    }

    // if (_trackers.empty()) // the first frame met
    // {
    //     // initialize kalman trackers using first detections.
    //     int loop_cnt = 0;
    //     for (auto&& i : tracking_box_vec) {
    //         KalmanTracker trk = KalmanTracker((i).rect);
    //         _trackers.emplace_back(trk);
    //         if (_show_msg) {
    //             std::cout << (i).frame << "," << loop_cnt++ << "," << (i).rect.x << "," << (i).rect.y << ","
    //                       << (i).rect.width << "," << (i).rect.height << std::endl;
    //         }
    //     }

    //     return std::vector<vtpl::TrackingBox>();
    // }

    if (_show_msg) {
        for (auto&& i : tracking_box_vec) {
            std::cout << (i).frame << "," << (i).rect.x << "," << (i).rect.y << "," << (i).rect.width << ","
                      << (i).rect.height << std::endl;
        }
    }

    // variables used in the for-loop

    // get predicted locations from existing trackers.
    std::vector<cv::Rect_<float>> predictedBoxes;

    // for (auto it = _trackers.begin(); it != _trackers.end();) {
    //     std::cout << "********* [" << (*it).m_id <<"][" << (*it).m_time_since_update << "][" << (*it).m_age << "]" <<
    //     std::endl;

    // }

    for (auto it = _trackers.begin(); it != _trackers.end();) {
        cv::Rect_<float> predicted_box = (*it).predict();
        if (predicted_box.x >= 0 && predicted_box.y >= 0) {
            predictedBoxes.emplace_back(predicted_box);
            it++;
        } else {
            predicted_box.x = 0;
            predicted_box.y = 0;
            predicted_box.width = 1;
            predicted_box.height = 1;
            predictedBoxes.emplace_back(predicted_box);
            it++;
        }
        // else {
        //     // int p = 0;
        //     if (_show_msg) {
        //         std::cerr << "Trackid deleted: " <<  (*it).m_id << std::endl;
        //         std::cerr << "Box invalid at frame: " << _frame_count << std::endl;
        //     }
        //     it = _trackers.erase(it);
        // }
    }

    // associate detections to tracked object (both represented as bounding boxes)
    // dets : detFrameData[fi]
    int trkNum = static_cast<int>(predictedBoxes.size());
    int detNum = static_cast<int>(tracking_box_vec.size());

    std::vector<std::vector<double>> iouMatrix;
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            if (_iou_mod) {
                iouMatrix[i][j] = 1 - getModIOU(predictedBoxes[i], tracking_box_vec.at(j).rect, _rc_ext, height, width,
                                                getIOUModMethod(), getWidthMultiplier());
            } else {
                iouMatrix[i][j] = 1 - getIOU(predictedBoxes[i], tracking_box_vec.at(j).rect);
            }
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;
    if (!iouMatrix.empty()) {
        HungarianAlgorithm::Solve(iouMatrix, assignment);
    }

    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs;
    if (detNum > trkNum) //	there are unmatched detections
    {
        if (_show_msg) {
            std::cout << "New Track Object ID ::" << trkNum + 1 << std::endl;
        }
        for (int n = 0; n < detNum; n++) {
            allItems.insert(n);
        }

        for (int i = 0; i < trkNum; ++i) {
            matchedItems.insert(assignment[i]);
        }

        set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(),
                       std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        if (_show_msg) {
            std::cout << "HERE 2" << std::endl;
        }
        for (int i = 0; i < trkNum; ++i) {
            if (assignment[i] == -1) { // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
            }
        }
    } else {
        ;
    }

    // filter out matched with low IOU
    matchedPairs.clear();
    for (int i = 0; i < trkNum; ++i) {
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
        _trackers[trkIdx].update(tracking_box_vec.at(detIdx).rect);
    }

    // create and initialise new trackers for unmatched detections
    for (const auto& umd : unmatchedDetections) {
        KalmanTracker tracker = KalmanTracker(tracking_box_vec.at(umd).rect);
        _trackers.emplace_back(tracker);
    }

    // get trackers' output
    std::vector<vtpl::TrackingBox> frameTrackingResult;
    for (auto it = _trackers.begin(); it != _trackers.end();) {
        if (_show_msg) {
            std::cout << "IT -- " << (*it).m_id << "-- " << (*it).m_time_since_update << std::endl;
        }
        if ((*it).m_time_since_update >= _max_age) {
            // remove dead tracklet
            if (_show_msg) {
                std::cout << "Deleting the trackid::::" << (*it).m_id << std::endl;
            }
            it = _trackers.erase(it);
        } else if (((*it).m_time_since_update < 1) && ((*it).m_hit_streak >= _min_hits || _frame_count <= _min_hits)) {
            TrackingBox res((*it).get_state());
            // res.rect = (*it).get_state();
            res.id = (*it).m_id;
            res.frame = _frame_count;
            // res.miss_count = (*it).m_time_since_update;
            frameTrackingResult.emplace_back(res);
            it++;
        } else {
            it++;
        }
    }

    return frameTrackingResult;
}

void SortTracker::setMaxAge(const int& max_age) { _max_age = max_age; }
void SortTracker::setMinHits(const int& min_hits) { _min_hits = min_hits; }
void SortTracker::setIOUThreshold(const double& iou_threshold) { _iou_threshold = iou_threshold; }
void SortTracker::setRCExt(const float& rc_ext) { _rc_ext = rc_ext; }
void SortTracker::setIOUMod(const bool& iou_mod) { _iou_mod = iou_mod; }
void SortTracker::setShowMsg(const bool& show_msg) { _show_msg = show_msg; }
int SortTracker::getMaxAge() const { return _max_age; }
int SortTracker::getMinHits() const { return _min_hits; }
double SortTracker::getIOUThreshold() const { return _iou_threshold; }
float SortTracker::getRCExt() const { return _rc_ext; }
bool SortTracker::getIOUMod() const { return _iou_mod; }
bool SortTracker::getShowMsg() const { return _show_msg; }
void SortTracker::setIOUModMethod(const int& method) { _iou_mod_method = method; }
int SortTracker::getIOUModMethod() const { return _iou_mod_method; }
void SortTracker::setWidthMultiplier(const float& factor) { _width_multiplier = factor; }
float SortTracker::getWidthMultiplier() const { return _width_multiplier; }
} // namespace vtpl