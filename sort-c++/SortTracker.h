#pragma once
#ifndef SortTracker_h
#define SortTracker_h
#include "KalmanTracker.h"
#include <opencv2/opencv.hpp>
#include <vector>
namespace vtpl
{
typedef struct _TrackingBox {
    int frame{0};
    int id{0};
    cv::Rect_<float> box;
} TrackingBox;

class SortTracker
{
  private:
    int _max_age;
    int _min_hits;
    double _iou_threshold;
    bool _show_msg; // show message
    int _frame_count;

    std::vector<KalmanTracker> _trackers;
  public:
    SortTracker(int max_age, int min_hits, double iou_threshold, bool show_msg);
    std::vector<vtpl::TrackingBox> getResult(const std::vector<vtpl::TrackingBox>& tracking_box_vec, float rc_ext,
                                             int height, int width, bool iou_mod);
    ~SortTracker();
};

} // namespace vtpl

#endif