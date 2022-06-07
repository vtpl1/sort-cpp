#pragma once
#ifndef SortTracker_h
#define SortTracker_h
#include <opencv2/opencv.hpp>
#include <vector>

#include "KalmanTracker.h"

namespace vtpl
{
// typedef struct _TrackingBox {
//     _TrackingBox() {}
//     _TrackingBox(const cv::Rect_<float>& rect): rect(rect) {}
//     int frame{0};
//     int id{0};
//     cv::Rect_<float> rect;
//     // int miss_count{0};
// } TrackingBox;

using TrackingBox = struct _TrackingBox {
    _TrackingBox() = delete;
    //_TrackingBox(const cv::Rect& rect) : rect(rect.x, rect.y, rect.width, rect.height) {}
    _TrackingBox(const cv::Rect2f& rect) : rect(rect) {}
    int frame{0};
    int id{0};
    cv::Rect2f rect;
    // int miss_count{0};
};

class SortTracker
{
  private:
    int _max_age;
    int _min_hits;
    double _iou_threshold;
    float _rc_ext{1.0};
    bool _iou_mod{true};
    bool _show_msg; // show message
    int _frame_count;

    std::vector<KalmanTracker> _trackers;
    int _iou_mod_method{0};
    float _width_multiplier{0.5};

  public:
    SortTracker(int max_age, int min_hits, double iou_threshold, bool show_msg);
    ~SortTracker();
    std::vector<vtpl::TrackingBox> getResult(const std::vector<vtpl::TrackingBox>& tracking_box_vec, int height,
                                             int width);
    void setMaxAge(const int& max_age);
    void setMinHits(const int& min_hits);
    void setIOUThreshold(const double& iou_threshold);
    void setRCExt(const float& rc_ext);
    void setIOUMod(const bool& iou_mod);
    void setShowMsg(const bool& show_msg);
    int getMaxAge() const;
    int getMinHits() const;
    double getIOUThreshold() const;
    float getRCExt() const;
    bool getIOUMod() const;
    bool getShowMsg() const;
    void setIOUModMethod(const int& method);
    int getIOUModMethod() const;
    void setWidthMultiplier(const float& factor);
    float getWidthMultiplier() const;
};

} // namespace vtpl

#endif