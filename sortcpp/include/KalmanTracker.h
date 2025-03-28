///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H

#include "sortcpp_export.h"
#include <opencv2/opencv.hpp>
using StateType = cv::Rect2f;

// This class represents the internel state of individual tracked objects observed as bounding box.
class SORTCPP_EXPORT KalmanTracker
{
  public:
    KalmanTracker(StateType initRect)
    {
        init_kf(initRect);
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        kf_count++;
    }

    ~KalmanTracker() { m_history.clear(); }

    StateType predict();
    void update(StateType stateMat);

    StateType get_state() const;
    static StateType get_rect_xysr(float cx, float cy, float s, float r);

    static int kf_count;

    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;

  private:
    void init_kf(StateType stateMat);

    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::vector<StateType> m_history;
};

#endif