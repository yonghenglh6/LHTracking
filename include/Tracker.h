//
// Created by 闫梓祯 on 07/04/2017.
//

#ifndef VSD_TRACKER_H
#define VSD_TRACKER_H
#define STC_TRACKER
#include <sys/time.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>
#include <deque>
#include <map>
#include <set>
using namespace std;
enum SpeedLevel {
    FAST_SPEED = 0, MED_SPEED = 1, SLOW_SPEED = 2, UNKNOWN_SPEED = 3,
};

struct Direction {
    short up_down_dir;
    short left_right_dir;
};

enum VehicleScore {
    KILL_THRESHOLD = 0, LOW_SCORE = 1, HIGH_SCORE = 2

};

//Tracking result
struct ObjResult {
    unsigned long check_still_cnt;
    unsigned long obj_id;
    cv::Rect loc;
    unsigned char type;
    float score;
    SpeedLevel sl;
    Direction dir;
};

//Output result
struct FrameResult {
    unsigned long frm_id;
    vector<ObjResult> obj;
};

typedef vector<FrameResult> TrackingResult;

//for check_still
struct Tstill_obj{
    unsigned long idx;
    unsigned long cnt;
    cv::Rect loc;
};

class Tracker {
public:
    virtual void Update(const cv::Mat &img, const unsigned long &frm_id,
                const bool &is_key_frame, const vector<cv::Rect> &det_box,
                const vector<unsigned char> &det_type, TrackingResult &result) = 0;
    virtual void Update(const cv::Mat &img, const unsigned long &frm_id,
                const bool &is_key_frame, const vector<cv::Rect> &det_box,
                const vector<unsigned char> &det_type, TrackingResult &result,
                vector<unsigned long> &kill_id) = 0;
};

#endif //VSD_TRACKER_H
