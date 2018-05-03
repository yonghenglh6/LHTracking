//
// Created by 闫梓祯 on 07/04/2017. Edit by liuhao.
//

#ifndef TRACKER_H
#define TRACKER_H

#include <sys/time.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>
#include <deque>
#include <map>
#include <set>

//using namespace std;
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
    std::vector<float> match_distance;
};

//Output result
struct FrameResult {
    unsigned long frm_id;
    std::vector<ObjResult> obj;
};

typedef std::vector<FrameResult> TrackingResult;

//for check_still
struct Tstill_obj {
    unsigned long idx;
    unsigned long cnt;
    cv::Rect loc;
};

typedef enum dgDG_TRACK_SCENE_E {
    DG_TRACK_SCENE_VEHICLE = 0, DG_TRACK_SCENE_FACE = 1,
} DG_TRACK_SCENE_E;

class Tracker {
public:

    static Tracker* createVSDTracker(DG_TRACK_SCENE_E scene, int image_width, int image_height);

    virtual void Update(const cv::Mat &img, const unsigned long &frm_id,
                        const bool &is_key_frame, const std::vector<cv::Rect> &det_box,
                        const std::vector<unsigned char> &det_type, TrackingResult &result,
                        std::vector<unsigned long> &kill_id,std::vector<float> &det_score) = 0;
};


#endif