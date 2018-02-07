//
// Created by liuhao on 18-1-31.
//

#ifndef LHTRACKING_LH_TRACKING_H
#define LHTRACKING_LH_TRACKING_H

#include <sys/time.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>
#include <deque>
#include <map>
#include <set>

using cv::Mat;
using cv::Rect;
using std::vector;
using cv::Point2f;
using cv::Point2i;
using std::deque;
using std::map;
using std::pair;
using std::endl;
// -------------------------------------- Binary Matching -----------------------------------------------------
#define MAX_MATCHING_NUM 101
#define INF 999999999

class BinaryMatching {
public:
    BinaryMatching() {
    };

    ~BinaryMatching() {
    };

    float match(const int n, const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM],
                int inv_link[MAX_MATCHING_NUM]);

private:
    int n_;
    int link_[MAX_MATCHING_NUM];
    float lx_[MAX_MATCHING_NUM], ly_[MAX_MATCHING_NUM],
            slack_[MAX_MATCHING_NUM];
    bool visx_[MAX_MATCHING_NUM], visy_[MAX_MATCHING_NUM];

    bool DFS(const int x, const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM]);
};
// --------------------------------------------------------------------------------------------------------------


class DetectObject {
public:
    unsigned long det_id;
    unsigned long frame_index;
    unsigned long type;
    vector<float> feature;
    float det_score;
    Rect location;
};

enum StateTrack {
    TRACKSTATE_UNINITIAL = 0, TRACKSTATE_INITIAL = 1, TRACKSTATE_NORMAL = 2, TRACKSTATE_LOST = 3, TRACKSTATE_DIE = 4,
};

class MatchPacket {
public:
    vector<float> match_distance;
    StateTrack state_track = TRACKSTATE_UNINITIAL;
    Point2f direction;
    DetectObject *detect_object;
    float speed;

    ~MatchPacket() {
        if (detect_object != NULL)
            delete detect_object;
    }
};

float rect_distance(Rect &rect1, Rect &rect2) {
    return sqrt(pow(rect1.x + rect1.width / 2 - rect2.x - rect2.width / 2, 2) +
                pow(rect1.y + rect1.height / 2 - rect2.y - rect2.height / 2, 2));
}

Point2f rect_diff(Rect &rect1, Rect &rect2) {
    return Point2f(rect1.x + rect1.width / 2 - rect2.x - rect2.width / 2,
                   rect1.y + rect1.height / 2 - rect2.y - rect2.height / 2);
}

inline Rect rect_move(Rect &rect1, int diff_x, int diff_y) {
    return Rect(rect1.x + diff_x, rect1.y + diff_y, rect1.width, rect1.height);
}

class TrackObject {
public:
    unsigned long long trk_id;
    StateTrack state_track;
    Point2f velocity;
    vector<MatchPacket *> match_list;
    std::map<int, MatchPacket *> frame_packet_index;

    TrackObject() {
        trk_id = -1;
        state_track = TRACKSTATE_UNINITIAL;
        velocity = Point2f(0, 0);
    }

    ~TrackObject() {
//        LOG(WARNING)<<"HELKKIU"<<endl;
        for (int i = 0; i < match_list.size(); i++) {
            delete match_list[i];
        }
        frame_packet_index.clear();
        match_list.clear();
    }

    void init_detect(long object_uid, DetectObject *detect_object) {
        trk_id = object_uid;
//        LOG(INFO) << "trk_id " << trk_id << std::endl;
        state_track = TRACKSTATE_INITIAL;
        MatchPacket *match_packet = new MatchPacket();
//        match_packet->match_distance.resize(1);
        match_packet->match_distance.push_back(0.0);
        match_packet->state_track = state_track;
        match_packet->direction = Point2f(0, 0);
        match_packet->detect_object = detect_object;
        detect_object->det_score = float(exp(-match_packet->match_distance[0]));
        match_list.push_back(match_packet);
        frame_packet_index[detect_object->frame_index] = match_packet;
    }

    void match_detect(DetectObject *detect_object, vector<float> match_distance) {
        state_track = TRACKSTATE_NORMAL;
        MatchPacket *match_packet = new MatchPacket();
        match_packet->match_distance = match_distance;
        match_packet->state_track = state_track;
        match_packet->detect_object = detect_object;

        DetectObject *last_detect_object = getLastDetectObject();
        int frame_interval = detect_object->frame_index - last_detect_object->frame_index;
        match_packet->direction = rect_diff(detect_object->location, last_detect_object->location);
        Point2f local_velocity = Point2f(match_packet->direction.x / frame_interval,
                                         match_packet->direction.y / frame_interval);
        if (state_track == TRACKSTATE_INITIAL)
            velocity = local_velocity;
        else
            velocity = Point2f(velocity.x * 0.1 + local_velocity.x * 0.9, velocity.y * 0.1 + local_velocity.y * 0.9);

        double speed = sqrt(velocity.x * velocity.x + velocity.y * velocity.y);

//        LOG(INFO) << "speed: " << speed << " , velocity.x: " << velocity.x << " , velocity.y: " << velocity.y << endl;
        if (speed < 0.1) {
            match_packet->direction.x = 0;
            match_packet->direction.y = 0;
        } else {
            match_packet->direction.x = velocity.x / speed;
            match_packet->direction.y = velocity.y / speed;
            if (match_packet->direction.x > 0.38)
                match_packet->direction.x = 1;
            else if (match_packet->direction.x < -0.38)
                match_packet->direction.x = -1;
            else
                match_packet->direction.x = 0;

            if (match_packet->direction.y > 0.38)
                match_packet->direction.y = 1;
            else if (match_packet->direction.y < -0.38)
                match_packet->direction.y = -1;
            else
                match_packet->direction.y = 0;
        }
        match_packet->speed = speed;

        match_list.push_back(match_packet);
        detect_object->det_score = float(exp(-match_packet->match_distance[0]));
        frame_packet_index[detect_object->frame_index] = match_packet;
    }

    DetectObject *getLastDetectObject() {
        return match_list.back()->detect_object;
    }

    MatchPacket *getLastMatchPacket() {
        return match_list.back();
    }
//    Rect loc;
//    unsigned char type;
//    float score;
//    SpeedLevel sl;
//    Direction dir;
};

class TrackFrame {
    unsigned long frame_id;
    bool is_key;
    int img_w, img_h;
//    vector<Mat> img_pyr;
    Mat image;
    vector<TrackObject *> track_objects;
};

class TrackSystem {
public:
    TrackSystem() {

    }

    void createTrackObject(DetectObject *detectobj) {
        auto *track_object = new TrackObject();
        track_object->init_detect(generate_trackobject_uid(), detectobj);
        alive_track_objects.push_back(track_object);
    }

    void match(TrackObject *track_object, DetectObject *detect_object, vector<float> match_distance) {
        track_object->match_detect(detect_object, std::move(match_distance));
    }

    vector<TrackObject *> &getALiveTrackObjects() {
        return alive_track_objects;
    }

    void inactivateTrackObjects(std::set<TrackObject *> trackObjectSet) {
        for (auto it = alive_track_objects.begin(); it != alive_track_objects.end();) {
            if (trackObjectSet.find(*it) != trackObjectSet.end()) {
                delete (*it);
                alive_track_objects.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    inline long generate_trackobject_uid() {
        long uid = trackobject_next_uid;
        trackobject_next_uid = (trackobject_next_uid + 1) % kMaxTrackObjectUID;
        return uid;
    }

    vector<TrackObject *> alive_track_objects;
    deque<TrackFrame *> frame_set;
    long frame_index_base = 0;

    long trackobject_next_uid = 0;
    const long kMaxTrackObjectUID = 9999999;
};


struct DistanceUnit {
    int i;
    int j;
    vector<float> distance;

    DistanceUnit(int _i, int _j, vector<float> _distance) {
        i = _i;
        j = _j;
        distance = _distance;
    }

    bool operator<(const DistanceUnit &rhs) const { return this->distance[0] > rhs.distance[0]; }

    bool operator>(const DistanceUnit &rhs) const { return this->distance[0] < rhs.distance[0]; }
};

class TrackStrategy {
public:
    explicit TrackStrategy(TrackSystem *track_system) {
        track_system_ = track_system;
        picture_width_ = 640;
        picture_height_ = 480;

        distance_threshold_ = 5.53;

        iou_weight_ = 4.84444;
        frame_weight_ = 2.0;
        pos_weight_ = 7.487;
        scale_weight_ = 3.00;
        feature_weight_ = 1.0;
        type_weight_ = 1.0;

//        distance_threshold_ = 1.5;
//
//        iou_weight_ = 1.0;
//        frame_weight_ = 1.0;
//        pos_weight_ = 1.0;
//        scale_weight_ = 0.5;
//        feature_weight_ = 1.0;
//        type_weight_ = 1.0;

        kMaxFrameIntervalKeep = 100;
        kBoardToDrop = 10;
    }

    void set_picture_size(int picture_width, int picture_height) {
        picture_width_ = picture_width;
        picture_height_ = picture_height;
    }

    void Update(vector<DetectObject *> &detectobject_set, vector<unsigned long> &kill_id) {
        // get frame id
        unsigned long current_frame_index = 0;
        if (detectobject_set.size() == 0)
            return;
        current_frame_index = detectobject_set[0]->frame_index;


        auto trackobject_set = track_system_->getALiveTrackObjects();
        vector<DistanceUnit> distance_vec;
        vector<bool> trackobject_matched(trackobject_set.size(), false);
        vector<bool> detectobject_matched(detectobject_set.size(), false);
        for (int i = 0; i < trackobject_set.size(); i++) {
            for (int j = 0; j < detectobject_set.size(); j++) {
                distance_vec.push_back(DistanceUnit(i, j, calculateDistance(trackobject_set[i], detectobject_set[j])));
            }
        }
        if (distance_vec.size() > 0) {
            std::sort(distance_vec.begin(), distance_vec.end(), std::greater<DistanceUnit>());
        }
//        for (int i = 0; i < distance_vec.size(); i++) {
//            std::cout << distance_vec[i].distance[0] << std::endl;
//        }
        //Matched
        for (int index = 0; index < distance_vec.size(); index++) {
            DistanceUnit &distanceUnit = distance_vec[index];

            vector<float> &match_distance = distanceUnit.distance;
            if (!trackobject_matched[distanceUnit.i] && !detectobject_matched[distanceUnit.j]) {
                if (match_distance[0] < distance_threshold_) {
                    TrackObject *trackObject = trackobject_set[distanceUnit.i];
                    DetectObject *detectObject = detectobject_set[distanceUnit.j];
                    trackobject_matched[distanceUnit.i] = true;
                    detectobject_matched[distanceUnit.j] = true;
                    track_system_->match(trackObject, detectObject, match_distance);
                }
            }
        }


        //No Matched, create new object
        for (int i = 0; i < detectobject_set.size(); i++) {
            if (!detectobject_matched[i]) {
                track_system_->createTrackObject(detectobject_set[i]);
            }
        }

        //Detele the unaliave object
        std::set<TrackObject *> to_detele;
        for (int i = 0; i < trackobject_set.size(); i++) {
            DetectObject *detectObject = trackobject_set[i]->getLastDetectObject();
            if (abs(current_frame_index - detectObject->frame_index) > kMaxFrameIntervalKeep) {
                to_detele.insert(trackobject_set[i]);
            } else if (abs(current_frame_index - detectObject->frame_index) > kMaxFrameIntervalKeep / 2) {
                if (detectObject->location.x < kBoardToDrop ||
                    detectObject->location.x + detectObject->location.width > picture_width_ - kBoardToDrop ||
                    detectObject->location.y < kBoardToDrop ||
                    detectObject->location.y + detectObject->location.height > picture_height_ - kBoardToDrop) {
                    to_detele.insert(trackobject_set[i]);
                }
            }
        }


        kill_id.clear();
        for (auto it = to_detele.begin(); it != to_detele.end(); it++)
            kill_id.push_back((*it)->trk_id);
        track_system_->inactivateTrackObjects(to_detele);
//        LOG(INFO)<<"track_system_->getALiveTrackObjects().size()"<<track_system_->getALiveTrackObjects().size()<<endl;
    }

private:
    float iou(Rect r1, Rect r2) {
        float overlop_w = std::min(r1.x + r1.width, r2.x + r2.width) - std::max(r1.x, r2.x);
        float overlop_h = std::min(r1.y + r1.height, r2.y + r2.height) - std::max(r1.y, r2.y);
        if (overlop_w <= 0 || overlop_h <= 0)
            return 0.0;
        float overlop_area = overlop_w * overlop_h;
        return overlop_area / (r1.width * r1.height + r2.width * r2.height - overlop_area);
    }

    vector<float> calculateDistance(TrackObject *track_object, DetectObject *detect_object) {
        DetectObject *last_detect_object = track_object->getLastDetectObject();
        long frame_interval = detect_object->frame_index - last_detect_object->frame_index;
        auto diff_x = int(track_object->velocity.x * frame_interval), diff_y = int(
                track_object->velocity.y * frame_interval);
        Rect vt_location = rect_move(last_detect_object->location, diff_x, diff_y);
        Rect &dt_location = detect_object->location;
        float distance = 0;
        float iou_weight = iou_weight_;
        float frame_weight = frame_weight_;
        float pos_weight = pos_weight_;
        float scale_weight = scale_weight_;
        float feature_weight = feature_weight_;
        float type_weight = type_weight_;
        if (track_object->state_track == TRACKSTATE_INITIAL) {
            pos_weight /= 2;
            iou_weight /= 2;
        }

        float iou_distance = 1.0f - iou(vt_location, dt_location);
        float frame_distance = frame_interval <= kMaxFrameIntervalKeep ? (frame_interval - 1) * 0.015f : 10;
        float max_unit = std::max(
                std::max(std::max(vt_location.width, dt_location.width), vt_location.height),
                dt_location.height);
        float mean_unit = (vt_location.width + dt_location.width + vt_location.height +
                           dt_location.height) / 4;
        float pos_distance = rect_distance(vt_location, dt_location) / max_unit;
        auto scale_distance = float(std::sqrt(
                pow(dt_location.width - vt_location.width, 2) + pow(dt_location.height - vt_location.height, 2)) /
                                    mean_unit);
        float feature_distance = 0;
        float type_distance = detect_object->type == last_detect_object->type ? 0.0f : 1.0f;
        distance = iou_weight * iou_distance + frame_weight * frame_distance +
                   pos_weight * pos_distance + scale_weight * scale_distance + feature_weight * feature_distance
                   + type_weight * type_distance;
        return vector<float>{distance, iou_distance, frame_distance, pos_distance, scale_distance, feature_distance,
                             type_distance};
    }

    TrackSystem *track_system_;
    int picture_width_, picture_height_;
    float distance_threshold_, iou_weight_, frame_weight_, pos_weight_, scale_weight_, feature_weight_, type_weight_;
    int kMaxFrameIntervalKeep, kBoardToDrop;
};

enum SpeedLevel {
    FAST_SPEED = 0, MED_SPEED = 1, SLOW_SPEED = 2, UNKNOWN_SPEED = 3,
};

struct Direction {
    short up_down_dir;
    short left_right_dir;
};
struct ObjResult {
    unsigned long obj_id;
    Rect loc;
    unsigned char type;
    float score;
    SpeedLevel sl;
    Direction dir;
};
struct FrameResult {
    unsigned long frm_id;
    vector<ObjResult> obj;
};
typedef vector<FrameResult> TrackingResult;

class LHTracker {
public:
    LHTracker() {
        trackSystem = new TrackSystem();
        trackStrategy = new TrackStrategy(trackSystem);
    }

    ~LHTracker() {
        delete trackSystem;
        delete trackStrategy;
    }

    void Update(const Mat &img, const unsigned long &frm_id,
                const bool &is_key_frame, const vector<Rect> &det_box,
                const vector<unsigned char> &det_type, TrackingResult &result, vector<unsigned long> &kill_id) {
        if (is_key_frame) {

            vector<DetectObject *> detectobject_set;
            for (int i = 0; i < det_box.size(); i++) {
                DetectObject *detectObject = new DetectObject();
                detectObject->type = det_type[i];
                detectObject->location = det_box[i];
                detectObject->frame_index = frm_id;
                detectobject_set.push_back(detectObject);
            }
            trackStrategy->set_picture_size(img.cols, img.rows);
            trackStrategy->Update(detectobject_set, kill_id);
//            if (!isFirstFrame) {
            result.clear();
            vector<TrackObject *> aliveTrackObjects = trackSystem->getALiveTrackObjects();
            vector<TrackObject *> to_be_returned;
            for (int i = 0; i < aliveTrackObjects.size(); i++) {
                TrackObject *trackObject = aliveTrackObjects[i];
                if (trackObject->getLastDetectObject()->frame_index == frm_id) {
                    to_be_returned.push_back(trackObject);
                }
            }
            if (isFirstFrame)
                last_key_frame_index = frm_id - 1;
            int frame_interval = frm_id - last_key_frame_index;
            for (long frame_index = last_key_frame_index + 1;
                 frame_index <= frm_id; frame_index++) {
                FrameResult frameResult;
                frameResult.frm_id = frame_index;
                float radio_rfirst = (frame_index - last_key_frame_index) * 1.0 / frame_interval;
                float radio_rsecond = 1.0 - radio_rfirst;
                for (int i = 0; i < to_be_returned.size(); i++) {
                    TrackObject *trackObject = to_be_returned[i];
                    int match_list_size = trackObject->match_list.size();
                    if (match_list_size > 1) {
                        DetectObject *rsecondDetectObject = trackObject->match_list[match_list_size -
                                                                                    2]->detect_object;
                        DetectObject *rfirstDetectObject = trackObject->match_list[match_list_size -
                                                                                   1]->detect_object;
                        if (rsecondDetectObject->frame_index == last_key_frame_index) {
                            ObjResult objResult;
                            objResult.type = (unsigned char) rfirstDetectObject->type;
                            objResult.obj_id = trackObject->trk_id;
//                                LOG(INFO) << "Object id:" << objResult.obj_id << std::endl;
                            objResult.score = trackObject->getLastDetectObject()->det_score;
                            objResult.loc.x = int(rfirstDetectObject->location.x * radio_rfirst +
                                                  rsecondDetectObject->location.x * radio_rsecond);
                            objResult.loc.y = int(rfirstDetectObject->location.y * radio_rfirst +
                                                  rsecondDetectObject->location.y * radio_rsecond);
                            objResult.loc.width = int(rfirstDetectObject->location.width * radio_rfirst +
                                                      rsecondDetectObject->location.width * radio_rsecond);
                            objResult.loc.height = int(rfirstDetectObject->location.height * radio_rfirst +
                                                       rsecondDetectObject->location.height * radio_rsecond);
                            MatchPacket *matchPacket = trackObject->getLastMatchPacket();
                            float speed = matchPacket->speed;
                            objResult.sl = MED_SPEED;
//                                LOG(INFO) << "speed: " << speed << endl;
                            if (speed > 5)
                                objResult.sl = FAST_SPEED;
                            else if (speed < 1)
                                objResult.sl = SLOW_SPEED;
                            objResult.dir.left_right_dir = matchPacket->direction.x;
                            objResult.dir.up_down_dir = matchPacket->direction.y;
                            frameResult.obj.push_back(objResult);
                        }
                    } else if (match_list_size == 1 && frame_index==frm_id){
                        DetectObject *rfirstDetectObject = trackObject->match_list[match_list_size -
                                                                                   1]->detect_object;
                        ObjResult objResult;
                        objResult.type = (unsigned char) rfirstDetectObject->type;
                        objResult.obj_id = trackObject->trk_id;
                        objResult.score = trackObject->getLastDetectObject()->det_score;
                        objResult.loc.x = int(rfirstDetectObject->location.x);
                        objResult.loc.y = int(rfirstDetectObject->location.y);
                        objResult.loc.width = int(rfirstDetectObject->location.width);
                        objResult.loc.height = int(rfirstDetectObject->location.height);
                        objResult.sl = UNKNOWN_SPEED;
                        objResult.dir.left_right_dir = 0;
                        objResult.dir.up_down_dir = 0;
                        frameResult.obj.push_back(objResult);
                    }
                }
                result.push_back(frameResult);
            }
//            }
            isFirstFrame = false;
            last_key_frame_index = frm_id;
        }
    }

private:
    TrackSystem *trackSystem;
    TrackStrategy *trackStrategy;
    bool isFirstFrame = true;
    unsigned long last_key_frame_index = 0;
};

#endif //LHTRACKING_LH_TRACKING_H
