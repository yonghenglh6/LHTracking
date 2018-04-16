//
// Created by liuhao on 18-1-31.
//

#ifndef LHTRACKING_LH_TRACKING_H
#define LHTRACKING_LH_TRACKING_H

#include "Tracker.h"

#include <sys/time.h>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include "glog/logging.h"

using cv::Mat;
using cv::Rect;
using std::vector;
using cv::Point2f;
using cv::Point2i;
using std::deque;
using std::map;
using std::pair;


class DetectObject {
public:
    unsigned long det_id;
    unsigned long frame_index;
    unsigned long type;
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
    DetectObject *detect_object = nullptr;
    float speed = 0;

    ~MatchPacket() {
        delete detect_object;
    }
};

static float rect_distance(Rect &rect1, Rect &rect2) {
    return sqrt(pow(rect1.x + rect1.width / 2.0 - rect2.x - rect2.width / 2.0, 2) +
                pow(rect1.y + rect1.height / 2.0 - rect2.y - rect2.height / 2.0, 2));
}

static Point2f rect_diff(Rect &rect1, Rect &rect2) {
    return Point2f(rect1.x + rect1.width / 2.0 - rect2.x - rect2.width / 2.0,
                   rect1.y + rect1.height / 2.0 - rect2.y - rect2.height / 2.0);
}

static inline Rect rect_move(Rect &rect1, int diff_x, int diff_y) {
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
        trk_id = 0;
        state_track = TRACKSTATE_UNINITIAL;
        velocity = Point2f(0, 0);
    }

    ~TrackObject() {
        for (auto match : match_list) {
            delete match;
        }
        frame_packet_index.clear();
        match_list.clear();
    }

    void init_detect(long object_uid, DetectObject *detect_object) {
        trk_id = object_uid;
        state_track = TRACKSTATE_INITIAL;
        auto *match_packet = new MatchPacket();
        match_packet->match_distance.push_back(0.0);
        match_packet->match_distance.push_back(detect_object->det_id);
        match_packet->state_track = state_track;
        match_packet->direction = Point2f(0, 0);
        match_packet->detect_object = detect_object;
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
            velocity = Point2f(velocity.x * 0.3 + local_velocity.x * 0.7, velocity.y * 0.3 + local_velocity.y * 0.7);


        double speed = sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
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
        frame_packet_index[detect_object->frame_index] = match_packet;
    }

    DetectObject *getLastDetectObject() {
        return match_list.back()->detect_object;
    }

    MatchPacket *getLastMatchPacket() {
        return match_list.back();
    }

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

class TrackStrategyParam {
public:
    int picture_width_, picture_height_;
    float distance_threshold_, iou_weight_, frame_weight_, pos_weight_, scale_weight_, feature_weight_, type_weight_;
    int kMaxFrameIntervalKeep, kBoardToDrop;
};

class TrackStrategyParamStrage {
public:
    static TrackStrategyParam getDefaultFaceTrackStrategyParam(int picture_width, int picture_height) {
        TrackStrategyParam param;
        param.picture_width_ = picture_width;
        param.picture_height_ = picture_height;

        param.distance_threshold_ = 2.0;

        param.iou_weight_ = 1.0;
        param.frame_weight_ = 1.5;
        param.pos_weight_ = 1.0;
        param.scale_weight_ = 0.5;
        param.feature_weight_ = 1.0;
        param.type_weight_ = 1.0;

        param.kMaxFrameIntervalKeep = 15;
        param.kBoardToDrop = 10;
        return param;
    }

    static TrackStrategyParam getDefaultVehicleTrackStrategyParam(int picture_width, int picture_height) {
        TrackStrategyParam param;
        param.picture_width_ = picture_width;
        param.picture_height_ = picture_height;

        param.distance_threshold_ = 6.0;

        param.iou_weight_ = 5.15;
        param.frame_weight_ = 2.0;
        param.pos_weight_ = 3.295;
        param.scale_weight_ = 1.635;
        param.feature_weight_ = 0.0;
        param.type_weight_ = 1.0;

        param.kMaxFrameIntervalKeep = 100;
        param.kBoardToDrop = 30;
        return param;
    }
};

class TrackStrategy {
public:
    explicit TrackStrategy(TrackSystem *track_system, TrackStrategyParam param_) {
        track_system_ = track_system;
        param = param_;
    }

    void set_picture_size(int picture_width, int picture_height) {
        param.picture_width_ = picture_width;
        param.picture_height_ = picture_height;
    }

    void Update(vector<DetectObject *> &detectobject_set, vector<unsigned long> &kill_id) {
        // get frame id
        unsigned long current_frame_index = 0;
        auto trackobject_set = track_system_->getALiveTrackObjects();
        if (detectobject_set.size() > 0) {
//            return;
            current_frame_index = detectobject_set[0]->frame_index;

//        for(DetectObject* detectObject : detectobject_set){
//            detectobject_set->inb
//        }


            vector<DistanceUnit> distance_vec;
            vector<bool> trackobject_matched(trackobject_set.size(), false);
            vector<bool> detectobject_matched(detectobject_set.size(), false);
            for (int i = 0; i < trackobject_set.size(); i++) {
                for (int j = 0; j < detectobject_set.size(); j++) {
                    distance_vec.push_back(
                            DistanceUnit(i, j, calculateDistance(trackobject_set[i], detectobject_set[j])));
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
                    TrackObject *trackObject = trackobject_set[distanceUnit.i];
                    DetectObject *detectObject = detectobject_set[distanceUnit.j];
                    float distance_threshold = param.distance_threshold_;
                    if (trackObject->state_track == TRACKSTATE_INITIAL)
                        distance_threshold = param.distance_threshold_ * 1.5;
                    if (match_distance[0] < distance_threshold) {

//                        LOG(INFO) << "feature_distance: " << match_distance[1];
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
        }
        //Detele the unaliave object
        std::set<TrackObject *> to_detele;
        for (int i = 0; i < trackobject_set.size(); i++) {
            DetectObject *detectObject = trackobject_set[i]->getLastDetectObject();
            int frame_interval = abs(current_frame_index - detectObject->frame_index);
            if (frame_interval > param.kMaxFrameIntervalKeep) {
                to_detele.insert(trackobject_set[i]);
            } else if (inboard(detectObject->location)) {
                if (frame_interval > param.kMaxFrameIntervalKeep / 2) {
                    to_detele.insert(trackobject_set[i]);
                } else if (frame_interval > 15 && trackobject_set[i]->getLastMatchPacket()->speed > 2) {
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
    bool inboard(Rect &location) {
        return location.x < param.kBoardToDrop ||
               location.x + location.width > param.picture_width_ - param.kBoardToDrop ||
               location.y < param.kBoardToDrop ||
               location.y + location.height > param.picture_height_ - param.kBoardToDrop;
    }

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
        float c_rate1 = detect_object->location.width * 1.0 / last_detect_object->location.width;
        float c_rate2 = detect_object->location.height * 1.0 / last_detect_object->location.height;
        float change_rate = c_rate1;
        if (abs(c_rate1 - 1.0) > abs(c_rate2 - 1.0))
            change_rate = c_rate2;

//        float change_rate = 1.0;
        if (change_rate < 0.5)change_rate = 0.5;
        if (change_rate > 2.0)change_rate = 2.0;
//        LOG(INFO) << "change_rate: " << change_rate << endl;
        auto diff_x = int(track_object->velocity.x * change_rate * frame_interval), diff_y = int(
                track_object->velocity.y * change_rate * frame_interval);

        Rect vt_location = last_detect_object->location;
        vt_location.height = vt_location.height * change_rate;
        vt_location.width = vt_location.width * change_rate;
        vt_location = rect_move(vt_location, diff_x, diff_y);
        Rect &dt_location = detect_object->location;
        float distance = 0;
        float iou_weight = param.iou_weight_;
        float frame_weight = param.frame_weight_;
        float pos_weight = param.pos_weight_;
        float scale_weight = param.scale_weight_;
        float feature_weight = param.feature_weight_;
        float type_weight = param.type_weight_;


        float iou_distance = 1.0f - iou(vt_location, dt_location);

        float frame_distance = frame_interval <= param.kMaxFrameIntervalKeep ? (frame_interval - 1) * 0.015f : 10;
        float max_unit = std::max(
                std::max(std::max(vt_location.width, dt_location.width), vt_location.height),
                dt_location.height);
        float mean_unit = (vt_location.width + dt_location.width + vt_location.height +
                           dt_location.height) / 4;
        float pos_distance = rect_distance(vt_location, dt_location) / max_unit;
        auto scale_distance = float(std::sqrt(
                pow(dt_location.width - vt_location.width, 2) + pow(dt_location.height - vt_location.height, 2)) /
                                    mean_unit);
//        float feature_distance =
//                cv::compareHist(last_detect_object->hist_feature, detect_object->hist_feature, 1);
        float feature_distance = 0;

        float type_distance = detect_object->type == last_detect_object->type ? 0.0f : 1.0f;
        distance = iou_weight * iou_distance + frame_weight * frame_distance +
                   pos_weight * pos_distance + scale_weight * scale_distance + feature_weight * feature_distance
                   + type_weight * type_distance;
        return vector<float>{distance, iou_distance, frame_distance, pos_distance, scale_distance, feature_distance,
                             type_distance, detect_object->det_id, vt_location.x, vt_location.y, vt_location.width,
                             vt_location.height};
    }

    TrackSystem *track_system_;
    TrackStrategyParam param;
//    int picture_width_, picture_height_;
//    float distance_threshold_, iou_weight_, frame_weight_, pos_weight_, scale_weight_, feature_weight_, type_weight_;
//    int kMaxFrameIntervalKeep, kBoardToDrop;
};


class LHTracker : public Tracker {
public:
    LHTracker(TrackStrategyParam &mparam) {
        trackSystem = new TrackSystem();
        trackStrategy = new TrackStrategy(trackSystem, mparam);
    }

    ~LHTracker() {
        delete trackSystem;
        delete trackStrategy;
    }


    int keep_bound_legal(Rect &box, int fwidth, int fheight) {
        if (box.x < 0)
            box.x = 0;
        if (box.y < 0)
            box.y = 0;
        if (box.width < 0)
            box.width = 0;
        if (box.height < 0)
            box.height = 0;
        if (box.x + box.width > fwidth) {
            if (box.x >= fwidth) {
                box.x = fwidth - 1;
                box.width = 0;
            } else {
                box.width = fwidth - box.x - 1;
            }
        }
        if (box.y + box.height > fheight) {
            if (box.y >= fheight) {
                box.y = fheight - 1;
                box.height = 0;
            } else {
                box.height = fheight - box.y - 1;
            }
        }
        return box.height * box.height;
    }

    void Update(DG_DETECT_FRAME_RESULT_S &detect_frame_result, const bool &is_key_frame,
                std::vector<DG_TRACK_FRAME_RESULT_S> &track_frame_result_set,
                std::vector<DG_U64> &finished_track_ids) {
        if (is_key_frame) {

            vector<DetectObject *> detectobject_set;
            for (int i = 0; i < detect_frame_result.trackObjects.size(); i++) {
                DG_DETECT_OBJECT_S &dg_detect_object = detect_frame_result.trackObjects[i];
                auto *detectObject = new DetectObject();
                detectObject->det_id = i;
                detectObject->type = dg_detect_object.type;
                detectObject->location = cv::Rect(dg_detect_object.location.u32X, dg_detect_object.location.u32Y,
                                                  dg_detect_object.location.u32Width,
                                                  dg_detect_object.location.u32Height);
                detectObject->frame_index = detect_frame_result.frameId;
                detectObject->det_score = dg_detect_object.score;
                detectobject_set.push_back(detectObject);
            }
            trackStrategy->Update(detectobject_set, finished_track_ids);

            track_frame_result_set.clear();
            vector<TrackObject *> aliveTrackObjects = trackSystem->getALiveTrackObjects();
            vector<TrackObject *> to_be_returned;
            for (int i = 0; i < aliveTrackObjects.size(); i++) {
                TrackObject *trackObject = aliveTrackObjects[i];
                if (trackObject->getLastDetectObject()->frame_index == detect_frame_result.frameId) {
                    to_be_returned.push_back(trackObject);
                }
            }
            if (isFirstFrame)
                last_key_frame_index = detect_frame_result.frameId - 1;
            int frame_interval = detect_frame_result.frameId - last_key_frame_index;
            for (long frame_index = last_key_frame_index + 1;
                 frame_index <= detect_frame_result.frameId; frame_index++) {
                DG_TRACK_FRAME_RESULT_S frameResult;
                frameResult.frameId = frame_index;
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
                        DG_TRACK_OBJECT_S objResult;
                        objResult.type = (unsigned char) rfirstDetectObject->type;
                        objResult.trackId = trackObject->trk_id;
                        objResult.score = trackObject->getLastDetectObject()->det_score;
                        objResult.location.u32X = (unsigned int) (rfirstDetectObject->location.x * radio_rfirst +
                                                                  rsecondDetectObject->location.x * radio_rsecond);
                        objResult.location.u32Y = (unsigned int) (rfirstDetectObject->location.y * radio_rfirst +
                                                                  rsecondDetectObject->location.y * radio_rsecond);
                        objResult.location.u32Width = (unsigned int) (
                                rfirstDetectObject->location.width * radio_rfirst +
                                rsecondDetectObject->location.width * radio_rsecond);
                        objResult.location.u32Height = (unsigned int) (
                                rfirstDetectObject->location.height * radio_rfirst +
                                rsecondDetectObject->location.height *
                                radio_rsecond);
                        MatchPacket *matchPacket = trackObject->getLastMatchPacket();
                        float speed = matchPacket->speed;
                        objResult.speed = DG_TRACK_SPEED_E::DG_TRACK_SPEED_MED;
                        if (speed > 5)
                            objResult.speed = DG_TRACK_SPEED_E::DG_TRACK_SPEED_FAST;
                        else if (speed < 1)
                            objResult.speed = DG_TRACK_SPEED_E::DG_TRACK_SPEED_SLOW;
                        objResult.direction.leftRight = matchPacket->direction.x;
                        objResult.direction.upDown = matchPacket->direction.y;
                        if (frame_index == detect_frame_result.frameId)
                            objResult.matchDistance = trackObject->getLastMatchPacket()->match_distance;
                        frameResult.trackObjects.push_back(objResult);

                    } else if (match_list_size == 1 && frame_index == detect_frame_result.frameId) {
                        DetectObject *rfirstDetectObject = trackObject->match_list[match_list_size -
                                                                                   1]->detect_object;
                        DG_TRACK_OBJECT_S objResult;
                        objResult.type = (unsigned char) rfirstDetectObject->type;
                        objResult.trackId = trackObject->trk_id;
                        objResult.score = trackObject->getLastDetectObject()->det_score;
                        objResult.location.u32X = (unsigned int) (rfirstDetectObject->location.x);
                        objResult.location.u32Y = (unsigned int) (rfirstDetectObject->location.y);
                        objResult.location.u32Width = (unsigned int) (rfirstDetectObject->location.width);
                        objResult.location.u32Height = (unsigned int) (rfirstDetectObject->location.height);
                        objResult.speed = DG_TRACK_SPEED_E::DG_TRACK_SPEED_UNKNOWN;
                        objResult.direction.leftRight = 0;
                        objResult.direction.upDown = 0;
                        frameResult.trackObjects.push_back(objResult);
                    }
                }
                track_frame_result_set.push_back(frameResult);
            }

            isFirstFrame = false;
            last_key_frame_index = detect_frame_result.frameId;
        }
    }

private:
    TrackSystem *trackSystem;
    TrackStrategy *trackStrategy;
    bool isFirstFrame = true;
    unsigned long last_key_frame_index = 0;
};


#endif //LHTRACKING_LH_TRACKING_H
