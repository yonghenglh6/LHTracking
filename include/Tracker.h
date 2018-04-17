//
// Created by 闫梓祯 on 07/04/2017. Edit by liuhao.
//

#ifndef DG_TRACKER_H
#define DG_TRACKER_H

#include "dg_common.h"
#include <vector>

typedef unsigned long DG_U64;

typedef struct dgDG_DETECT_OBJECT_S {
    DG_U64 detectId;
    DG_RECT_S location;
    DG_U8 type;
    DG_F32 score;
} DG_DETECT_OBJECT_S;


typedef enum dgDG_TRACK_SPEED_E {
    DG_TRACK_SPEED_FAST = 0, DG_TRACK_SPEED_MED = 1, DG_TRACK_SPEED_SLOW = 2, DG_TRACK_SPEED_UNKNOWN = 3
} DG_TRACK_SPEED_E;

typedef struct dgDG_TRACK_DIRECTION_S {
    DG_F32 upDown;
    DG_F32 leftRight;
} DG_TRACK_DIRECTION_S;

typedef struct dgDG_TRACK_OBJECT_S {
    DG_U64 trackId;
    DG_RECT_S location;
    DG_U8 type;
    DG_F32 score;
    DG_TRACK_SPEED_E speed;
    DG_TRACK_DIRECTION_S direction;
    std::vector<DG_F32> matchDistance;
} DG_TRACK_OBJECT_S;


typedef struct dgDG_DETECT_FRAME_RESULT_S {
    DG_U64 frameId;
    std::vector<DG_DETECT_OBJECT_S> trackObjects;
} DG_DETECT_FRAME_RESULT_S;

typedef struct dgDG_TRACK_FRAME_RESULT_S {
    DG_U64 frameId;
    std::vector<DG_TRACK_OBJECT_S> trackObjects;
} DG_TRACK_FRAME_RESULT_S;


class Tracker {
public:
    virtual ~Tracker() = default;

    virtual void Update(DG_DETECT_FRAME_RESULT_S &detect_frame_result, const bool &is_key_frame,
                        std::vector<DG_TRACK_FRAME_RESULT_S> &track_frame_result_set,
                        std::vector<DG_U64> &finished_track_ids)=0;
};


typedef enum dgDG_TRACK_SCENE_E {
    DG_TRACK_SCENE_VEHICLE = 0, DG_TRACK_SCENE_FACE = 1,
} DG_TRACK_SCENE_E;

Tracker *createTracker(DG_TRACK_SCENE_E scene, DG_U32 image_width, DG_U32 image_height);

#endif //DG_TRACKER_H
