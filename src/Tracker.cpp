#include "Tracker.h"
#include "lh_tracking.h"



Tracker* Tracker::createVSDTracker(DG_TRACK_SCENE_E scene, int image_width, int image_height){
    TrackStrategyParam param;
    switch (scene) {
        case DG_TRACK_SCENE_E::DG_TRACK_SCENE_FACE:
            param = TrackStrategyParamStrage::getDefaultFaceTrackStrategyParam(image_width, image_height);
            break;
        case DG_TRACK_SCENE_E::DG_TRACK_SCENE_VEHICLE:
            param = TrackStrategyParamStrage::getDefaultVehicleTrackStrategyParam(image_width, image_height);
            break;
        default:
            LOG(FATAL) << "Unknown scene.";
    }
    return new LHTracker(param);
}