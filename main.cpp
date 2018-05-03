#include <stdio.h>
#include <stdlib.h>
#include "Tracker.h"
#include "data_loader.h"
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <map>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <deque>

using namespace std;

using namespace cv;

const unsigned char LabelColors[][3] = {{251, 144, 17},
                                        {2,   224, 17},
                                        {247, 13,  145},
                                        {206, 36,  255},
                                        {0,   78,  255}};

int main(int argn, char **arg) {
    string dataset_name("beijing");
    bool read_image = true;
    bool display = true;
    bool skip_show = true;
    int show_time = 1;
    int fps = 6;

    std::string prefix_directory = "";
    if (argn > 1)
        dataset_name = string(arg[1]);
    if (argn > 2)
        prefix_directory = string(arg[2]);

    string base_dir("/home/liuhao/workspace/1_dgvehicle/LHTracking/");

    string detectresult_file = base_dir + "data/detect_" + dataset_name;

    int skip_frames = 0;

    skip_frames = int(skip_frames / fps) * fps + 1;
    LOG(INFO) << "SKIP FRAME: " << skip_frames;


    string imagelist_file = base_dir + "data/" + dataset_name + ".list";
    dataloader::DataReader *data_reader = new dataloader::ImagelistDataReader(imagelist_file, skip_frames,
                                                                              prefix_directory);

//    string video_file=base_dir + "data/" + dataset_name + ".mp4";
//    dataloader::DataReader *data_reader = new dataloader::VideoDataReader(video_file,skip_frames);

    string output_dir = base_dir + "/output/";
//    string detectresult_file = base_dir + "data/detect_gd_" + dataset_name;
    string track_output_file = output_dir + "track_" + dataset_name + ".txt";
    string track_image_output_directory = output_dir + "track_" + dataset_name + "/";

    if (access(track_image_output_directory.c_str(), F_OK) != 0) {
        CHECK(mkdir(track_image_output_directory.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) == 0)
        << "Can not create the image folder.";
    }
    ofstream track_output(track_output_file.c_str());


    map<int, vector<dataloader::DetectObject> > detect_result;
    dataloader::read_detection_result(detectresult_file, detect_result);

    std::deque<Mat> all_imgs;

    Tracker *tracker = Tracker::createVSDTracker(DG_TRACK_SCENE_E::DG_TRACK_SCENE_FACE, 1920, 1080);
    Mat frame;
    int frame_index = skip_frames;
    bool stop = false;

    if (!read_image)
        display = false;
    while (!stop) {
        if (read_image) {
            if (skip_show)
                frame_index = data_reader->readNextImage(frame, fps);
            else
                frame_index = data_reader->readNextImage(frame);
        } else {
            frame_index += fps;
            if (frame_index >= data_reader->getTotalFrames())
                frame_index = -2;
        }
        LOG(INFO)<<"frame_index: " << frame_index << endl;
        if (frame_index == -1) {
            LOG(FATAL) << "Error: can not read frames.\n";
            break;
        } else if (frame_index == -2) {
            LOG(INFO) << "Image done.";
            break;

//            delete data_reader;
//            data_reader = new dataloader::ImagelistDataReader(imagelist_file);
//            continue;
        }
//        LOG(INFO) << "FrameIndex: " << frame_index << endl;
        if (display)
            all_imgs.push_back(frame.clone());
        std::vector<FrameResult> result;
        if ((frame_index - 1) % fps == 0) {

//            FrameResult detectFrameResult;
//            detectFrameResult.frm_id = frame_index;
//            vector<Rect> pos;
//            vector<unsigned char> type;
            vector<unsigned long> kill_id;
            vector<dataloader::DetectObject> &det_vec = detect_result[int(frame_index)];


            std::vector<cv::Rect> det_box;
            std::vector<unsigned char> det_type;
            std::vector<float> det_score;
            for (int i = 0; i < det_vec.size(); i++) {

                det_box.push_back(det_vec[i].position);
                det_type.push_back(det_vec[i].typeID);
                det_score.push_back(det_vec[i].confidence);
//                DG_DETECT_OBJECT_S detectObject;
//                detectObject.detectId = i;
//                detectObject.location.u32X = det_vec[i].position.x;
//                detectObject.location.u32Y = det_vec[i].position.y;
//                detectObject.location.u32Width = det_vec[i].position.width;
//                detectObject.location.u32Height = det_vec[i].position.height;
//                detectObject.type = det_vec[i].typeID;
//                detectFrameResult.trackObjects.push_back(detectObject);
            }
//            tracker->Update(detectFrameResult, true, result, kill_id);


            tracker->Update(frame, frame_index, true, det_box, det_type, result, kill_id, det_score);
//            LOG(INFO) << "FrameIndex: " << frame_index << " , Detect Size: " << det_vec.size() << " , Track Size: "
//                      << (result.size() > 0 ? result[0].trackObjects.size() : 0) << endl;
        }
        if (display) {
            for (auto result_frame: result) {
                if (!skip_show || ((result_frame.frm_id - 1) % fps == 0)) {
                    char tmp_char[512];
                    for (auto track_object: result_frame.obj) {
                        unsigned long trackid = track_object.obj_id;

                        Scalar mcolor(trackid * 997 % 255, trackid * 4447 % 255,
                                      trackid * 6563 % 255);
                        int color_idx = trackid % 13;
                        rectangle(all_imgs[0], cv::Rect(track_object.loc.x, track_object.loc.y,
                                                        track_object.loc.width,
                                                        track_object.loc.height),
                                  mcolor, 3, 8, 0);

                        vector<float> &match_distance = track_object.match_distance;
                        if (match_distance.size() > 4) {
                            sprintf(tmp_char, "%lu", trackid);
//                            sprintf(tmp_char, "%lu [%.1f] %.1f %.1f %.1f %.1f %.1f %.1f id %.1f",
//                                    track_object.trackId, match_distance[0],
//                                    match_distance[1], match_distance[2], match_distance[3], match_distance[4],
//                                    match_distance[5], match_distance[6], match_distance[7]);

                            Rect vtbox(match_distance[8], match_distance[9], match_distance[10], match_distance[11]);
//                            rectangle(all_imgs[0], vtbox, mcolor, 3, 8, 0);
                        } else {
                            sprintf(tmp_char, "%lu",
                                    trackid);
//                            sprintf(tmp_char, "%lu [initial]",
//                                    track_object.trackId);
                        }
                        cv::putText(all_imgs[0], tmp_char,
                                    cv::Point(track_object.loc.x,
                                              track_object.loc.y - 12),
                                    CV_FONT_HERSHEY_COMPLEX, 0.7, mcolor, 2);
                    }
                    sprintf(tmp_char, (track_image_output_directory + "/%lu.jpg").c_str(),
                            result_frame.frm_id);
                    imshow("Tracking Debug", all_imgs[0]);
                    imwrite(tmp_char, all_imgs[0]);

                    int c = waitKey(show_time);
                    if ((char) c == 27) {
                        stop = true;
                    }
                }
                if (!skip_show || ((result_frame.frm_id - 1) % fps == 0)) all_imgs.pop_front();
            }
        }
        for (auto result_frame: result) {
            if (((result_frame.frm_id - 1) % fps == 0)) {
                for (auto track_object: result_frame.obj) {
                    track_output << result_frame.frm_id << " " << track_object.obj_id << " "
                                 << int(track_object.type) << " "\
 << track_object.loc.x << " " << track_object.loc.y << " " << track_object.loc.width << \
                            " " << track_object.loc.height << "\n";
                }
            }
        }


    }
    track_output.close();
    delete tracker;
    delete data_reader;
    return EXIT_SUCCESS;
}
