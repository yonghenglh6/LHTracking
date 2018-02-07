#include <stdio.h>
#include <stdlib.h>
#include "data_loader.h"
#include "lh_tracking.h"
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <map>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

using namespace std;
using namespace cv;

const unsigned char LabelColors[][3] = {{251, 144, 17},
                                        {2,   224, 17},
                                        {
                                         247, 13,  145},
                                        {206, 36,  255},
                                        {0,   78,  255}};

int main(int argn, char **arg) {

    string dataset_id("n1");
    string base_dir("/home/liuhao/workspace/1_dgvehicle/LHTracking/");
    string dataset_name = dataset_id;
    string imagelist_file = base_dir + "data/" + dataset_name + ".list";
    string detectresult_file = base_dir + "data/detect_gd_" + dataset_name;
    string track_output_file = base_dir + "track_" + dataset_name + ".txt";
    string track_image_output_directory = base_dir + "track_" + dataset_name + "/";

    if (access(track_image_output_directory.c_str(), F_OK) != 0) {
        CHECK(mkdir(track_image_output_directory.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) == 0)
        << "Can not create the image folder.";
    }
    ofstream track_output(track_output_file.c_str());
    int fps = 15;
    int skip_frames = 0;
    skip_frames = int(skip_frames / fps) * fps + 1;
    LOG(INFO) << "SKIP FRAME: " << skip_frames;
    dataloader::DataReader *data_reader = new dataloader::ImagelistDataReader(imagelist_file, skip_frames);

    map<int, vector<dataloader::DetectObject> > detect_result;
    dataloader::read_detection_result(detectresult_file, detect_result);

    deque<Mat> all_imgs;
    bool display = true;
    LHTracker *tracker = new LHTracker();
    Mat frame;
    int frame_index;
    bool stop = false;


    while (!stop) {
        frame_index = data_reader->readNextImage(frame);
        if (frame_index == -1) {
            LOG(FATAL) << "Error: can not read frames.\n";
        } else if (frame_index == -2) {
            LOG(INFO) << "Image done.";
            break;

//            delete data_reader;
//            data_reader = new dataloader::ImagelistDataReader(imagelist_file);
//            continue;
        }
//        LOG(INFO) << "FrameIndex: " << frame_index << endl;

        all_imgs.push_back(frame.clone());
        TrackingResult result;
        if ((frame_index - 1) % fps == 0) {

            vector<Rect> pos;
            vector<unsigned char> type;
            vector<unsigned long> kill_id;
            vector<dataloader::DetectObject> &det_vec = detect_result[int(frame_index)];
            for (int i = 0; i < det_vec.size(); i++) {
                pos.push_back(det_vec[i].position);
                type.push_back(det_vec[i].typeID);
            }
            unsigned long long tt, tt2;
            struct timeval start;
            gettimeofday(&start, NULL);
            tt = (start.tv_sec) * 1000000 + start.tv_usec;
//            sleep(1);
            tracker->Update(frame, frame_index, true, pos, type, result, kill_id);

            struct timeval end;
            gettimeofday(&end, NULL);
            tt2 = (end.tv_sec) * 1000000 + end.tv_usec;
            LOG(INFO) << "time:" << float((tt2 - tt) / 1000.f);

            LOG(INFO) << "FrameIndex: " << frame_index << " , Detect Size: " << det_vec.size() << " , Track Size: "
                      << (result.size() > 0 ? result[0].obj.size() : 0) << endl;
        }
        if (display) {
            for (int i = 0; i < result.size(); i++) {
                if (((result[i].frm_id - 1) % fps >= 0)) {
                    char tmp_char[512];
                    for (int j = 0; j < result[i].obj.size(); j++) {
                        int color_idx = result[i].obj[j].obj_id % 5;
                        rectangle(all_imgs[0], result[i].obj[j].loc,
                                  Scalar(LabelColors[color_idx][2],
                                         LabelColors[color_idx][1],
                                         LabelColors[color_idx][0]), 3, 8, 0);
                        sprintf(tmp_char, "%lu type-%u score-%.0f sl-%d dir-%d-%d",
                                result[i].obj[j].obj_id, result[i].obj[j].type,
                                result[i].obj[j].score, result[i].obj[j].sl,
                                result[i].obj[j].dir.up_down_dir,
                                result[i].obj[j].dir.left_right_dir);
                        cv::putText(all_imgs[0], tmp_char,
                                    cv::Point(result[i].obj[j].loc.x,
                                              result[i].obj[j].loc.y - 12),
                                    CV_FONT_HERSHEY_COMPLEX, 0.7,
                                    Scalar(LabelColors[color_idx][2],
                                           LabelColors[color_idx][1],
                                           LabelColors[color_idx][0]), 2);
                        track_output << result[i].frm_id << " " << result[i].obj[j].obj_id << " "
                                     << int(result[i].obj[j].type) << " "\
 << result[i].obj[j].loc.x << " " << result[i].obj[j].loc.y << " " << result[i].obj[j].loc.width << \
                            " " << result[i].obj[j].loc.height << "\n";
                    }
                    sprintf(tmp_char, (track_image_output_directory + "/%lu.jpg").c_str(),
                            result[i].frm_id);
                    imshow("Tracking Debug", all_imgs[0]);
                    if ((result[i].frm_id - 1) % fps == 0)
                        imwrite(tmp_char, all_imgs[0]);

                    int c = waitKey(1);
                    if ((char) c == 27) {
                        stop = true;
                    }
                }
//                if ((result[i].frm_id - 1) % fps == 0)
                all_imgs.pop_front();
            }
        }

    }
    track_output.close();
    delete tracker;
    delete data_reader;
    return EXIT_SUCCESS;
}