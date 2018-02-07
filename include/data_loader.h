//
// Created by liuhao on 18-1-31.
//

#ifndef LHTRACKING_DATA_LOADER_H
#define LHTRACKING_DATA_LOADER_H

#include <iostream>
#include <map>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace dataloader {
    class DataReader {
    public:
        DataReader() {};

        virtual ~DataReader() {

        };

        virtual int readNextImage(cv::Mat &mat) {
            return 0;
        };

        virtual int readNextImage(cv::Mat &mat, int next) {
            return 0;
        };

        virtual bool isOpened() {
            return false;
        };
    };

    class VideoDataReader : public DataReader {
    public:
        VideoDataReader(std::string videofile, int SKIPED_FRAME = 1) {
            LOG(INFO) << "Reading video: " << videofile << std::endl;
            capture.open(videofile);
            CHECK(capture.isOpened()) << "Error: can not open video.";
            tot_frm_num = capture.get(CV_CAP_PROP_FRAME_COUNT);
            capture.set(CV_CAP_PROP_POS_FRAMES, SKIPED_FRAME);
            index_frame = SKIPED_FRAME;
            openned = true;
        }

        virtual ~VideoDataReader() {
            capture.release();
        }

        virtual int readNextImage(cv::Mat &mat) {
            if (index_frame > tot_frm_num - 1)
                return -2;
            if (!capture.read(mat))
                return -1;
            index_frame += 1;
            return index_frame - 1;
        }

        virtual int readNextImage(cv::Mat &mat, int next) {
            CHECK(false) << "Not support function";
        }

        virtual bool isOpened() {
            return openned;
        }

        cv::VideoCapture capture;
        int index_frame = 0;
        unsigned long tot_frm_num;
        bool openned = false;
    };

    class ImagelistDataReader : public DataReader {
    public:
        ImagelistDataReader(std::string listfile, int SKIPED_FRAME = 0) {
            read_labeled_imagelist(listfile.c_str(), imagenames);
            tot_frm_num = imagenames.size();
            index_frame = SKIPED_FRAME;
            openned = true;
        }

        void read_labeled_imagelist(const char *filename,
                                    std::vector<std::string> &imagenames) {
            std::ifstream fcin(filename);
            CHECK(fcin.is_open()) << "Imagelist file cannot be opened.";
            std::string imagename;
            while (!fcin.eof()) {
                fcin >> imagename;
                imagenames.push_back(imagename);
//                LOG(INFO) << "Imagename: " << imagename << std::endl;
            }
        }

        virtual ~ImagelistDataReader() {

        }

        virtual int readNextImage(cv::Mat &mat) {
            return readNextImage(mat, 1);
        }

        virtual int readNextImage(cv::Mat &mat, int next) {
            if (index_frame >= tot_frm_num)
                return -2;
            mat = cv::imread(imagenames[index_frame]);
            if (mat.cols <= 0)
                return -1;
            index_frame += next;
            return index_frame - next;
        }

        virtual bool isOpened() {
            return openned;
        }

        std::vector<std::string> imagenames;
        int index_frame;
        unsigned long tot_frm_num;
        bool openned = false;
    };

    typedef struct detect_object_t {
        float confidence;
        int typeID;
        cv::Rect position;

        void readFrom(std::istream &in) {
            in >> confidence >> typeID >> position.x >> position.y >> position.width
               >> position.height;
//            LOG(INFO) << " " << confidence << " " << typeID << " " << position.x << " " << position.y << " "
//                      << position.width << " " << position.height << std::endl;
        }

        void writeTo(std::ostream &out) {
            out << " " << confidence << " " << typeID << " " << position.x << " "
                << position.y << " " << position.width << " "
                << position.height;

        }
    } DetectObject;

    void read_detection_result(std::string filename,
                               std::map<int, std::vector<DetectObject> > &detect_result) {
        std::ifstream in(filename.c_str());
        if (!in)
            LOG(FATAL) << "Detection result can not be read." << std::endl;
        while (!in.eof()) {
            int frameId, objectNum;
            in >> frameId >> objectNum;
            for (int i = 0; i < objectNum; i++) {
                DetectObject detectobj;
                detectobj.readFrom(in);
                detect_result[frameId].push_back(detectobj);
            }
        }
    }
}

#endif //LHTRACKING_DATA_LOADER_H
