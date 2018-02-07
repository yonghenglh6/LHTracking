cmake_minimum_required(VERSION 3.5)
project(LHTracking)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
SET(LIB_OPENCV opencv_video opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)





INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include/")
LIST(APPEND SRC_LIST ${SRC_UTIL} "${PROJECT_SOURCE_DIR}/src/")
LIST(APPEND LIBS glog ${LIB_OPENCV})
add_executable(LHTracking main.cpp ${SRC_LIST})
TARGET_LINK_LIBRARIES(LHTracking ${LIBS})