cmake_minimum_required(VERSION 2.8)

project( OpenCVTesting )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( DisplayWebcam DisplayWebcam.cpp )
add_executable( FaceDetect FaceDetect.cpp )
add_executable( DnnObjectDetect DnnObjectDetect.cpp )

target_link_libraries( DisplayWebcam ${OpenCV_LIBS} )
target_link_libraries( FaceDetect ${OpenCV_LIBS} )
target_link_libraries( DnnObjectDetect ${OpenCV_LIBS} )
