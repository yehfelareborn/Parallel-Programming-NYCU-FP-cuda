cmake_minimum_required(VERSION 2.8)
project( DisplayImage LANGUAGES CXX CUDA)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage BM_serial_1.cpp stereo.cu)
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

