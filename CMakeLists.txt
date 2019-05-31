# CMake file for GigaRender project
# Author: Shane Yuan
# E-mail: yuanxy92@gmail.com
#

##
##                     _oo0oo_
##                    o8888888o
##                    88" . "88
##                    (| -_- |)
##                    0\  =  /0
##                  ___/`---'\___
##                .' \\|     |// '.
##               / \\|||  :  |||// \
##              / _||||| -:- |||||- \
##             |   | \\\  -  /// |   |
##             | \_|  ''\---/''  |_/ |
##             \  .-\__  '-'  ___/-. /
##           ___'. .'  /--.--\  `. .'___
##        ."" '<  `.___\_<|>_/___.' >' "".
##       | | :  `- \`.;`\ _ /`;.`/ - ` : | |
##       \  \ `_.   \_ __\ /__ _/   .-` /  /
##   =====`-.____`.___ \_____/___.-`___.-'=====
##                     `=---='
##
##             佛祖保佑         永无BUG
##               Buddha Bless No Bug
##

message( 
"##
##                     _oo0oo_
##                    o8888888o
##                    88\" . \"88
##                    (| -_- |)
##                    0\\  =  /0
##                  ___/`---'\\___
##                .' \\\\|     |// '.
##               / \\\\|||  :  |||// \\
##              / _||||| -:- |||||- \\
##             |   | \\\\\\  -  /// |   |
##             | \\_|  ''\\---/''  |_/ |
##             \\  .-\\__  '-'  ___/-. /
##           ___'. .'  /--.--\\  `. .'___
##        .\"\" '<  `.___\\_<|>_/___.' >' \"\".
##       | | :  `- \\`.;`\\ _ /`;.`/ - ` : | |
##       \\  \\ `_.   \\_ __\\ /__ _/   .-` /  /
##   =====`-.____`.___ \\_____/___.-`___.-'=====
##                     `=---='
##
##             佛祖保佑         永无BUG
##               Buddha Bless No Bug
##
"
)

cmake_minimum_required (VERSION 3.10)
set(name "VideoStitcher")
project ( ${name} )

# set output path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin/Debug)    
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin/Debug)    
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin/Release)  

# cmake module
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# C++ version
set (CMAKE_CXX_STANDARD 11)

# C++ library postfix
set (CMAKE_DEBUG_POSTFIX "_d")

#cuda
find_package(CUDA)
set(CUDA_PROPAGATE_HOST_FLAGS on)
set(CUDA_USE_STATIC_CUDA_RUNTIME off)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
#    -O3 
	-gencode=arch=compute_61,code=\"sm_61,compute_61\"
)

#opencv 
if ( MSVC )
	set(OpenCV_DIR "E:/libs/OpenCV/x64/vc14/lib")
	set(OpenCV_CONFIG_PATH "E:/libs/OpenCV")
	set(WIN_HEADER_PATH "${PROJECT_SOURCE_DIR}/winheadfile")
    include_directories(${WIN_HEADER_PATH})
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen3
if (MSVC)
	set(EIGEN_INCLUDE_DIR "E:/libs/Eigen/include/eigen3")
elseif ( UNIX )
		set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
endif()
include_directories(${EIGEN_INCLUDE_DIR})

include_directories(${WIN_HEADER_PATH})

cuda_add_executable(${name}
  src/stitching.hpp
  src/precomp.hpp
  src/util_log.hpp
  src/stitcher.cpp
  main.cpp
)

target_link_libraries(${name}
	# opencv libs
	${OpenCV_LIBS}
)



