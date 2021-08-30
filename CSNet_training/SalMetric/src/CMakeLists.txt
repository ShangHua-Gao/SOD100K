
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
#set(SalMetric_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
#include_directories(${SalMetric_INCLUDE_DIR} ${PROJECT_BINARY_DIR})
set(src_lib sal_metric.cpp)
#add_library(salmetric ${src_lib})
#target_link_libraries(salmetric ${CUDA_LIBRARIES} ${OpenCV_LIBS})

add_executable(salmetric ${src_lib})
target_link_libraries(salmetric ${OpenCV_LIBS} ${CUDA_LIBRARIES})
set_target_properties(salmetric PROPERTIES OUTPUT_NAME "${PROJECT_BINARY_DIR}/salmetric")
