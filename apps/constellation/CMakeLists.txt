#
# F E T C H   C O N S T E L L A T I O N
#
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(fetch-constellation)

setup_compiler()

add_executable(constellation
               bootstrap_monitor.cpp
               bootstrap_monitor.hpp
               constellation.cpp
               constellation.hpp
               main.cpp)
target_link_libraries(constellation PRIVATE fetch-ledger fetch-miner)
target_include_directories(constellation PRIVATE ${FETCH_ROOT_DIR}/libs/python/include)
