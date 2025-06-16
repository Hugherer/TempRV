cmake_minimum_required(VERSION 3.14)

find_package(Boost COMPONENTS program_options)

project(MoniTAal-bin LANGUAGES CXX)

# 设置默认构建类型为 Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 添加调试标志
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

add_executable(MoniTAal-bin main.cpp)

target_link_libraries(MoniTAal-bin PRIVATE
        MoniTAal
        ${Boost_LIBRARIES})

install(TARGETS MoniTAal-bin
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)