cmake_minimum_required(VERSION 3.17)
project(monitaal)

set(CMAKE_CXX_STANDARD 20)

set(EXTERNAL_INSTALL_LOCATION "${CMAKE_BINARY_DIR}/external")

OPTION(MONITAAL_BUILD_TEST "Builds tests" ON)
OPTION(MONITAAL_BUILD_BIN "Builds binary" ON)
OPTION(MONITAAL_BUILD_BENCH "Builds benchmark binary" ON)

FIND_PACKAGE(Boost REQUIRED)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_LOCATION})
file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_LOCATION}/lib)
file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_LOCATION}/include)


set(CMAKE_PREFIX_PATH ${EXTERNAL_INSTALL_LOCATION};${CMAKE_PREFIX_PATH})
include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)


include_directories(include)
add_subdirectory(src/monitaal)
#add_subdirectory(src/monitaal-bin)
# 包含pugixml
add_subdirectory(src/pugixml)
include_directories(src/pugixml/src/pugixml/include)

# 包含pardibaal
add_subdirectory(src/pardibaal)
include_directories(src/pardibaal/src/pardibaal/include)


target_include_directories(MoniTAal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MONITAAL_BUILD_BIN)
    add_subdirectory(src/monitaal-bin)
endif()

if(MONITAAL_BUILD_BENCH)
    add_subdirectory(benchmark)
endif()
    
if(MONITAAL_BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()