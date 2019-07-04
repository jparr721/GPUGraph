file(GLOB SOURCES "src/*.cc")

add_library(graph SHARED ${SOURCES})

target_include_directories(graph
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")
