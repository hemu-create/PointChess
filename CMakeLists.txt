cmake_minimum_required(VERSION 3.16)
project(PointChess VERSION 2.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -fno-exceptions -fomit-frame-pointer -fno-rtti -fstrict-aliasing")

include_directories(
    ${CMAKE_SOURCE_DIR}/src/glaurung
    ${CMAKE_SOURCE_DIR}/src
)

set(POINTCHESS_SOURCES
    src/glaurung/bitboard.cpp
    src/glaurung/color.cpp
    src/glaurung/pawns.cpp
    src/glaurung/material.cpp
    src/glaurung/endgame.cpp
    src/glaurung/evaluate.cpp
    src/glaurung/main.cpp
    src/glaurung/misc.cpp
    src/glaurung/move.cpp
    src/glaurung/movegen.cpp
    src/glaurung/history.cpp
    src/glaurung/movepick.cpp
    src/glaurung/search.cpp
    src/glaurung/piece.cpp
    src/glaurung/position.cpp
    src/glaurung/square.cpp
    src/glaurung/direction.cpp
    src/glaurung/tt.cpp
    src/glaurung/value.cpp
    src/glaurung/uci.cpp
    src/glaurung/ucioption.cpp
    src/glaurung/mersenne.cpp
    src/glaurung/book.cpp
    src/glaurung/bitbase.cpp
    src/glaurung/san.cpp
    src/glaurung/benchmark.cpp
    src/nnue/nnue.cpp
    src/nnue/pointbeta.cpp
    src/eval/backend.cpp
    src/tools/selfplay_gen.cpp
)

add_executable(pointchess ${POINTCHESS_SOURCES})

find_package(Threads REQUIRED)
target_link_libraries(pointchess PRIVATE Threads::Threads m)
