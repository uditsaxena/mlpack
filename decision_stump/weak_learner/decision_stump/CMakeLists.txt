cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  decision_stump.hpp
  decision_stump_impl.cpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(dec_stu
  decision_stump_main.cpp
)
target_link_libraries(dec_stu
  mlpack
)

add_executable(dec_stu_test
  decision_stump_test.cpp
)
target_link_libraries(dec_stu_test
  mlpack
  boost_unit_test_framework
)