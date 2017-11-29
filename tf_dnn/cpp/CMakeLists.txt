cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
#add_executable(example example.cpp)

add_executable(example2 loader_est.cpp)

# try the shared tensorflow library first
find_package(TensorflowCC COMPONENTS Shared)
if(TensorflowCC_Shared_FOUND)
  #target_link_libraries(example TensorflowCC::Shared)
    target_link_libraries(example2 TensorflowCC::Shared)
# fallback to the static library
else()
  find_package(TensorflowCC REQUIRED COMPONENTS Static)
  #target_link_libraries(example TensorflowCC::Static)
  target_link_libraries(example2 TensorflowCC::Static)
endif()
