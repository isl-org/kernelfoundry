cmake_minimum_required(VERSION 3.23)
project(clops LANGUAGES CXX)

find_package(pybind11 REQUIRED)

add_subdirectory(clops/cl)
