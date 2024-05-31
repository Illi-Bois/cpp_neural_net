SRC_DIR = ./src
TEST_DIR = ./test
BUILD_DIR = ./build/debug
TEST_BUILD_DIR = ./build/tests

CC = g++

ENTRY_FILE = $(SRC_DIR)/main.cpp    						# main will be used as entry only
SRC_ALL_FILE = $(wildcard $(SRC_DIR)/*.cpp)
SRC_FILE = $(filter-out $(ENTRY_FILE), $(SRC_ALL_FILE))   # use for test src.m All src files except main.cpp
TEST_FILE = $(wildcard $(TEST_DIR)/*.cpp)
CATHC2_FILE = $(TEST_DIR)/catch.hpp

OBJ_NAME = main
TEST_NAME = test

# same as -I./include
INCLUDE_PATHS = -Iinclude 

COMPILER_FLAG =  -std=c++11 -Wall -O0 -g 
LINKER_FLAG = 

all:
	$(CC) $(COMPILER_FLAG) $(LINKER_FLAG) $(INCLUDE_PATHS) $(SDL_INCLUDE_PATHS) $(LIBRARY_PATHS) $(SRC_ALL_FILE) -o $(BUILD_DIR)/$(OBJ_NAME)

tests: 
	$(CC) $(COMPILER_FLAG) $(LINKER_FLAG) $(INCLUDE_PATHS) $(SDL_INCLUDE_PATHS) $(LIBRARY_PATHS) $(TEST_FILE) $(SRC_FILE) -o $(TEST_BUILD_DIR)/$(TEST_NAME)
	$(TEST_BUILD_DIR)/$(TEST_NAME)

run:	# build and run all
	make all
	$(BUILD_DIR)/$(OBJ_NAME)


clean: 
	rm -r $(BUILD_DIR)/* && rm -r $(TEST_BUILD_DIR)/*