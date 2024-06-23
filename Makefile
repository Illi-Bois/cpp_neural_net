# SRC_DIR = ./src
# TEST_DIR = ./test
# BUILD_DIR = ./build/debug
# TEST_BUILD_DIR = ./build/tests

# CC = g++

# ENTRY_FILE = $(SRC_DIR)/main.cpp    						# main will be used as entry only
# SRC_ALL_FILE = $(wildcard $(SRC_DIR)/*.cpp)
# SRC_FILE = $(filter-out $(ENTRY_FILE), $(SRC_ALL_FILE))   # use for test src.m All src files except main.cpp
# TEST_FILE = $(wildcard $(TEST_DIR)/*.cpp)
# CATHC2_FILE = $(TEST_DIR)/catch.hpp

# OBJ_NAME = main
# TEST_NAME = test

# # same as -I./include
# INCLUDE_PATHS = -Iinclude 

# COMPILER_FLAG =  -std=c++17 -Wall -O0 -g 
# LINKER_FLAG = 

# all:
# 	$(CC) $(COMPILER_FLAG) $(LINKER_FLAG) $(INCLUDE_PATHS) $(SDL_INCLUDE_PATHS) $(LIBRARY_PATHS) $(SRC_ALL_FILE) -o $(BUILD_DIR)/$(OBJ_NAME)

# tests: 
# 	$(CC) $(COMPILER_FLAG) $(LINKER_FLAG) $(INCLUDE_PATHS) $(SDL_INCLUDE_PATHS) $(LIBRARY_PATHS) $(TEST_FILE) $(SRC_FILE) -o $(TEST_BUILD_DIR)/$(TEST_NAME)
# 	$(TEST_BUILD_DIR)/$(TEST_NAME)

# run:	# build and run all
# 	make all
# 	$(BUILD_DIR)/$(OBJ_NAME)


# clean: 
# 	rm -r $(BUILD_DIR)/* && rm -r $(TEST_BUILD_DIR)/*






# CXX := clang++

# HEADER_DIR = include

# INCLUDE_FLAG = -I$(HEADER_DIR)

# COMPILER_FLAG =  -std=c++17 -Wall -O0 -g $(INCLUDE_FLAG)
# LINKER_FLAG =

# SRC_DIR := ./src

# SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)

# # TODO
# ALL_OBJS := main.o

# TARGET_EXEC := main

# main: $(ALL_OBJS)
# 	$(CXX) $(LINKER_FLAG) $? -o $(TARGET_EXEC)

# %.o: 
# 	$(CXX) $(COMPILER_FLAG) $(filter %$($@:.o=.cpp), $(SRC_FILES)) -o $@



CXX = clang++

COMPILER_FLAG = -std=c++17 -Wall -O0 -g 
LINKER_FLAG = 

INCLUDE_FLAG = $(foreach dir,$(INCLUDE_DIR), -I$(dir))


BUILD_DIR = ./build
TEMP_DIR = $(BUILD_DIR)/temp

SRC_DIR = ./src
INCLUDE_DIR = ./include
INCLUDE_DIR += ./include/CPPNeuralNet/Utils

ALL_DIR := $(BUILD_DIR) $(TEMP_DIR) $(SRC_DIR)

TARGET_EXEC = $(BUILD_DIR)/main


ALL_FILES = $(wildcard $(SRC_DIR)/*.cpp) 
ALL_FILES += $(SRC_DIR)/CPPNeuralNet/Utils/sanity_check.cpp

ALL_OBJS = $(TEMP_DIR)/main.o $(TEMP_DIR)/sanity_check.o 



.DELETE_ON_ERROR:
run: $(ALL_DIR) $(TARGET_EXEC) 
	$(TARGET_EXEC)

# For each directory, if doesnt exit, make it
$(ALL_DIR): %:
	mkdir $@


$(TARGET_EXEC): $(ALL_OBJS)
	$(CXX) $(LINKER_FLAG) $? -o $@


# name of the src file only   .cpp
%.o: SRC_FILE_NAME = $(subst .o,.cpp,$(notdir $@))
# actual directory location of said src file
%.o: SRC_FILE_LOC = $(filter %/$(SRC_FILE_NAME), $(ALL_FILES)) 
%.o:
	$(CXX) $(COMPILER_FLAG) $(INCLUDE_FLAG) -c $(SRC_FILE_LOC) -o $@





.PHONY: clean
clean:
	@echo Cleaning....
	rm -rf $(TARGET_EXEC) $(TEMP_DIR)

