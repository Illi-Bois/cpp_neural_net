CXX = clang++

COMPILER_FLAG := -std=c++17 -Wall -O0 -g 
LINKER_FLAG = 

INCLUDE_FLAG = $(foreach dir,$(INCLUDE_DIR), -I$(dir))

BUILD_DIR = ./build
TEMP_DIR = $(BUILD_DIR)/temp

OUTPUT_DIR := $(BUILD_DIR) $(TEMP_DIR)

SRC_DIR = ./src
INCLUDE_DIR = ./include
INCLUDE_DIR += ./include/CPPNeuralNet

ALL_SRC_DIR := $(shell find $(SRC_DIR) -type d -print)

TARGET_EXEC = $(BUILD_DIR)/main

ENTRY_FILE = $(SRC_DIR)/main.cpp
ALL_FILES := $(shell find $(SRC_DIR) -name "*.cpp" -print)
# Strip path and only leave .cpp names
ALL_SRC_FILE_NAMES := $(foreach FILE, $(ALL_FILES), $(notdir $(FILE)))

ALL_OBJS = $(foreach NAME,$(ALL_SRC_FILE_NAMES),$(TEMP_DIR)/$(subst .cpp,.o,$(NAME)))

test:
	@echo ALL_SRC_DIR $(ALL_SRC_DIR)
	@echo ALL_FILES $(ALL_FILES)
	@echo ALL_SRC_FILE_NAMES $(ALL_SRC_FILE_NAMES)
	@echo ALL_OBJS $(ALL_OBJS)


.PHONY: run
.DELETE_ON_ERROR:
run: $(OUTPUT_DIR) $(TARGET_EXEC) 
	$(TARGET_EXEC)

# Make Output Dir if doesn't exist
$(OUTPUT_DIR): %:
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

