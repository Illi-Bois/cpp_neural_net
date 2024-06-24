# CXX = clang++

# COMPILER_FLAG := -std=c++17 -Wall -O0 -g 
# LINKER_FLAG = 

# INCLUDE_FLAG = $(foreach dir,$(INCLUDE_DIR), -I$(dir))

# BUILD_DIR = ./build
# TEMP_DIR = $(BUILD_DIR)/temp

# OUTPUT_DIR = $(BUILD_DIR) $(TEMP_DIR) $(GTEST_TEMP_DIR)

# SRC_DIR = ./src
# INCLUDE_DIR = ./include
# INCLUDE_DIR += ./include/CPPNeuralNet

# ALL_SRC_DIR := $(shell find $(SRC_DIR) -type d -print)

# TARGET_EXEC = $(BUILD_DIR)/main

# ENTRY_FILE = $(SRC_DIR)/main.cpp
# ALL_FILES := $(shell find $(SRC_DIR) -name "*.cpp" -print)
# # Strip path and only leave .cpp names
# ALL_SRC_FILE_NAMES := $(foreach FILE, $(ALL_FILES), $(notdir $(FILE)))

# ALL_OBJS = $(foreach NAME,$(ALL_SRC_FILE_NAMES),$(TEMP_DIR)/$(subst .cpp,.o,$(NAME)))



# # GTEST------------------------------------------------------
# # GTEST_DIR = ./googletest

# # GTEST_SRC = $(wildcard $(GTEST_DIR)/src/*.cc)

# # GTEST_INCLUDE_DIR := $(GTEST_DIR)/include

# # GTEST_INCLUDE_FLAG := -I$(GTEST_INCLUDE_DIR)

# # # TODO add to output dir
# # GTEST_TEMP_DIR := $(build)/gtest_temp

# # GTEST_OBJ = $(foreach FILE,$(GTEST_SRC), $(GTEST_TEMP_DIR)/$(subst .cc,.o,$(notdir $(FILE))))

# # gtests: $(GTEST_OBJ)
# # 	@echo  $(GTEST_OBJ)
# # # $(CXX) $(COMPILER_FLAG) $(LINKER_FLAG) $(GTEST_INCLUDE_FLAG) $(GTEST_SRC) -o $(GTEST_TEMP_DIR)


# # # # name of the src file only   .cpp
# # %.o: SRC_FILE_NAME = $(subst .o,.cc,$(notdir $@))
# # # # actual directory location of said src file
# # # $(GTEST_OBJ): %.o: SRC_FILE_LOC = $(filter %/$(SRC_FILE_NAME), $(GTEST_SRC)) 
# # $(GTEST_OBJ): %.o:
# # 	@echo for $@ $(SRC_FILE_NAME)
# # #	$(CXX) $(COMPILER_FLAG) $(INCLUDE_FLAG) $(GTEST_INCLUDE_FLAG) $(GTEST_SRC) -o $(GTEST_TEMP_DIR)


# TEST_BUILD_DIR = $(BUILD_DIR)/test
# TEST_BUILD_TEMP_DIR = $(BUILD_DIR)/test_temp

# TEST_EXEC = $(TEST_BUILD_DIR)/test

# TEST_SRC_FILES := $(filter-out $(ENTRY_FILE), $(ALL_FILES))
# # PUT IN TEMP DIR SO WE CAN RESUE
# TEST_OBJS := $(foreach FILE,$(TEST_SRC_FILES),$(TEMP_DIR)/$(notdir $(subst .cpp,.o,$(FILE))))

# gtest: $(TEST_BUILD_TEMP_DIR) $(TEST_OBJS)
# 	@echo $(TEST_OBJS)


# #TODO MOVE THIS ALL INTO MAKEDIR
# $(TEST_BUILD_TEMP_DIR):
# 	mkdir $@

# .PHONY: test
# test: $(TEST_EXEC)

# # requires all the testing files and all the gtest files objs
# $(TEST_EXEC): $(TEST_OBJS) $(GTEST_OBJS)
# 	$(CXX) $(TEST_OBJS) $(GTEST_OBJS) -o $(TEST_EXEC)

# # Will rely on the old make commands, as it is the same
# # $(TEST_OBJS): %.o:

# # End of GTEST-----------------------------------------------

# test:
# 	@echo ALL_SRC_DIR $(ALL_SRC_DIR)
# 	@echo ALL_FILES $(ALL_FILES)
# 	@echo ALL_SRC_FILE_NAMES $(ALL_SRC_FILE_NAMES)
# 	@echo ALL_OBJS $(ALL_OBJS)


# .PHONY: run
# .DELETE_ON_ERROR:
# run: $(OUTPUT_DIR) $(TARGET_EXEC) 
# 	$(TARGET_EXEC)

# # Make Output Dir if doesn't exist
# $(OUTPUT_DIR): %:
# 	mkdir $@

# $(TARGET_EXEC): $(ALL_OBJS)
# 	$(CXX) $(LINKER_FLAG) $? -o $@


# # name of the src file only   .cpp
# %.o: SRC_FILE_NAME = $(subst .o,.cpp,$(notdir $@))
# # actual directory location of said src file
# %.o: SRC_FILE_LOC = $(filter %/$(SRC_FILE_NAME), $(ALL_FILES)) 
# $(ALL_OBJS): %.o:
# 	@echo Making SRC.o
# 	$(CXX) $(COMPILER_FLAG) $(INCLUDE_FLAG) -c $(SRC_FILE_LOC) -o $@





# .PHONY: clean
# clean:
# 	@echo Cleaning....
# 	rm -rf $(TARGET_EXEC) $(TEMP_DIR) $(TEST_BUILD_TEMP_DIR)


CXX := clang++

COMPILER_FLAG = -std=c++17 -Wall -O0 -g 
LINKER_FLAG 	=

INCLUDE_FLAG = -I$(INCLUDE_DIR)
GTEST_INCLUDE_FLAG = -I$(GTEST_INCLUDE_DIR)


BUILD_DIR := ./build

# Where all the src.o goes
TEMP_DIR := $(BUILD_DIR)/temp
# Where all the gtest.o goes
GTEST_TEMP_DIR := $(BUILD_DIR)/gtest_temp

# Executables
MAIN_EXEC := $(BUILD_DIR)/main
TEST_EXEC := $(BUILD_DIR)/test


# Where our src files are found
SRC_DIR := ./src
# Where our headers are found
INCLUDE_DIR := ./include

# Where all the files, meaning src and header for gtests are found
GTEST_DIR := ./googletest
# Where src for gtest are found
GTEST_SRC_DIR := $(GTEST_DIR)/src
# Where headers for gtest are found
GTEST_INCLUDE_DIR := $(GTEST_DIR)/include


# Main entry to MAIN
ENTRY_FILE := $(SRC_DIR)/main.cpp
# All src files, with path, excluding main
SRC_FILES := $(filter-out $(ENTRY_FILE), \
													$(shell find $(SRC_DIR) -name "*.cpp" -print))
# All src files for gtest
GTEST_FILES := $(wildcard $(GTEST_SRC_DIR)/*.cc)

# Obj for main only
ENTRY_OBJ := $(TEMP_DIR)/main.o
# Obj for all other src files
SRC_OBJS := $(foreach FILE,$(SRC_FILES), \
													$(TEMP_DIR)/$(subst .cpp,.o,$(notdir $(FILE))))
# Obj for gtest
GTEST_OBJS := $(foreach FILE,$(GTEST_FILES), \
													$(GTEST_TEMP_DIR)/$(subst .cc,.o,$(notdir $(FILE))))


TESTNAMES:
	@echo ENTRY_FILE $(ENTRY_FILE)
	@echo SRC_FILES $(SRC_FILES)
	@echo GTEST_FILES $(GTEST_FILES)
	@echo & echo
	@echo ENTRY_OBJ $(ENTRY_OBJ)
	@echo SRC_OBJS $(SRC_OBJS)
	@echo GTEST_OBJS $(GTEST_OBJS)

# TOP LEVEL TARGETS ===========================================
.PHONY: run
run: $(MAIN_EXEC)
	@echo Running Main....
#	$(MAIN_EXEC)

.PHONY: test
test: $(TEST_EXEC)
	@echo Running Test....
#	$(TEST_EXEC)
# End of TOP LEVEL TARGETS ====================================


# EXEC LINKAGE ================================================
$(MAIN_EXEC): $(ENTRY_OBJ) $(SRC_OBJS)
	@echo Main Exec Linking....

$(TEST_EXEC): $(GTEST_OBJS) $(SRC_OBJ)
	@echo Test Exec Linking....
# End of EXEC LINKAGE =========================================


# OBJ ASSEMBLY ================================================
$(ENTRY_OBJ): $(TEMP_DIR)
	@echo making main...
	$(CXX) $(COMPILER_FLAG) $(INCLUDE_FLAG) $(ENTRY_FILE) -o $(ENTRY_OBJ)

$(SRC_OBJS): SRC_FILE_NAME = $(subst .o,.cpp,$(notdir $@))
$(SRC_OBJS): SRC_FILE_LOC = $(filter %/$(SRC_FILE_NAME), $(SRC_FILES))
$(SRC_OBJS): $(TEMP_DIR)
	@echo making src.o, namely: $@
	@echo     with src.cpp: $(SRC_FILE_NAME)
	@echo which is $(SRC_FILE_LOC)
	$(CXX) $(COMPILER_FLAG) $(INCLUDE_FLAG) $(SRC_FILE_LOC) -o $@


$(GTEST_OBJS): GTEST_FILE_NAME = $(subst .o,.cc,$(notdir $@))
$(GTEST_OBJS): GTEST_FILE_LOC = $(filter %/$(GTEST_FILE_NAME), $(GTEST_FILES))
$(GTEST_OBJS): $(GTEST_TEMP_DIR)
	@echo making gtest.o, namely: $@
	@echo     with gtest.cpp: $(GTEST_FILE_NAME)
	@echo which is $(GTEST_FILE_LOC)
	$(CXX) $(COMPILER_FLAG) $(INCLUDE_FLAG) $(GTEST_INCLUDE_FLAG) $(GTEST_FILE_LOC) -o $@

# End of OBJ ASSEMBLY =========================================



# Make Build Dir
$(BUILD_DIR):
	mkdir $@
# Sub-build dir
$(TEMP_DIR) $(GTEST_TEMP_DIR): $(BUILD_DIR)
	mkdir $@



# Cleans everything built
.PHONY: clean_all
clean:
	rm -r $(MAIN_EXEC) $(TEST_EXEC) $(TEMP_DIR) $(GTEST_TEMP_DIR)

