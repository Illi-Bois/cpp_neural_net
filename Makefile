CXX := clang++

ASSMBLE_FLAG = -c -std=c++17 -Wall -O0 -g 
LINKER_FLAG 	=

INCLUDE_FLAG = -I$(INCLUDE_DIR)
GTEST_INCLUDE_FLAG = $(foreach INC_DIR,$(GTEST_INCLUDE_DIR), -I$(INC_DIR))


BUILD_DIR := ./build

# Where all the src.o goes
TEMP_DIR := $(BUILD_DIR)/temp
# Where all the test.o goes, cannot simply be called test cuz test is already a target
TEST_TEMP_DIR := $(BUILD_DIR)/test_temp
# Where all the gtest.o goes
GTEST_TEMP_DIR := $(BUILD_DIR)/gtest_temp

# Executables
MAIN_EXEC := $(BUILD_DIR)/main
TEST_EXEC := $(BUILD_DIR)/test


# Where our src files are found
SRC_DIR := ./src
# Where our test src files are found
TEST_SRC_DIR := ./tests
# Where our headers are found
INCLUDE_DIR := ./include

# Where all the files, meaning src and header for gtests are found
GTEST_DIR := ./googletest
# Where src for gtest are found
GTEST_SRC_DIR := $(GTEST_DIR)/src
# Where headers for gtest are found
GTEST_INCLUDE_DIR := $(GTEST_DIR)/include
# seems to need this
GTEST_INCLUDE_DIR += $(GTEST_DIR)


# Main entry to MAIN
ENTRY_FILE := $(SRC_DIR)/main.cpp
# All src files, with path, excluding main
SRC_FILES := $(filter-out $(ENTRY_FILE), \
													$(shell find $(SRC_DIR) -name "*.cpp" -print))
# All src files, with path, excluding main
TEST_SRC_FILES := $(shell find $(TEST_SRC_DIR) -name "*.cpp" -print)
# All src files for gtest
GTEST_FILES := $(shell find $(GTEST_SRC_DIR) -name "*.cc" -print)

# Obj for main only
ENTRY_OBJ := $(TEMP_DIR)/main.o
# Obj for all other src files
SRC_OBJS := $(foreach FILE,$(SRC_FILES), \
													$(TEMP_DIR)/$(subst .cpp,.o,$(notdir $(FILE))))
# Obj for all other src files
TEST_SRC_OBJS := $(foreach FILE,$(TEST_SRC_FILES), \
													$(TEST_TEMP_DIR)/$(subst .cpp,.o,$(notdir $(FILE))))
# Obj for gtest
GTEST_OBJS := $(foreach FILE,$(GTEST_FILES), \
													$(GTEST_TEMP_DIR)/$(subst .cc,.o,$(notdir $(FILE))))


# TOP LEVEL TARGETS ===========================================
.PHONY: run
run: $(MAIN_EXEC)
	@echo Running Main....
	$(MAIN_EXEC)

.PHONY: test
test: $(TEST_EXEC)
	@echo Running Test....
	$(TEST_EXEC)
# End of TOP LEVEL TARGETS ====================================


# EXEC LINKAGE ================================================
$(MAIN_EXEC): $(ENTRY_OBJ) $(SRC_OBJS)
	@echo Main Exec Linking....
	$(CXX) $(LINKER_FLAG)  $(ENTRY_OBJ) $(SRC_OBJS)  -o $@

$(TEST_EXEC): $(SRC_OBJS) $(TEST_SRC_OBJS) | $(GTEST_OBJS) 
	@echo Test Exec Linking....
	$(CXX) $(LINKER_FLAG)  $(SRC_OBJS) $(TEST_SRC_OBJS) $(GTEST_OBJS)   -o $@
# End of EXEC LINKAGE =========================================


# OBJ ASSEMBLY ================================================
# Will only produce .o files
$(ENTRY_OBJ): $(TEMP_DIR)/
	@echo making main...
	$(CXX) $(ASSMBLE_FLAG) $(INCLUDE_FLAG) $(ENTRY_FILE) -o $(ENTRY_OBJ)

$(SRC_OBJS): SRC_FILE_NAME = $(subst .o,.cpp,$(notdir $@))
$(SRC_OBJS): SRC_FILE_LOC = $(filter %/$(SRC_FILE_NAME), $(SRC_FILES))
$(SRC_OBJS): | $(TEMP_DIR)/
	@echo making src.o, namely: $@
	@echo     with src.cpp: $(SRC_FILE_NAME)
	@echo which is $(SRC_FILE_LOC)
	$(CXX) $(ASSMBLE_FLAG) $(INCLUDE_FLAG) $(SRC_FILE_LOC) -o $@

$(TEST_SRC_OBJS): TEST_SRC_FILE_NAME = $(subst .o,.cpp,$(notdir $@))
$(TEST_SRC_OBJS): TEST_SRC_FILE_LOC = $(filter %/$(TEST_SRC_FILE_NAME), $(TEST_SRC_FILES))
$(TEST_SRC_OBJS): $(TEST_TEMP_DIR)/
	@echo making testsrc.o, namely: $@
	@echo     with testsrc.cpp: $(TEST_SRC_FILE_NAME)
	@echo which is $(TEST_SRC_FILE_LOC)
	$(CXX) $(ASSMBLE_FLAG) $(INCLUDE_FLAG) $(GTEST_INCLUDE_FLAG) $(TEST_SRC_FILE_LOC) -o $@


$(GTEST_OBJS): GTEST_FILE_NAME = $(subst .o,.cc,$(notdir $@))
$(GTEST_OBJS): GTEST_FILE_LOC = $(filter %/$(GTEST_FILE_NAME), $(GTEST_FILES))
$(GTEST_OBJS): | $(GTEST_TEMP_DIR)/
	@echo making gtest.o, namely: $@
	@echo     with gtest.cpp: $(GTEST_FILE_NAME)
	@echo which is $(GTEST_FILE_LOC)
	$(CXX) $(ASSMBLE_FLAG) $(INCLUDE_FLAG) $(GTEST_INCLUDE_FLAG) $(GTEST_FILE_LOC) -o $@
# End of OBJ ASSEMBLY =========================================



# Make Build Dir
$(BUILD_DIR):
	mkdir $@
# Sub-build dir
$(TEMP_DIR) $(TEST_TEMP_DIR) $(GTEST_TEMP_DIR): $(BUILD_DIR)
	mkdir $@


# Cleans everything built
#  Except for GTEST DIR
.PHONY: clean_all
clean: clean_run clean_test
	@echo Cleaned All

.PHONY: clean_test
clean_test: 
	@echo Cleaning tests....
	rm -rf $(TEST_EXEC) $(TEST_TEMP_DIR)
.PHONY: clean_run
clean_run: 
	@echo Cleaning src....
	rm -rf $(MAIN_EXEC) $(TEMP_DIR)

