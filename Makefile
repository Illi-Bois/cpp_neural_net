CXX := clang++

ASSMBLE_FLAG = -c -std=c++17 -Wall -O0 -g 
LINKER_FLAG 	=

INCLUDE_FLAG = -I$(INCLUDE_DIR)
GTEST_INCLUDE_FLAG = $(foreach INC_DIR,$(GTEST_INCLUDE_DIR), -I$(INC_DIR))


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
GTEST_FILES := $(wildcard $(GTEST_SRC_DIR)/*.cc)

# Obj for main only
ENTRY_OBJ := $(TEMP_DIR)/main.o
# Obj for all other src files
SRC_OBJS := $(foreach FILE,$(SRC_FILES), \
													$(TEMP_DIR)/$(subst .cpp,.o,$(notdir $(FILE))))
# Obj for all other src files
TEST_SRC_OBJS := $(foreach FILE,$(TEST_SRC_FILES), \
													$(TEMP_DIR)/$(subst .cpp,.o,$(notdir $(FILE))))
# Obj for gtest
GTEST_OBJS := $(foreach FILE,$(GTEST_FILES), \
													$(GTEST_TEMP_DIR)/$(subst .cc,.o,$(notdir $(FILE))))


TESTNAMES:
	@echo ENTRY_FILE $(ENTRY_FILE)
	@echo SRC_FILES $(SRC_FILES)
	@echo GTEST_FILES $(GTEST_FILES)
	@echo TEST_SRC_FILES $(TEST_SRC_FILES)
	@echo & echo
	@echo ENTRY_OBJ $(ENTRY_OBJ)
	@echo SRC_OBJS $(SRC_OBJS)
	@echo GTEST_OBJS $(GTEST_OBJS)
	@echo TEST_SRC_OBJS $(TEST_SRC_OBJS)

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
	$(CXX) $(LINKER_FLAG) $? -o $@

$(TEST_EXEC): $(GTEST_OBJS) $(SRC_OBJ) $(TEST_SRC_OBJS)
	@echo Test Exec Linking....
	$(CXX) $(LINKER_FLAG) $? -o $@
# End of EXEC LINKAGE =========================================


# OBJ ASSEMBLY ================================================
# Will only produce .o files
$(ENTRY_OBJ): $(TEMP_DIR)/
	@echo making main...
	$(CXX) $(ASSMBLE_FLAG) $(INCLUDE_FLAG) $(ENTRY_FILE) -o $(ENTRY_OBJ)

$(SRC_OBJS): SRC_FILE_NAME = $(subst .o,.cpp,$(notdir $@))
$(SRC_OBJS): SRC_FILE_LOC = $(filter %/$(SRC_FILE_NAME), $(SRC_FILES))
$(SRC_OBJS): $(TEMP_DIR)/
	@echo making src.o, namely: $@
	@echo     with src.cpp: $(SRC_FILE_NAME)
	@echo which is $(SRC_FILE_LOC)
	$(CXX) $(ASSMBLE_FLAG) $(INCLUDE_FLAG) $(SRC_FILE_LOC) -o $@

$(TEST_SRC_OBJS): TEST_SRC_FILE_NAME = $(subst .o,.cpp,$(notdir $@))
$(TEST_SRC_OBJS): TEST_SRC_FILE_LOC = $(filter %/$(TEST_SRC_FILE_NAME), $(TEST_SRC_FILES))
$(TEST_SRC_OBJS): $(TEMP_DIR)/
	@echo making testsrc.o, namely: $@
	@echo     with testsrc.cpp: $(TEST_SRC_FILE_NAME)
	@echo which is $(TEST_SRC_FILE_LOC)
	$(CXX) $(ASSMBLE_FLAG) $(INCLUDE_FLAG) $(GTEST_INCLUDE_FLAG) $(TEST_SRC_FILE_LOC) -o $@


$(GTEST_OBJS): GTEST_FILE_NAME = $(subst .o,.cc,$(notdir $@))
$(GTEST_OBJS): GTEST_FILE_LOC = $(filter %/$(GTEST_FILE_NAME), $(GTEST_FILES))
$(GTEST_OBJS): $(GTEST_TEMP_DIR)/
	@echo making gtest.o, namely: $@
	@echo     with gtest.cpp: $(GTEST_FILE_NAME)
	@echo which is $(GTEST_FILE_LOC)
	$(CXX) $(ASSMBLE_FLAG) $(INCLUDE_FLAG) $(GTEST_INCLUDE_FLAG) $(GTEST_FILE_LOC) -o $@

# End of OBJ ASSEMBLY =========================================



# Make Build Dir
$(BUILD_DIR):
	mkdir $@
# Sub-build dir
$(TEMP_DIR) $(GTEST_TEMP_DIR): $(BUILD_DIR)
	@echo $< $^
	mkdir $@



# Cleans everything built
#  Except for GTEST DIR
.PHONY: clean_all
clean:
	rm -rf $(MAIN_EXEC) $(TEST_EXEC) $(TEST_EXEC) $(TEMP_DIR)


