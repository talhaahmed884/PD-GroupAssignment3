CXX      = mpicxx
CXXFLAGS = -O2 -std=c++17
SRC_DIR  = src
BIN      = mm
CORRBIN  = correctness_check

# Primary target: build mm from all .cpp files in src/
# Excludes correctness_check.cpp so it compiles cleanly
MM_SRCS = $(filter-out $(SRC_DIR)/correctness_check.cpp, $(wildcard $(SRC_DIR)/*.cpp))

all: $(BIN)

$(BIN): $(MM_SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# correctness_check does not use MPI — compile with plain g++
check-build: $(CORRBIN)

$(CORRBIN): $(SRC_DIR)/correctness_check.cpp
	g++ -O2 -std=c++17 $< -o $@

clean:
	rm -f $(BIN) $(CORRBIN)

# Fallback: if teammates deliver separate executables (mm_ser, mm_2d)
# Update SRC_SER/2D below and run `make all-separate`
SRC_SER = $(SRC_DIR)/mm_ser.cpp
SRC_2D  = $(SRC_DIR)/mm_2d.cpp

all-separate:
	g++    -O2 -std=c++17 $(SRC_SER) -o mm_ser
	$(CXX) $(CXXFLAGS)    $(SRC_2D)  -o mm_2d

.PHONY: all check-build clean all-separate
