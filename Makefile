CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

BUILD := build
GPT_OBJS := $(BUILD)/tensor.o $(BUILD)/gpt.o

all: gpt tests

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/tensor.o: src/tensor.cpp src/tensor.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c src/tensor.cpp -o $@

$(BUILD)/gpt.o: src/gpt.cpp src/gpt.h src/tensor.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c src/gpt.cpp -o $@

gpt: src/main.cpp $(GPT_OBJS)
	$(CXX) $(CXXFLAGS) src/main.cpp $(GPT_OBJS) -o gpt

tests: tests/test_main.cpp $(GPT_OBJS)
	$(CXX) $(CXXFLAGS) tests/test_main.cpp $(GPT_OBJS) -o $(BUILD)/tests

check: tests
	./$(BUILD)/tests

clean:
	rm -rf $(BUILD) gpt

.PHONY: all check clean
