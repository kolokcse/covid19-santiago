#!/bin/sh
g++ -O3 -std=c++11 -o ./bin/main main.cpp include/sampler.h include/sampler.cpp include/Parser.h

#./bin/build.sh
#./bin/main --config input/hun/config_KSH.json