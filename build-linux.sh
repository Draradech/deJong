#!/bin/bash
clang -lglut -lGL -lm -march=x86-64-v3 -O3 src/dejong.c src/timer.c -o dejong
