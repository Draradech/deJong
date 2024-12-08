#!/bin/bash
clang -lglut -lGL -lm -Wall -Wextra -Werror -march=x86-64-v3 -O3 -ffast-math src/dejong.c src/timer.c -o dejong
