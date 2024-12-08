@echo off

set incs=-I external\freeglut\include -I external\pthreads4w\include
set libs=-L external\freeglut\lib -L external\pthreads4w\lib -lpthreadVC3
set defs=-DNDEBUG -D_USE_MATH_DEFINES -DGLUT_DISABLE_ATEXIT_HACK

@echo on
clang.exe %incs% %libs% %defs% -Wall -Wextra -Werror -march=x86-64-v3 -O3 -ffast-math src/dejong.c src/timer.c -o dejong.exe
