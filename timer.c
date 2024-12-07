#include <stdio.h>

#include "timer.h"

static int active_timers = 0;

#ifdef _WIN32
#include <windows.h>
static struct {
    LONGLONG start;
    const char* name;
    double total_ms;
    int calls;
} timers[MAX_TIMERS];

void timer_start(const char* name) {
    LARGE_INTEGER start;
    for(int i = 0; i < active_timers; i++) {
        if(strcmp(timers[i].name, name) == 0) {
            QueryPerformanceCounter(&start);
            timers[i].start = start.QuadPart;
            return;
        }
    }
    if(active_timers < MAX_TIMERS) {
        timers[active_timers].name = name;
        timers[active_timers].total_ms = 0;
        timers[active_timers].calls = 0;
        QueryPerformanceCounter(&start);
        timers[active_timers].start = start.QuadPart;
        active_timers++;
    }
}

void timer_stop(const char* name) {
    LARGE_INTEGER end, freq;
    QueryPerformanceCounter(&end);
    QueryPerformanceFrequency(&freq);
    
    for(int i = 0; i < active_timers; i++) {
        if(strcmp(timers[i].name, name) == 0) {
            double ms = (end.QuadPart - timers[i].start) * 1000.0 / freq.QuadPart;
            timers[i].total_ms += ms;
            timers[i].calls++;
            return;
        }
    }
}
#else
static struct {
    struct timespec start;
    const char* name;
    double total_ms;
    int calls;
} timers[MAX_TIMERS];

void timer_start(const char* name) {
    for(int i = 0; i < active_timers; i++) {
        if(strcmp(timers[i].name, name) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &timers[i].start);
            return;
        }
    }
    if(active_timers < MAX_TIMERS) {
        timers[active_timers].name = name;
        timers[active_timers].total_ms = 0;
        timers[active_timers].calls = 0;
        clock_gettime(CLOCK_MONOTONIC, &timers[active_timers].start);
        active_timers++;
    }
}

void timer_stop(const char* name) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    for(int i = 0; i < active_timers; i++) {
        if(strcmp(timers[i].name, name) == 0) {
            double ms = (end.tv_sec - timers[i].start.tv_sec) * 1000.0 +
                       (end.tv_nsec - timers[i].start.tv_nsec) / 1000000.0;
            timers[i].total_ms += ms;
            timers[i].calls++;
            return;
        }
    }
}
#endif

void timer_report(char* output, int reset) {
    for(int i = 0; i < active_timers; i++) {
        sprintf(output, "%s: %.3f ms avg (%.1f ms total, %d calls)\n",
               timers[i].name,
               timers[i].total_ms / timers[i].calls,
               timers[i].total_ms,
               timers[i].calls);
        output += strlen(output);
        if(reset) {
            timers[i].total_ms = 0;
            timers[i].calls = 0;
        }
    }
}
