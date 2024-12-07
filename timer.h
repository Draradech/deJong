#ifndef TIMER_H
#define TIMER_H

 #define MAX_TIMERS 8
 
 void timer_start(const char* name);
 void timer_report(char* output, int reset);
 void timer_stop(const char* name);

#endif