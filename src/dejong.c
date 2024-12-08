#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/freeglut.h>
#include <pthread.h>

#include "avx_mathfun.h"
#include "sse_mathfun.h"

#include "timer.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define NUM_THREADS (32)
#define BASE_RESOLUTION (1440 * 1440)
#define BASE_STEPS (800000)
#define BASE_BRIGHT (1e9)

#ifdef _WIN32
#include <windows.h>
static int usDiv = 0;
static int64_t micros()
{
    int64_t time;
    if (usDiv == 0)
    {
        int64_t freq;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        if (freq % 1000000 != 0)
        {
            fprintf(stderr, "PerfCounter non-integer fraction of us: freq = %lld\n", freq);
            exit(0);
        }
        usDiv = freq / 1000000;
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&time);

    return time / usDiv;
}

void usleep(unsigned int usec)
{
	HANDLE timer;
	LARGE_INTEGER ft;

	ft.QuadPart = -(10 * (__int64)usec);

	timer = CreateWaitableTimer(NULL, TRUE, NULL);
	SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
	WaitForSingleObject(timer, INFINITE);
	CloseHandle(timer);
}
#else
#include <unistd.h> // usleep
#include <sys/time.h>
int64_t micros()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
#endif

typedef struct
{
   const char* desc;
   const char* fmt;
   double* var;
   double min;
   double max;
   double step;
} Parameter;

typedef struct
{
   uint16_t r, g, b, a;
} rgb;

typedef struct
{
   double x1, y1;
   int over;
} ThreadState;

GLuint gltex;
rgb * tex;

static int steps;
static int screenW;
static int screenH;
static int texSize;

static double overlay = 1;
static int param_index = 0;

static double avx = 1;
static double sse = 1;
static double speedup = 1;
static double fs = 0;
static double t_from_dt = 1;
static double a_from_t = 1;
static double b_from_t = 1;
static double c_from_t = 1;
static double d_from_t = 1;
static double step_factor = 1.0;
static double bright_factor = 1.0;
static double fade_factor = 0.95;
static double anim_speed = 1;
static double t = 0;
static double a = 0, b = 0, c = 0, d = 0;

Parameter parameters[] = {
   {"avx fade         ", "%s %s: %.0lf\n", &avx,             0, 1, 1},
   {"sse sin          ", "%s %s: %.0lf\n", &sse,             0, 1, 1},
   {"skip fixed-point ", "%s %s: %.0lf\n", &speedup,         0, 1, 1},
   {"full screen      ", "%s %s: %.0lf\n", &fs,              0, 1, 1},  
   {"t from dt        ", "%s %s: %.0lf\n", &t_from_dt,       0, 1, 1},
   {"a from t         ", "%s %s: %.0lf\n", &a_from_t,        0, 1, 1},
   {"b from t         ", "%s %s: %.0lf\n", &b_from_t,        0, 1, 1},
   {"c from t         ", "%s %s: %.0lf\n", &c_from_t,        0, 1, 1},
   {"d from t         ", "%s %s: %.0lf\n", &d_from_t,        0, 1, 1},
   {"steps factor     ", "%s %s: %.2lf\n", &step_factor,     0, 3, 0.1},
   {"brightness       ", "%s %s: %.2lf\n", &bright_factor,   0, 10, 0.05},
   {"fade factor      ", "%s %s: %.2lf\n", &fade_factor,     0, 1, 0.01},
   {"animation speed  ", "%s %s: %.2lf\n", &anim_speed,      0, 3, 0.01},
   {"t                ", "%s %s: %.4lf\n", &t,               0, 1e4, 0.0001},
   {"a                ", "%s %s: %.3lf\n", &a,             -10, 10, 0.001},
   {"b                ", "%s %s: %.3lf\n", &b,             -10, 10, 0.001},
   {"c                ", "%s %s: %.3lf\n", &c,             -10, 10, 0.001},
   {"d                ", "%s %s: %.3lf\n", &d,             -10, 10, 0.001},
};

static pthread_t threads[NUM_THREADS];
static ThreadState states[NUM_THREADS];
static pthread_barrier_t frame_start;
static pthread_barrier_t frame_end;

static double frand(double max)
{
   return max * rand() / RAND_MAX;
}

static void fadeby_avx(double f)
{
   int pixels = texSize * texSize;
   __m256i mul = _mm256_set1_epi16((uint16_t)(f * 65536));
   for (int i = 0; i < pixels; i += 4) // Process 4 64bit pixels at a time
   {
      __m256i p = _mm256_load_si256((__m256i*)&tex[i]);
      // Taking only the high 16 bits of the multiplication result divides by 65536, we scaled f accordingly
      p = _mm256_mulhi_epu16(p, mul);
      _mm256_store_si256((__m256i*)&tex[i], p);
   }
}

static void fadeby(double f)
{
    int pixels = texSize * texSize;
    for (int i = 0; i < pixels; i++)
    {
        tex[i].r = tex[i].r * f;
        tex[i].g = tex[i].g * f;
        tex[i].b = tex[i].b * f;
    }
}

void attractor_fast(ThreadState* state)
{
   float bright = bright_factor * BASE_BRIGHT / BASE_STEPS / step_factor;
   float x1 = state->x1;
   float y1 = state->y1;
   float la = a;
   float lb = b;
   float lc = c;
   float ld = d;

   state->over = 0;
   for (int i = 0; i < steps / NUM_THREADS; i++)
   {
      float res[4];
      v4sf in = _mm_set_ps(
         la * y1,
         lb * x1 + (float)M_PI_2,
         lc * x1,
         ld * y1 + (float)M_PI_2
         );
      _mm_store_ps(res, sin_ps(in)); // 4 approximate sin values simultaneously
      float x2 = res[3] - res[2]; // float x2 = sin(a * state->y1) - cos(b * state->x1);
      float y2 = res[1] - res[0]; // float y2 = sin(c * state->x1) - cos(d * state->y1);
      if(i > 10)
      {
         float dx = x2 - x1;
         float dy = y2 - y1;
         int dr = bright * fabsf(dx);
         int dg = bright * fabsf(dy);
         int db = bright;
         int x = x2 * 0.25 * texSize * 0.96 + texSize * 0.5;
         int y = y2 * 0.25 * texSize * 0.96 + texSize * 0.5;
         rgb col = tex[y * texSize + x];
         col.r = MIN(col.r + dr, 65535);
         col.g = MIN(col.g + dg, 65535);
         col.b = MIN(col.b + db, 65535); 
         if (col.b == 65535) state->over++;
         tex[y * texSize + x] = col;
      }
      x1 = x2;
      y1 = y2;
   }
}

void attractor_precise(ThreadState* state)
{
   double bright = bright_factor * BASE_BRIGHT / BASE_STEPS / step_factor;
   double x1 = state->x1;
   double y1 = state->y1;

   state->over = 0;
   for (int i = 0; i < steps / NUM_THREADS; i++)
   {
      double x2 = sin(a * y1) - cos(b * x1);
      double y2 = sin(c * x1) - cos(d * y1);
      if(i > 10)
      {
         double dx = x2 - x1;
         double dy = y2 - y1;
         int dr = bright * fabs(dx);
         int dg = bright * fabs(dy);
         int db = bright;
         int x = x2 * 0.25 * texSize * 0.96 + texSize * 0.5;
         int y = y2 * 0.25 * texSize * 0.96 + texSize * 0.5;
         rgb col = tex[y * texSize + x];
         col.r = MIN(col.r + dr, 65535);
         col.g = MIN(col.g + dg, 65535);
         col.b = MIN(col.b + db, 65535); 
         if (col.b == 65535) state->over++;
         tex[y * texSize + x] = col;
      }
      x1 = x2;
      y1 = y2;
   }
}

void* attractor_thread(void* arg)
{
   ThreadState* state = (ThreadState*)arg;
   while(1) {
      // Wait for the main thread to signal that a new frame is ready
      pthread_barrier_wait(&frame_start);

      if (sse) attractor_fast(state);
      else attractor_precise(state);

      // Signal to the main thread that this thread has finished the frame
      pthread_barrier_wait(&frame_end);
   }
   return NULL;
}

static void render(int dt)
{
   timer_start("fade");
   if (avx) fadeby_avx(fade_factor);
   else fadeby(fade_factor);
   timer_stop("fade");

   timer_start("attr");
   if (a_from_t) a = 4 * sin(t * 1.03);
   if (b_from_t) b = 4 * sin(t * 1.07);
   if (c_from_t) c = 4 * sin(t * 1.09);
   if (d_from_t) d = 4 * sin(t * 1.13);

   // Threads need random starting points each frame or they will all converge on the same points eventually
   for(int i = 0; i < NUM_THREADS; i++)
   {
      states[i].x1 = frand(2.0) - 1.0;
      states[i].y1 = frand(2.0) - 1.0;
   }
   
   pthread_barrier_wait(&frame_start); // Signal threads to start rendering
   pthread_barrier_wait(&frame_end);   // Wait for all threads to finish frame

   int over = 0;
   for (int i = 0; i < NUM_THREADS; i++)
   {
      over += states[i].over;
   }

   double speed = 1.0;
   if(speedup)
   {
      // Calculate speed based on how many points are overexposed (meaning, they fall on top of each other, a "boring" attractor)
      double threshold = 0.6 * steps;
      double max_speed = 30.0;
      speed = 1.0 + (max_speed - 1.0) * (over - threshold) / (steps - threshold);
      speed = MAX(1.0, speed);
   }
   if (t_from_dt) t += 1e-8 * dt * speed * anim_speed;
   timer_stop("attr");
}

void idle(void)
{
    // next frame time
    static int64_t timeLast;
    int64_t time = micros();
    int64_t timeNext = timeLast + 1e6 / 60; // target 60 fps

    // sleep until 1ms before frame
    if (timeNext - time > 1000)
    {
        // don't sleep longer than 1ms
        usleep(MIN(1000, timeNext - time - 1000));
        time = micros();
    }

    // busy wait last 1ms for accurate frame timing
    if (time >= timeNext)
    {
        render(time - timeLast);
        glutPostRedisplay();
        timeLast = time;
    }
}

char status[1024];
void buildStatus(void)
{
   static unsigned long timeOld = -1000;
   static int frameCounter = 0;
   
   unsigned long time = glutGet(GLUT_ELAPSED_TIME);
   frameCounter++;
   
   if(time > timeOld + 1000)
   {
      sprintf(status, "%.1lf fps\n", frameCounter * 1000.0 / (time - timeOld));
      frameCounter = 0;
      timeOld = time;
      timer_report(status + strlen(status), 1);
   }
}

char buffer[1024];
void draw(void)
{
   timer_start("draw");
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_TEXTURE_2D);
   glBindTexture(GL_TEXTURE_2D, gltex);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, texSize, texSize, 0, GL_RGBA, GL_UNSIGNED_SHORT, tex);

   int y0 = (screenH - texSize) / 2;
   int y1 = y0 + texSize;
   int x0 = (screenW - texSize) / 2;
   int x1 = x0 + texSize;
   
   glBegin(GL_TRIANGLE_STRIP);
   glTexCoord2d(0, 0);
   glVertex2d(x0, y0);
   glTexCoord2d(1, 0);
   glVertex2d(x1, y0);
   glTexCoord2d(0, 1);
   glVertex2d(x0, y1);
   glTexCoord2d(1, 1);
   glVertex2d(x1, y1);
   glEnd();
   
   glDisable(GL_TEXTURE_2D);
   timer_stop("draw");

   timer_start("ovrl");
   buildStatus();
   if (overlay)
   {
      glRasterPos2i(10, 20);
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"\"o\" to show/hide overlay\n\n");
      for (int i = 0; i < (int)(sizeof(parameters) / sizeof(Parameter)); i++)
      {
         sprintf(buffer, parameters[i].fmt, i == param_index ? ">" : " ", parameters[i].desc, *parameters[i].var);
         glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)buffer);
      }
      glRasterPos2i(10, screenH - 90);
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)status);
      sprintf(buffer, "W: %d, H: %d, T: %d, S: %d\n", screenW, screenH, texSize, steps);
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)buffer);
      glRasterPos2i(screenW - 300, 20);
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"Peter de Jong attractor\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  x2 = sin(a * y1) - cos(b * x1)\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  y2 = sin(c * x1) - cos(d * y1)\n\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"Coloured by:\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  red:   abs(x2 - x1)\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  green: abs(y2 - y1)\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  blue:  1.0\n\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"Animated by:\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  a = 4 * sin(t * 1.03)\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  b = 4 * sin(t * 1.07)\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  c = 4 * sin(t * 1.09)\n");
      glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"  d = 4 * sin(t * 1.13)\n");
   }
   timer_stop("ovrl");
   glutSwapBuffers();
}

static void key(int k, int x, int y)
{
   (void)x;
   (void)y;
   double fs_old = fs;
   double step_old = step_factor;
   Parameter* p = &parameters[param_index];
   switch(k)
   {
      case GLUT_KEY_DOWN:
         param_index = (param_index + 1) % (sizeof(parameters) / sizeof(Parameter));
         break;
      case GLUT_KEY_UP:
         param_index = (param_index - 1 + sizeof(parameters) / sizeof(Parameter)) % (sizeof(parameters) / sizeof(Parameter));
         break;
      case GLUT_KEY_LEFT:
         *p->var -= p->step;
         *p->var = MAX(p->min, *p->var);
         *p->var = MIN(p->max, *p->var);
         break;
      case GLUT_KEY_RIGHT:
         *p->var += p->step;
         *p->var = MAX(p->min, *p->var);
         *p->var = MIN(p->max, *p->var);
         break;
   }
   if(fs != fs_old)
   {
      if(fs) glutFullScreen();
      else glutReshapeWindow(1600, 900);
   }
   if(step_factor != step_old)
   {
      steps = step_factor * BASE_STEPS * texSize * texSize / BASE_RESOLUTION;
   }
}

static void key_c(unsigned char k, int x, int y)
{
   (void)x;
   (void)y;
   double fs_old = fs;
   double step_old = step_factor;
   Parameter* p = &parameters[param_index];
   switch(k)
   {
      case 27:
         glutLeaveMainLoop();
         break;
      case 'o':
         overlay = !overlay;
         break;
      case 'f':
         fs = !fs;
         break;
      case 'p':
         t_from_dt = !t_from_dt;
         break;
      case 'r':
         if(p->step != 1)
         {
            *p->var = frand(p->max - p->min) + p->min;
            *p->var = floor(*p->var / p->step) * p->step;
         }
         break;
   }
   if(fs != fs_old)
   {
      if(fs) glutFullScreen();
      else glutReshapeWindow(1600, 900);
   }
   if(step_factor != step_old)
   {
      steps = step_factor * BASE_STEPS * texSize * texSize / BASE_RESOLUTION;
   }
}

void* traw;
static void reshape(int w, int h)
{
   screenW = w;
   screenH = h;
   int texSizeOld = texSize;
   texSize = screenH / 8 * 8;
   
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0, screenW, screenH, 0, -100, 100); /* left, right, bottom, top, near, far */
   
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glViewport(0, 0, screenW, screenH);
   
   if(texSizeOld == texSize) return;
   
   if(traw) free(traw);
   traw = malloc(sizeof(rgb) * texSize * texSize + 32);
   // tex needs to be 32-bit aligned for avx2
   tex = (rgb*)((uint64_t)traw / 32 * 32 + 32);
   memset(tex, 0, sizeof(rgb) * texSize * texSize);

   steps = step_factor * BASE_STEPS * texSize * texSize / BASE_RESOLUTION;
}

int main(int argc, char* argv[])
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
   glutInitWindowSize(1600, 900);
   glutCreateWindow("Peter de Jong attractor");
   
   glutIdleFunc(idle);
   glutDisplayFunc(draw);
   glutReshapeFunc(reshape);
   glutKeyboardFunc(key_c);
   glutSpecialFunc(key);
   
   glEnable(GL_TEXTURE_2D);
   
   glGenTextures(1, &gltex);
   glBindTexture(GL_TEXTURE_2D, gltex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   t = frand(1e4);

   pthread_barrier_init(&frame_start, NULL, NUM_THREADS + 1);  // +1 for main thread
   pthread_barrier_init(&frame_end, NULL, NUM_THREADS + 1);  // +1 for main thread
   for(int i = 0; i < NUM_THREADS; i++)
   {
      pthread_create(&threads[i], NULL, attractor_thread, &states[i]);
   }

   // Prepopulate timers for correct ordering
   timer_start("fade");
   timer_stop("fade");
   timer_start("attr");
   timer_stop("attr");
   glutMainLoop();
   
   return 0;
}
