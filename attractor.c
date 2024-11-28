#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <GL/freeglut.h>
#include <time.h>
#include <pthread.h>

#include "timer.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define NUM_THREADS 32
#define BASE_STEPS 200000
#define BASE_RESOLUTION (1440 * 1440)
#define FADE_FACTOR 0.98

typedef struct
{
   unsigned char r;
   unsigned char g;
   unsigned char b;
   unsigned char a;
} rgb;

typedef struct
{
   double x1, y1;
   int steps;
   int over;
} AttractorState;

GLuint gltex;
rgb * tex;

static int steps;
static int screenW;
static int screenH;
static int texSize;
static double a, b, c, d;
static double t;
static int p = 1;
static int avx = 1;

pthread_t threads[NUM_THREADS];
AttractorState states[NUM_THREADS];
pthread_barrier_t frame_start;
pthread_barrier_t frame_end;

static double frand(double max)
{
   return max * rand() / RAND_MAX;
}

static void fadeby_avx(float f)
{
   __m256i mul = _mm256_set1_epi16(f * 256);
   __m256i mask = _mm256_set1_epi16(0xFF);
   
   for (int y = 0; y < texSize; y++)
   {
      for (int x = 0; x < texSize; x += 8)
      {
         int i = y * texSize + x;
         __m256i p = _mm256_load_si256((__m256i*)&tex[i]);
         
         // Separate odd and even bytes
         __m256i even = _mm256_and_si256(p, mask);
         __m256i odd = _mm256_and_si256(_mm256_srli_epi16(p, 8), mask);
         
         // Multiply each
         even = _mm256_mullo_epi16(even, mul);
         odd = _mm256_mullo_epi16(odd, mul);
         
         // Divide by 256 (shift right by 8)
         even = _mm256_srli_epi16(even, 8);
         odd = _mm256_srli_epi16(odd, 8);

         // move odd bytes back to high position
         odd = _mm256_slli_epi16(odd, 8);
         
         // Combine results
         p = _mm256_or_si256(even, odd);
         _mm256_store_si256((__m256i*)&tex[i], p);
      }
   }
}

static void fadeby(float f)
{
   for (int y = 0; y < texSize; y++)
   {
      for (int x = 0; x < texSize; x++)
      {
         int i = y * texSize + x;
         tex[i].r = tex[i].r * f;
         tex[i].g = tex[i].g * f;
         tex[i].b = tex[i].b * f;
      }
   }
}

void* attractor_thread(void* arg)
{
   AttractorState* state = (AttractorState*)arg;
   while(1) {
      // Wait for the main thread to signal that a new frame is ready
      pthread_barrier_wait(&frame_start);
      state->over = 0;
      double bright = 10;
      for (int i = 0; i < state->steps; i++)
      {
         double x2 = sin(a * state->y1) - cos(b * state->x1);
         double y2 = sin(c * state->x1) - cos(d * state->y1);
         double dx = x2 - state->x1;
         double dy = y2 - state->y1;
         int dr = bright * fabs(dx);
         int dg = bright * fabs(dy);
         int db = bright;
         int x = x2 * 0.25 * texSize * 0.96 + texSize * 0.5;
         int y = y2 * 0.25 * texSize * 0.96 + texSize * 0.5;
         rgb col = tex[y * texSize + x];
         col.r = MIN(col.r + dr, 255);
         col.g = MIN(col.g + dg, 255);
         col.b = MIN(col.b + db, 255); 
         if (col.b == 255) state->over++;
         tex[y * texSize + x] = col;
         state->x1 = x2;
         state->y1 = y2;
      }
      // Wait for all threads to finish the frame
      pthread_barrier_wait(&frame_end);
   }
   return NULL;
}

static void render(int dt)
{
   timer_start("fade");
   if (avx)
   {
      fadeby_avx(FADE_FACTOR);
   }
   else
   {
      fadeby(FADE_FACTOR);
   }
   timer_stop("fade");

   timer_start("attr");
   a = 4 * sin(t * 1.03);
   b = 4 * sin(t * 1.07);
   c = 4 * sin(t * 1.09);
   d = 4 * sin(t * 1.13);

   // threads need random starting points each frame or they will all converge on the same point eventually
   for(int i = 0; i < NUM_THREADS; i++)
   {
      states[i].x1 = frand(2.0) - 1.0;
      states[i].y1 = frand(2.0) - 1.0;
   }
   
   pthread_barrier_wait(&frame_start); // signal frame ready
   pthread_barrier_wait(&frame_end);   // wait for all threads to finish

   int over = 0;
   for (int i = 0; i < NUM_THREADS; i++)
   {
      over += states[i].over;
   }

   // Calculate speed based on how many points are overexposed (meaning, they fall on top of each other, a "boring" attractor)
   const double threshold_percent = 0.6;
   const double max_speed = 30.0;
   double threshold = threshold_percent * steps;
   double speed = 1.0 + (max_speed - 1.0) * (over - threshold) / (steps - threshold);
   speed = MAX(1.0, speed);
   t += 0.00001 * dt * speed * p;
   timer_stop("attr");
}

static void reset()
{
   t = frand(1e6);
}

void idle(void)
{
   static unsigned long timeOld;
   unsigned long time = glutGet(GLUT_ELAPSED_TIME);
   int dt = time - timeOld;
   if(dt < 16)
   { 
      // not yet time for a new frame, back to mainloop
      return;
   }
   timeOld = time;
   render(dt);
   glutPostRedisplay();
}

void checkFps(void)
{
   static unsigned long timeOld;
   static int frameCounter;
   
   unsigned long time = glutGet(GLUT_ELAPSED_TIME);
   frameCounter++;
   
   if(time > timeOld + 2000)
   {
      printf("%.1lf fps\n", frameCounter * 1000.0 / (time - timeOld));
      frameCounter = 0;
      timeOld = time;
      timer_report(1);
   }
}

void draw(void)
{
   glEnable(GL_TEXTURE_2D);
   glBindTexture(GL_TEXTURE_2D, gltex);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSize, texSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);

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
   checkFps();
   
   glutSwapBuffers();
}

static void key(unsigned char key, int x, int y)
{
   static int fs = 0;
   switch(key)
   {
      case 27:
         glutLeaveMainLoop();
         break;
      case 'f':
         fs = !fs;
         if(fs) glutFullScreen();
         else glutReshapeWindow(800, 600);
         break;
      case 'r':
         reset();
         break;
      case 'p':
         p = !p;
         break;
      case 'a':
         avx = !avx;
         break;
   }
}

static void calculate_steps(void)
{
   int current_resolution = screenH * screenH;
   steps = (int)(BASE_STEPS * ((double)current_resolution / BASE_RESOLUTION));
   for(int i = 0; i < NUM_THREADS; i++)
   {
      states[i].steps = steps / NUM_THREADS;
   }
}

static void reshape(int w, int h)
{
   printf("%d, %d\n", w, h);
   screenW = w;
   screenH = h;
   texSize = screenH / 8 * 8;
   
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0, screenW, screenH, 0, -100, 100); /* left, right, bottom, top, near, far */
   
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glViewport(0, 0, screenW, screenH);
   
   if(tex) free(tex);
   
   tex = malloc(sizeof(rgb) * texSize * texSize);
   memset(tex, 0, sizeof(rgb) * texSize * texSize);
   
   calculate_steps();  // Calculate steps when the window is reshaped
}

int main(int argc, char* argv[])
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
   glutCreateWindow("RenderTest");
   glutReshapeWindow(800, 600);
   
   glutIdleFunc(idle);
   glutDisplayFunc(draw);
   glutReshapeFunc(reshape);
   glutKeyboardFunc(key);
   
   glEnable(GL_TEXTURE_2D);
   
   glGenTextures(1, &gltex);
   glBindTexture(GL_TEXTURE_2D, gltex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   reset();
   
   pthread_barrier_init(&frame_start, NULL, NUM_THREADS + 1);  // +1 for main thread
   pthread_barrier_init(&frame_end, NULL, NUM_THREADS + 1);  // +1 for main thread
   for(int i = 0; i < NUM_THREADS; i++)
   {
      pthread_create(&threads[i], NULL, attractor_thread, &states[i]);
   }
   glutMainLoop();
   
   return 0;
}
