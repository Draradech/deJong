#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <immintrin.h>
#include <GL/freeglut.h>
#include <time.h>
#include <pthread.h>

#include "timer.h"


#include "avx_mathfun.h"
#define USE_SSE2
#include "sse_mathfun.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define NUM_THREADS 32
#define BASE_RESOLUTION (1440 * 1440)

int base_steps = 800000;
float fade_factor = 0.95;
float brightness_factor = 1.0;

typedef struct
{
   uint16_t r, g, b, a;
} rgb;

typedef struct
{
   float x1, y1;
   int over;
} AttractorState;

GLuint gltex;
rgb * tex;

static int steps;
static int screenW;
static int screenH;
static int texSize;
static float a, b, c, d;
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

static void fadeby_avx16(float f)
{
   int pixels = texSize * texSize;
   __m256i mul = _mm256_set1_epi16((uint16_t)(f * 65536));
   for (int i = 0; i < pixels; i += 4) // Process 4 64bit pixels at a time
   {
      __m256i p = _mm256_load_si256((__m256i*)&tex[i]);
      // taking only the high 16 bits of the multiplication result divides by 65536, we scaled f accordingly
      p = _mm256_mulhi_epu16(p, mul);
      _mm256_store_si256((__m256i*)&tex[i], p);
   }
}

static void fadeby(float f)
{
    int pixels = texSize * texSize;
    for (int i = 0; i < pixels; i++)
    {
        tex[i].r = tex[i].r * f;
        tex[i].g = tex[i].g * f;
        tex[i].b = tex[i].b * f;
    }
}

void* attractor_thread(void* arg)
{
   AttractorState* state = (AttractorState*)arg;
   while(1) {
      // Wait for the main thread to signal that a new frame is ready
      pthread_barrier_wait(&frame_start);
      state->over = 0;
      double bright = brightness_factor * 500e6 / base_steps;
      for (int i = 0; i < steps / NUM_THREADS; i++)
      {
         float res[4];
         v4sf in = _mm_set_ps(
            a * state->y1,
            b * state->x1 + (float)M_PI_2,
            c * state->x1,
            d * state->y1 + (float)M_PI_2
            );
         _mm_store_ps(res, sin_ps(in)); // 4 approximate sin values simultaneously
         float x2 = res[3] - res[2]; // float x2 = sin(a * state->y1) - cos(b * state->x1);
         float y2 = res[1] - res[0]; // float y2 = sin(c * state->x1) - cos(d * state->y1);
         if(i > 10)
         {
            float dx = x2 - state->x1;
            float dy = y2 - state->y1;
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
      fadeby_avx16(fade_factor);
   }
   else
   {
      fadeby(fade_factor);
   }
   timer_stop("fade");

   timer_start("attr");
   // a = 1.4;
   // b = -2.3;
   // c = 2.4;
   // d = -2.1;
   a = 4 * sin(t * 1.03);
   b = 4 * sin(t * 1.07);
   c = 4 * sin(t * 1.09);
   d = 4 * sin(t * 1.13);

   // threads need random starting points each frame or they will all converge on the same points eventually
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
   double threshold = 0.6 * steps;
   double max_speed = 30.0;
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

char status[1024] ="\n\n\n\n\n";
void checkFps(void)
{
   static unsigned long timeOld;
   static int frameCounter;
   
   unsigned long time = glutGet(GLUT_ELAPSED_TIME);
   frameCounter++;
   
   if(time > timeOld + 2000)
   {
      sprintf(status, "%.1lf fps\n", frameCounter * 1000.0 / (time - timeOld));
      frameCounter = 0;
      timeOld = time;
      timer_report(status + strlen(status), 1);
   }
}

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

   timer_start("text");
   checkFps();

   glRasterPos2i(10, 20);
   glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)"\"o\" to show/hide overlay");

   glRasterPos2i(10, screenH - 75);
   glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)status);

   timer_stop("text");
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
         else glutReshapeWindow(1600, 900);
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
   int current_resolution = texSize * texSize;
   steps = (int)(base_steps * ((double)current_resolution / BASE_RESOLUTION));
}

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

   tex = malloc(sizeof(rgb) * texSize * texSize);
   memset(tex, 0, sizeof(rgb) * texSize * texSize);

   printf("Allocated at %lx\n", (unsigned long)tex);
   
   calculate_steps();  // Calculate steps when the window is reshaped
}

int main(int argc, char* argv[])
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
   glutInitWindowSize(1600, 900);
   glutCreateWindow("Peter De Jong Attractor");
   
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
   timer_start("fade");
   timer_stop("fade");
   timer_start("attr");
   timer_stop("attr");
   glutMainLoop();
   
   return 0;
}
