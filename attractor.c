#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <GL/freeglut.h>
#include <time.h>
#include <threads.h>

GLuint gltex;

typedef struct
{
   unsigned char r;
   unsigned char g;
   unsigned char b;
   unsigned char a;
} rgb;
rgb * tex;

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define BASE_STEPS 200000
#define BASE_RESOLUTION (1440 * 1440)
int steps;

static int screenW;
static int screenH;
static double t;

static int p = 1;
static int avx = 1;


#define MAX_TIMERS 8
static int active_timers = 0;
#ifdef _WIN32
#include <windows.h>
static struct {
    LONGLONG start;  // QuadPart from QueryPerformanceCounter
    const char* name;
    double total_ms;
    int calls;
} timers[MAX_TIMERS];

static void timer_start(const char* name) {
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

static void timer_stop(const char* name) {
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

static void timer_start(const char* name) {
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

static void timer_stop(const char* name) {
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
static void timer_report(int reset) {
    for(int i = 0; i < active_timers; i++) {
        printf("%s: %.3f ms avg (%.1f ms total, %d calls)\n",
               timers[i].name,
               timers[i].total_ms / timers[i].calls,
               timers[i].total_ms,
               timers[i].calls);
        
        if(reset) {
            timers[i].total_ms = 0;
            timers[i].calls = 0;
        }
    }
}

static double frand(double max)
{
   return max * rand() / RAND_MAX;
}

static void calculate_steps(void)
{
   int current_resolution = screenH * screenH;
   steps = (int)(BASE_STEPS * ((double)current_resolution / BASE_RESOLUTION));
}

static void fadeby_avx(float f)
{
   int x0 = (screenW - screenH) / 2;
   int x1 = x0 + screenH;
   x0 = x0 / 8 * 8;
   x1 = x1 / 8 * 8;
   
   __m256i mul = _mm256_set1_epi16(f * 256);
   __m256i mask = _mm256_set1_epi16(0xFF);
   
   for (int y = 0; y < screenH; y++)
   {
      for (int x = x0; x < x1; x += 8)
      {
         int i = y * screenW + x;
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
   for (int y = 0; y < screenH; y++)
   {
      for (int x = (screenW - screenH) / 2; x < (screenW + screenH) / 2; x++)
      {
         int i = y * screenW + x;
         tex[i].r = tex[i].r * f;
         tex[i].g = tex[i].g * f;
         tex[i].b = tex[i].b * f;
      }
   }
}

typedef struct
{
    double x1, y1;
    int steps;
    int over;
} AttractorState;

static double a, b, c, d;

#define NUM_THREADS 4
thrd_t threads[NUM_THREADS];
AttractorState states[NUM_THREADS];
mtx_t frame_mutex;
cnd_t frame_start_cond;
cnd_t frame_complete_cond;
static int threads_ready = 0;
static int threads_complete = 0;
static int frame_active = 0;

int attractor_thread(void* arg) {
    AttractorState* state = (AttractorState*)arg;
    while(1) {
        // Wait for frame start
        mtx_lock(&frame_mutex);
        threads_ready++;
        while(!frame_active) {
            cnd_wait(&frame_start_cond, &frame_mutex);
        }
        mtx_unlock(&frame_mutex);
        
        // Do work...
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
            int x = x2 * 0.25 * (screenH - 11) + screenW * 0.5;
            int y = y2 * 0.25 * (screenH - 11) + screenH * 0.5;
            rgb col = tex[y * screenW + x];
            col.r = MIN(col.r + dr, 255);
            col.g = MIN(col.g + dg, 255);
            col.b = MIN(col.b + db, 255); 
            if (col.b == 255) state->over++;
            tex[y * screenW + x] = col;
            state->x1 = x2;
            state->y1 = y2;
        }

        // Signal completion
        mtx_lock(&frame_mutex);
        threads_complete++;
        if(threads_complete == NUM_THREADS) {
            cnd_signal(&frame_complete_cond);
        }
        mtx_unlock(&frame_mutex);
    }
    return 0;
}

static void render(int dt)
{
   timer_start("fade");
   if (avx)
   {
      fadeby_avx(0.98);
   }
   else
   {
      fadeby(0.98);
   }
   timer_stop("fade");

   timer_start("attr");
   a = 4 * sin(t * 1.03);
   b = 4 * sin(t * 1.07);
   c = 4 * sin(t * 1.09);
   d = 4 * sin(t * 1.13);
   
   // Start new frame
   mtx_lock(&frame_mutex);
   threads_complete = 0;
   frame_active = 1;
   cnd_broadcast(&frame_start_cond);
   
   // Wait for all threads to complete
   while(threads_complete < NUM_THREADS) {
       cnd_wait(&frame_complete_cond, &frame_mutex);
   }
   frame_active = 0;
   mtx_unlock(&frame_mutex);

   int over = 0;
   for (int i = 0; i < NUM_THREADS; i++)
   {
      over += states[i].over;
   }

   const double threshold_percent = 0.6;
   const double max_speed = 30.0;

   // Calculate speed based on how many points are overexposed (meaning, they fall on top of each other, a "boring" attractor)
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
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenW, screenH, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
   
   glBegin(GL_TRIANGLE_STRIP);
   glTexCoord2d(0,0);
   glVertex2d(0,0);
   glTexCoord2d(1,0);
   glVertex2d(screenW,0);
   glTexCoord2d(0,1);
   glVertex2d(0,screenH);
   glTexCoord2d(1,1);
   glVertex2d(screenW,screenH);
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

static void reshape(int w, int h)
{
   printf("%d, %d\n", w, h);
   screenW = w;
   screenH = h;
   
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0, screenW, screenH, 0, -100, 100); /* left, right, bottom, top, near, far */
   
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glViewport(0, 0, screenW, screenH);
   
   if(tex) free(tex);
   
   tex = malloc(sizeof(rgb) * screenW * screenH);
   memset(tex, 0, sizeof(rgb) * screenW * screenH);
   
   calculate_steps();  // Calculate steps when the window is reshaped
   for(int i = 0; i < NUM_THREADS; i++)
   {
      states[i].steps = steps / NUM_THREADS;
   }
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
   
   mtx_init(&frame_mutex, mtx_plain);
   cnd_init(&frame_start_cond);
   cnd_init(&frame_complete_cond);
   for(int i = 0; i < NUM_THREADS; i++)
   {
      states[i].x1 = frand(2.0) - 1.0;
      states[i].y1 = frand(2.0) - 1.0;
      thrd_create(&threads[i], attractor_thread, &states[i]);
   }
   glutMainLoop();
   
   return 0;
}
