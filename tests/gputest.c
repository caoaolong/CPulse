#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "qwen/gpu.h"
#include "qwen/ops.h"

#define ROWS 2048
#define COLS 2048
#define TOL 1e-4

/* Wall clock: clock() only counts CPU time and stays 0 for GPU waits. */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main()
{

    if (gpu_init() != 0)
    {
        fprintf(stderr, "gputest: gpu_init failed\n");
        return 1;
    }

    static uint16_t W[ROWS * COLS];
    static float x[COLS], y_cpu[ROWS], y_gpu[ROWS];

    srand(42);
    for (size_t i = 0; i < ROWS * COLS; i++)
    {
        float v = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        W[i] = cp_fp32_to_bf16(v);
    }
    for (size_t i = 0; i < COLS; i++)
    {
        x[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }

    double t0 = now_sec();
    cp_matvec(y_cpu, W, x, ROWS, COLS);
    double t_cpu = now_sec() - t0;

    t0 = now_sec();
    if (gpu_matvec(y_gpu, W, x, ROWS, COLS) != 0)
    {
        fprintf(stderr, "gputest: gpu_matvec failed\n");
        gpu_shutdown();
        return 1;
    }
    double t_gpu = now_sec() - t0;

    double maxdiff = 0.0;
    for (size_t i = 0; i < ROWS; i++)
    {
        double d = fabs((double)y_cpu[i] - (double)y_gpu[i]);
        if (d > maxdiff)
            maxdiff = d;
    }

    printf("gputest: %dx%d  cpu=%.3f ms  gpu=%.3f ms  maxdiff=%.3e  %s\n",
           ROWS, COLS, t_cpu * 1000.0, t_gpu * 1000.0, maxdiff,
           maxdiff <= TOL ? "PASS" : "FAIL");

    gpu_shutdown();
    return maxdiff <= TOL ? 0 : 1;
}
