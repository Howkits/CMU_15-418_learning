#include <stdio.h>
#include <pthread.h>

#include "CycleTimer.h" // Use this code to time your threads

// Core computation of Mandelbrot set membershop
// Iterate complex number c to determine whether it diverges
static inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

//
// MandelbrotSerial --
//
// Compute an image visualizing the mandelbrot set.  The resulting
// array contains the number of iterations required before the complex
// number corresponding to a pixel could be rejected from the set.
//
// * x0, y0, x1, y1 describe the complex coordinates mapping
//   into the image viewport.
// * width, height describe the size of the output image
// * startRow, totalRows describe how much of the image to compute
void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int totalRows,
    int maxIterations,
    int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    int endRow = startRow + totalRows;

    for (int j = startRow; j < endRow; j++)
    {
        for (int i = 0; i < width; ++i)
        {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}

// Struct for passing arguments to thread routine
typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

//
// workerThreadStart --
//
// Thread entrypoint.
void *workerThreadStart(void *threadArgs)
{

    WorkerArgs *args = static_cast<WorkerArgs *>(threadArgs);

    // TODO: Implement worker thread here.

    float dx = (args->x1 - args->x0) / args->width;
    float dy = (args->y1 - args->y0) / args->height;

    int eachThreadRow = args->height / args->numThreads;
    int startRow = args->threadId * eachThreadRow;

    int totalRows = eachThreadRow;

    int endRow = startRow + totalRows;

    if (args->threadId == args->numThreads - 1)
    {
        endRow = args->height;
    }

    //printf("thread:%d dx:%f  dy:%f\n", args->threadId, dx, dy);
    //printf("thread:%d dx:%d  dy:%d\n", args->threadId, startRow, endRow);

    for (int j = int(startRow); j < int(endRow); j++)
    {
        for (unsigned int i = 0; i < args->width; ++i)
        {
            float x = args->x0 + i * dx;
            float y = args->y0 + j * dy;

            //int index = (args->threadId * args->width * args->height + j * args->width + i);
            int index = (j * args->width + i);
            args->output[index] = mandel(x, y, args->maxIterations);
        }
    }

    //printf("Hello world from thread %d\n", args->threadId);

    return NULL;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++)
    {
        // TODO: Set thread arguments here.
        args[i].x0 = x0;
        args[i].x1 = x1;

        args[i].y0 = y0;
        args[i].y1 = y1;
        //args[i].y0 = y0 + i * (y1 - y0) / numThreads;
        //args[i].y1 = args[i].y0 + (y1 - y0) / numThreads;

        args[i].width = width;
        args[i].height = height;
        //args[i].height = height / numThreads;

        args[i].maxIterations = maxIterations;

        args[i].output = output;

        args[i].threadId = i;

        args[i].numThreads = numThreads;

        //printf("thread:%d  x0:%f  x1:%f  y0:%f  y1:%f\n", args[i].threadId, args[i].x0, args[i].x1, args[i].y0, args[i].y1);
    }

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.

    for (int i = 1; i < numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i = 1; i < numThreads; i++)
        pthread_join(workers[i], NULL);
}
