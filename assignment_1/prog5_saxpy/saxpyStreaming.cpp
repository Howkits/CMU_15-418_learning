#include <smmintrin.h> // For _mm_stream_load_si128
#include <emmintrin.h> // For _mm_mul_ps
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#pragma pack(16)

extern void saxpySerial(int N,
                        float scale,
                        float X[],
                        float Y[],
                        float result[]);

void saxpyStreaming(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    // Replace this code with ones that make use of the streaming instructions

    __m128i ans;
    __m128 _scale = _mm_set_ps(scale, scale, scale, scale);

    for (int i = 0; i < N; i += 4)
    {
        __m128i _x = _mm_stream_load_si128((__m128i *)(X + i));
        __m128i _y = _mm_stream_load_si128((__m128i *)(Y + i));

        ans = (__m128i)_mm_add_ps(_mm_mul_ps((__m128)_x, (__m128)_scale), (__m128)_y);

        _mm_store_si128((__m128i *)(result + i), ans);
    }

    //saxpySerial(N, scale, X, Y, result);
}
