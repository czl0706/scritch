#ifndef __MATH_UTILS_H__
#define __MATH_UTILS_H__

static inline float invSqrt(float x) 
{
    float half_x = 0.5f * x;
    float y = x;
    long i = *(long *)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float *)&i;
    y = y * (1.5f - (half_x * y * y));
    return y;
}

static inline float invNorm(float x, float y, float z)
{
    return invSqrt(x * x + y * y + z * z);
}

static inline void transform(float *accX, float *accY, float *accZ, float sin, float cos, float rotx, float roty)
{
    float x = *accX;
    float y = *accY;
    float z = *accZ;
    *accX = x * cos + roty * z * sin;
    *accY = y * cos - rotx * z * sin;
    *accZ = z * cos + (rotx * y - roty * x) * sin;
}

#endif