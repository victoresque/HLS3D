#include "render.h"

half triangle_area (half v0[3], half v1[3], half v2[3]) {
    #pragma HLS inline
    #pragma HLS pipeline
    return (v1[0]-v0[0])*(v2[1]-v0[1])-(v1[1]-v0[1])*(v2[0]-v0[0]);
}

void matrix_mul_vector (bool mode, half m[3][4], half scale, half v[3], half result[3]) {
    #pragma HLS inline
    #pragma HLS pipeline
    for (int i = 0; i < 3; i++) {
        half sum = 0;
        for (int j = 0; j < 3; j++) {
            sum += m[i][j] * v[j] * scale;
        }

        if (mode) {  // rotation only
            result[i] = sum;
        } else {
            result[i] = sum + m[i][3];
        }
    }
}

half vector_dot_vector (half u[3], half v[3]) {
    #pragma HLS inline
    #pragma HLS pipeline
    half sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += u[i] * v[i];
    }
    return sum;
}

void cam_project (half cv[3], float scale[3], float offset[3], half fv[3], int fvi[3]) {
    #pragma HLS inline
    #pragma HLS pipeline
    for (int i = 0; i < 3; i++) {
        fv[i] = cv[i] * ((half) scale[i]) + ((half) offset[i]);
        fvi[i] = round(fv[i]);
    }
}

half hmax (half a, half b) {
    #pragma HLS inline
    #pragma HLS pipeline
    return (a > b) ? a : b;
}

half hmin (half a, half b) {
    #pragma HLS inline
    #pragma HLS pipeline
    return (a < b) ? a : b;
}
