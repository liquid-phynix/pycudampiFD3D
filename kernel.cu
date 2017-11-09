extern "C" {

__device__ float lap(float shar[14][14][14], float ddm2, int x, int y, int z, int ox, int oy, int oz, int c){
    int m = c - 1;
    int p = c + 1;
    return ddm2 *
        (shar[x + ox + m][y + oy + c][z + oz + c] + shar[x + ox + p][y + oy + c][z + oz + c]
         + shar[x + ox + c][y + oy + m][z + oz + c] + shar[x + ox + c][y + oy + p][z + oz + c]
         + shar[x + ox + c][y + oy + c][z + oz + m] + shar[x + ox + c][y + oy + c][z + oz + p]
         - 6.0f * shar[x + ox + c][y + oy + c][z + oz + c]);
}

__global__ void kernel_timestep(float* in, float* out, int sx, int sy, int sz, float ddm2){
    //    ddm2 = 1.0f;

    __shared__ float cache     [14][14][14];
    __shared__ float cache_lap [14][14][14];  // 12x12x12
    __shared__ float cache_lap2[14][14][14]; // 10x10x10
    __shared__ float cache_lap3[14][14][14]; // 8x8x8

    int sq14 = 14 * 14;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    int bx = blockIdx.x * 8;
    int by = blockIdx.y * 8;
    int bz = blockIdx.z * 8;

    int load_idx = (z * 8 + y) * 8 + x;
    int load_y = load_idx / 14;
    int load_z = load_idx % 14;

    int base_addr = ((bz + load_z) * sy + by + load_y) * sx + bx;

    if(load_idx < sq14){
        for(int load_x = 0; load_x < 14; load_x++){
            cache[load_x][load_y][load_z] = in[base_addr + load_x];
        }
    }

    __syncthreads();
    // cache is filled now

    // ********************************************************************************
    // compute lap  
    // phase 0,0,0
    int ox = 0; int oy = 0; int oz = 0;
    cache_lap[x + ox + 1][y + oy + 1][z + oz + 1]                             = lap(cache, ddm2, x, y, z, ox, oy, oz, 1);
    __syncthreads();
    // phase x,0,0
    ox = 8; oy = 0; oz = 0;
    if(x < 4) cache_lap[x + ox + 1][y + oy + 1][z + oz + 1]                   = lap(cache, ddm2, x, y, z, ox, oy, oz, 1);
    __syncthreads();
    // phase 0,y,0
    ox = 0; oy = 8; oz = 0;
    if(y < 4) cache_lap[x + ox + 1][y + oy + 1][z + oz + 1]                   = lap(cache, ddm2, x, y, z, ox, oy, oz, 1);
    __syncthreads();
    // phase 0,0,z
    ox = 0; oy = 0; oz = 8;
    if(z < 4) cache_lap[x + ox + 1][y + oy + 1][z + oz + 1]                   = lap(cache, ddm2, x, y, z, ox, oy, oz, 1);
    __syncthreads();
    // phase x,y,0
    ox = 8; oy = 8; oz = 0;
    if(x < 4 && y < 4) cache_lap[x + ox + 1][y + oy + 1][z + oz + 1]          = lap(cache, ddm2, x, y, z, ox, oy, oz, 1);
    __syncthreads();
    // phase x,0,z
    ox = 8; oy = 0; oz = 8;
    if(x < 4 && z < 4) cache_lap[x + ox + 1][y + oy + 1][z + oz + 1]          = lap(cache, ddm2, x, y, z, ox, oy, oz, 1);
    __syncthreads();
    // phase 0,y,z
    ox = 0; oy = 8; oz = 8;
    if(y < 4 && z < 4) cache_lap[x + ox + 1][y + oy + 1][z + oz + 1]          = lap(cache, ddm2, x, y, z, ox, oy, oz, 1);
    __syncthreads();
    // phase x,y,z
    ox = 8; oy = 8; oz = 8;
    if(x < 4 && y < 4 && z < 4) cache_lap[x + ox + 1][y + oy + 1][z + oz + 1] = lap(cache, ddm2, x, y, z, ox, oy, oz, 1);
    __syncthreads();
    // ********************************************************************************


    // ********************************************************************************
    // compute lap2
    // phase 0,0,0
    ox = 0; oy = 0; oz = 0;
    cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2]                             = lap(cache_lap, ddm2, x, y, z, ox, oy, oz, 2);
    __syncthreads();
    // phase x,0,0
    ox = 8; oy = 0; oz = 0;
    if(x < 2) cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2]                   = lap(cache_lap, ddm2, x, y, z, ox, oy, oz, 2);
    __syncthreads();
    // phase 0,y,0
    ox = 0; oy = 8; oz = 0;
    if(y < 2) cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2]                   = lap(cache_lap, ddm2, x, y, z, ox, oy, oz, 2);
    __syncthreads();
    // phase 0,0,z
    ox = 0; oy = 0; oz = 8;
    if(z < 2) cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2]                   = lap(cache_lap, ddm2, x, y, z, ox, oy, oz, 2);
    __syncthreads();
    // phase x,y,0
    ox = 8; oy = 8; oz = 0;
    if(x < 2 && y < 2) cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2]          = lap(cache_lap, ddm2, x, y, z, ox, oy, oz, 2);
    __syncthreads();
    // phase x,0,z
    ox = 8; oy = 0; oz = 8;
    if(x < 2 && z < 2) cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2]          = lap(cache_lap, ddm2, x, y, z, ox, oy, oz, 2);
    __syncthreads();
    // phase 0,y,z
    ox = 0; oy = 8; oz = 8;
    if(y < 2 && z < 2) cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2]          = lap(cache_lap, ddm2, x, y, z, ox, oy, oz, 2);
    __syncthreads();
    // phase x,y,z
    ox = 8; oy = 8; oz = 8;
    if(x < 2 && y < 2 && z < 2) cache_lap2[x + ox + 2][y + oy + 2][z + oz + 2] = lap(cache_lap, ddm2, x, y, z, ox, oy, oz, 2);
    __syncthreads();
    // ********************************************************************************

    // ********************************************************************************
    // compute lap3
    cache_lap3[x + 3][y + 3][z + 3] = lap(cache_lap2, ddm2, x, y, z, 0, 0, 0, 3);
    __syncthreads();
    // ********************************************************************************

    float c = cache[x + 3][y + 3][z + 3];
    float xm = cache[x + 2][y + 3][z + 3];
    float xp = cache[x + 4][y + 3][z + 3];
    float ym = cache[x + 3][y + 2][z + 3];
    float yp = cache[x + 3][y + 4][z + 3];
    float zm = cache[x + 3][y + 3][z + 2];
    float zp = cache[x + 3][y + 3][z + 4];

    float lapPsi3 = ddm2 *
        (xm * xm * xm + xp * xp * xp
         + ym * ym * ym + yp * yp * yp
         + zm * zm * zm + zp * zp * zp
         - 6.0f * c * c * c);

    // ********************************************************************************
    // PFC
    out[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3] = 
        cache[x + 3][y + 3][z + 3]
        + 0.0001f * (lapPsi3
                + (1.0f - 0.25f) * cache_lap[x + 3][y + 3][z + 3]
                + 2.0f * cache_lap2[x + 3][y + 3][z + 3]
                + cache_lap3[x + 3][y + 3][z + 3]);
    // ********************************************************************************

    // ********************************************************************************
    // LAPLACE
    //    out[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3] = 0.16666666666f * (cache_lap[x + 3][y + 3][z + 3] + 6.0f * cache[x + 3][y + 3][z + 3]);
    // ********************************************************************************

    // ********************************************************************************
    // ID
    // out[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3] = cache_lap[x + 3][y + 3][z + 3];
    // ********************************************************************************

    // ********************************************************************************
    // ID
    // out[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3] = in[((bz + z + 3) * sy + by + y + 3) * sx + bx + x + 3];
    // ********************************************************************************
}



__device__ int ifun(int sx, int sy, int sz, int x, int y, int z){ return (z * sy + y) * sx + x; }
#define I(x,y,z) ifun(sx, sy, sz, x, y, z)
__device__ int mod(float a, float b){ return ((int)a) - ((int)b) * floor(a / b); }
__device__ int wrap(int a, int b){ return mod(a - 3, b) + 3; }

__global__ void kernel_pbc_noz(float* ar, int sx, int sy, int sz){
    int sxp = sx - 6;
    int syp = sy - 6;
    int szp = sz - 6;
    int a = threadIdx.x;
    int b = threadIdx.y;
    // threadblock: 16x16
    int x_mul_max = sx / 16;
    int y_mul_max = sy / 16;
    int z_mul_max = sz / 16;

    // x - y
    // for(int x_mul = 0; x_mul <= x_mul_max; x_mul++){
    //   for(int y_mul = 0; y_mul <= y_mul_max; y_mul++){
    //     int xx = 16 * x_mul + a;
    //     int yy = 16 * y_mul + b;
    //     if(xx < sx && yy < sy){
    // 	int xc = wrap(xx, sxp);
    // 	int yc = wrap(yy, syp);
    // 	ar[I(xx, yy, sz - 3)] = ar[I(xc, yc, 3)];
    // 	ar[I(xx, yy, sz - 2)] = ar[I(xc, yc, 4)];
    // 	ar[I(xx, yy, sz - 1)] = ar[I(xc, yc, 5)];
    // 	ar[I(xx, yy, 2)] = ar[I(xc, yc, sz - 4)];
    // 	ar[I(xx, yy, 1)] = ar[I(xc, yc, sz - 5)];
    // 	ar[I(xx, yy, 0)] = ar[I(xc, yc, sz - 6)];
    //     }
    //   }
    // }  

    // x - z
    for(int x_mul = 0; x_mul <= x_mul_max; x_mul++){
        for(int z_mul = 0; z_mul <= z_mul_max; z_mul++){
            int xx = 16 * x_mul + a;
            int zz = 16 * z_mul + b;
            if(xx < sx && zz < sz){
                int xc = wrap(xx, sxp);
                int zc = wrap(zz, szp);
                ar[I(xx, sy - 3, zz)] = ar[I(xc, 3, zc)];
                ar[I(xx, sy - 2, zz)] = ar[I(xc, 4, zc)];
                ar[I(xx, sy - 1, zz)] = ar[I(xc, 5, zc)];
                ar[I(xx, 2, zz)] = ar[I(xc, sy - 4, zc)];
                ar[I(xx, 1, zz)] = ar[I(xc, sy - 5, zc)];
                ar[I(xx, 0, zz)] = ar[I(xc, sy - 6, zc)];
            }
        }
    }

    // y - z
    for(int y_mul = 0; y_mul <= y_mul_max; y_mul++){
        for(int z_mul = 0; z_mul <= z_mul_max; z_mul++){
            int yy = 16 * y_mul + a;
            int zz = 16 * z_mul + b;
            if(yy < sy && zz < sz){
                int yc = wrap(yy, syp);
                int zc = wrap(zz, szp);
                ar[I(sx - 3, yy, zz)] = ar[I(3, yc, zc)];
                ar[I(sx - 2, yy, zz)] = ar[I(4, yc, zc)];
                ar[I(sx - 1, yy, zz)] = ar[I(5, yc, zc)];
                ar[I(2, yy, zz)] = ar[I(sx - 4, yc, zc)];
                ar[I(1, yy, zz)] = ar[I(sx - 5, yc, zc)];
                ar[I(0, yy, zz)] = ar[I(sx - 6, yc, zc)];
            }
        }
    }
}

__global__ void kernel_ghost_copy(float* ar, float* out_left, float* out_right, int sx, int sy, int sz){
    int y_mul_max = sy / 256;
    for(int z = 0; z < 3; z++){
        for(int y_mul = 0; y_mul <= y_mul_max; y_mul++){
            int y = y_mul * 256 + threadIdx.x;
            if(y < sy){
                int base_addr_lin = (z * sy + y) * sx;
                int base_addr_left = ((z + 3) * sy + y) * sx;
                int base_addr_right = ((sz - 6 + z) * sy + y) * sx;
                for(int x = 0; x < sx; x++){
                    out_left[base_addr_lin + x] = ar[base_addr_left + x];
                    out_right[base_addr_lin + x] = ar[base_addr_right + x];
                }
            }
        }
    }
}

__global__ void kernel_ghost_copy_inv(float* ar, float* out_left, float* out_right, int sx, int sy, int sz){
    int y_mul_max = sy / 256;
    for(int z = 0; z < 3; z++){
        for(int y_mul = 0; y_mul <= y_mul_max; y_mul++){
            int y = y_mul * 256 + threadIdx.x;
            if(y < sy){
                int base_addr_lin = (z * sy + y) * sx;
                int base_addr_left = ((z + 0) * sy + y) * sx;
                int base_addr_right = ((sz - 3 + z) * sy + y) * sx;
                for(int x = 0; x < sx; x++){
                    ar[base_addr_left + x] = out_left[base_addr_lin + x];
                    ar[base_addr_right + x] = out_right[base_addr_lin + x];
                }
            }
        }
    }
}

  __global__ void kernel_source(float* ar, int sx, int sy, int sz){
    ar[((3) * sy + 3) * sx + 3] = 1.0f;
    //    ar[((3 + 200) * sy + 3) * sx + 3 + 1200] = 1.0f;
  }
}

} // extern "C"
