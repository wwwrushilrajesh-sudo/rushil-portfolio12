/*
READ THIS FIRST -- A SINGLE, COMMENTED CUDA C++ SOLVER
===================================================
This replaces backend.cu in the previously supplied Python/native project.
No kernels.cuh or solver_api.h is needed to compile this file. Python still uses
main.py and python_api.py, and your existing build_windows.bat builds this file.

DO NOT TRY TO READ THE FILE STRAIGHT THROUGH ON YOUR FIRST PASS.
1. Search for "static void advance_one_step". Its numbered stages are the solver.
2. Search for "gauge_reset_kernel". This is the easiest complete GPU kernel.
3. Read semi_lagrangian_transport, stretch_impulse, molecular_diffusion,
   add_wale_stresses and project, then the kernels each function launches.
4. Read allocation, interface and plotting-query code last.

HOW TO RECOGNISE CPU VERSUS GPU CODE
  __global__ void ... : GPU kernel; many threads execute the same function.
  __device__ ...      : helper used inside a GPU thread.
  static void ...    : ordinary C++ host function; orders/launches GPU work.
The .cu extension does not make every function run on the GPU.

YOUR NUMBA CODE AND THE SAME IDEA IN CUDA C++
  Numba: x,y,z = cuda.grid(3)
  C++:   x = blockIdx.x*blockDim.x + threadIdx.x; similarly for y and z.
  Numba: kernel[blocks,threads](...)
  C++:   kernel<<<blocks,threads>>>(...)
Each thread handles one face or cell; it does NOT loop through the entire domain.
Short loops inside a thread handle a 3x3 matrix, RK4 stages or eight corners.

THE ARRAY NAMES
  velocity[0],velocity[1],velocity[2] = u,v,w on MAC faces.
  impulse[0], impulse[1], impulse[2]  = mx,my,mz on the same faces.
  trial_* = candidate fields, kept separate until a step passes its checks.
  phi = gauge/projection potential, with u = m - grad(phi).
  gradient[3*direction+component] = derivative of that velocity component.
  nut = WALE subgrid viscosity; nu = molecular viscosity.
  s = the Solver record (parameters and array pointers, not a numerical class).

POINTERS AND ARRAY INDEXING
  float* field points to a contiguous GPU array.
  field[i] reads/writes one entry INSIDE a GPU kernel.
  const float* field is read-only through that pointer.
  float& out = output[i] means out is an alias for output[i].
  A[i] logically means A[z,y,x], where i=(z*ny+y)*nx+x.
  x-face u/mx uses nx+1 values per row instead of nx.

IMPORTANT: HOST FUNCTIONS DO NOT ADD A LAPLACIAN TO GPU ARRAYS THEMSELVES.
For implicit viscosity, molecular_diffusion launches alternating red/black
kernels. The sweep kernel reads neighbours AND updates impulse. The host repeats
these launches until the true residual meets the tolerance.

The code retains the native solver's equations and options. This merge and its
explanatory comments do not change the calculation. Actual NVCC/GPU execution
has not been verified in this environment.
*/

// Native cylinder solver. Python never launches its momentum kernels.
// Read in this order: Solver -> geometry -> Poisson -> projection -> advance.
// Python interface declarations (included here to keep one source file).
// Plain C interface: ctypes can call these functions without pybind11.
// Python and C++ must use this exact field order and these exact types.
struct SolverConfig {
    int nx, ny, nz;
    int les, adaptive, adapt_cfl, maccormack;
    int be_max_sweeps, be_check_every, max_retries;
    int mode, high_order, particle_advection, particles_per_cell, two_d, seed;
    double diameter, inlet, reynolds, lx, ly, lz, body_x;
    double dt, dt_max, retention, stretch_alpha, wale_cw, be_tolerance;
    double correction_blend, cfl_target, gradient_limit, dt_safety;
    double diffusion_limit, determinant_limit, speed_growth, energy_growth;
    double perturbation, flip_fraction, vc_epsilon, stats_start, probe_x;
};
struct SolverReport {
    long long steps;
    double time, dt, cfl, energy, peak_velocity, peak_impulse;
    double divergence_rms, wall_slip, nut_ratio_mean, nut_ratio_max;
    double min_determinant, gradient_dt, sgs_number;
    double cd_wall, cl_wall, probe_velocity, averaging_duration;
    int diffusion_sweeps, retries;
};
#ifdef _WIN32
#define SOLVER_EXPORT extern "C" __declspec(dllexport)
#else
#define SOLVER_EXPORT extern "C"
#endif
SOLVER_EXPORT int solver_create(const SolverConfig* config, void** handle);
SOLVER_EXPORT int solver_advance(void* handle, int number_of_steps, SolverReport* report);
SOLVER_EXPORT int solver_report(void* handle, SolverReport* report);
// Field IDs: 0=u,1=v,2=w,3=mx,4=my,5=mz,6=phi,7=nu_t.
SOLVER_EXPORT int solver_copy_field(void* handle, int field, float* destination, long long count);
SOLVER_EXPORT int solver_copy_line(void* handle, double* mean, double* rms, int count);
SOLVER_EXPORT int solver_q_volume(void* handle, int x0, int y0, int width, int height,
                                 float* q, float* speed, long long count);
SOLVER_EXPORT int solver_destroy(void* handle);
SOLVER_EXPORT const char* solver_last_error();
SOLVER_EXPORT int solver_copy_crop(void* handle,int field,int x0,int y0,int width,int height,
                                  float* destination,long long count);

#include <cuda_runtime.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <memory>
using std::min;
using std::max;
// GPU CALCULATIONS: each kernel below computes grid-point values.
// Actual CUDA C++ kernels, not CuPy snippets. Included by backend.cu.
// Read indexing first, then the equation-specific arithmetic below it.
// Grid: x=inlet/outlet direction, y=cross-stream periodic, z=spanwise periodic.
// u/mx: [nz,ny,nx+1]; v/my,w/mz,phi: [nz,ny,nx].


// Average the two faces around each cell. This gives u,v,w at cell centres.
__global__ void cell_velocity_kernel(
    const float* u,
    const float* v,
    const float* w,
    long long nx,
    long long ny,
    long long nz,
    float* uc_out,
    float* vc_out,
    float* wc_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float& uc = uc_out[i];
    float& vc = vc_out[i];
    float& wc = wc_out[i];

    long x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
    long xm=(x>0 ? i-1 : i), xp1=(x+1<nx ? i+1 : i);
    long ym=(y>0 ? i-nx : i+(ny-1)*nx);
    long yp=(y+1<ny ? i+nx : i-(ny-1)*nx);
    long zm=(z>0 ? i-nx*ny : i+(nz-1)*nx*ny);
    long zp=(z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny);

        long a=(z*ny+y)*(nx+1)+x;
        uc=0.5f*(u[a]+u[a+1]);
         vc=0.5f*(v[i]+v[yp]);
         wc=0.5f*(w[i]+w[zp]);
}


// Compute g[direction][component]. Normal derivatives use faces; cross derivatives use centred velocities.
__global__ void velocity_gradients_kernel(
    const float* u,
    const float* v,
    const float* w,
    const float* uc,
    const float* vc,
    const float* wc,
    int nx,
    int ny,
    int nz,
    float h,
    float hz,
    float* g00_out,
    float* g01_out,
    float* g02_out,
    float* g10_out,
    float* g11_out,
    float* g12_out,
    float* g20_out,
    float* g21_out,
    float* g22_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float& g00 = g00_out[i];
    float& g01 = g01_out[i];
    float& g02 = g02_out[i];
    float& g10 = g10_out[i];
    float& g11 = g11_out[i];
    float& g12 = g12_out[i];
    float& g20 = g20_out[i];
    float& g21 = g21_out[i];
    float& g22 = g22_out[i];

    int x=i%nx,y=(i/nx)%ny,z=i/(nx*ny);
    int xm=x>0?i-1:i,xp1=x+1<nx?i+1:i;
    int ym=y>0?i-nx:i+(ny-1)*nx,yp=y+1<ny?i+nx:i-(ny-1)*nx;
    int zm=z>0?i-nx*ny:i+(nz-1)*nx*ny,zp=z+1<nz?i+nx*ny:i-(nz-1)*nx*ny;
    int a=(z*ny+y)*(nx+1)+x;

    g00=(u[a+1]-u[a])/h;
     g11=(v[yp]-v[i])/h;
     g22=(w[zp]-w[i])/hz;
    g01=(vc[xp1]-vc[xm])/(2*h);
     g02=(wc[xp1]-wc[xm])/(2*h);
    g10=(uc[yp]-uc[ym])/(2*h);
     g12=(wc[yp]-wc[ym])/(2*h);
    g20=(uc[zp]-uc[zm])/(2*hz);
     g21=(vc[zp]-vc[zm])/(2*hz);
}


// m_reset = r*m + (1-r)*u. This retains r of the previous gauge gradient.
__global__ void gauge_reset_kernel(
    const float* m_in,
    const float* u_in,
    float r,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float m = m_in[i];
    float u = u_in[i];
    float& out = out_out[i];

    out=r*m+(1-r)*u;
}

// Interpolation and RK4 tracing helpers, used inside each face thread.

__device__ __forceinline__ float fast_sample(const float* a, float zz, float yy, float xx,
            int nz, int ny, int nx, float oz, float oy, float ox,
            float inlet, bool apply_inlet, float &lo, float &hi, bool BOUNDS) {
            float pz=zz-oz, py=yy-oy, px=fminf(fmaxf(xx-ox,0.0f),(float)(nx-1));
            int zb=(int)floorf(pz), yb=(int)floorf(py), x0=(int)floorf(px);
            float wz=pz-floorf(pz), wy=py-floorf(py), wx=px-x0;
            // Typical departure is within one periodic box. Avoid costly integer
            // division in that hot path; retain exact modulo for large excursions.
            int z0=zb<0 ? zb+nz : (zb>=nz ? zb-nz : zb);
            int y0=yb<0 ? yb+ny : (yb>=ny ? yb-ny : yb);
            if(z0<0 || z0>=nz) z0=((zb%nz)+nz)%nz;
            if(y0<0 || y0>=ny) y0=((yb%ny)+ny)%ny;
            int z1=(z0+1==nz ? 0 : z0+1);
            int y1=(y0+1==ny ? 0 : y0+1);
            int x1=x0+1<nx ? x0+1 : nx-1;
            float q0=a[(z0*ny+y0)*nx+x0], q1=a[(z0*ny+y0)*nx+x1];
            float q2=a[(z0*ny+y1)*nx+x0], q3=a[(z0*ny+y1)*nx+x1];
            float q4=a[(z1*ny+y0)*nx+x0], q5=a[(z1*ny+y0)*nx+x1];
            float q6=a[(z1*ny+y1)*nx+x0], q7=a[(z1*ny+y1)*nx+x1];
            if(BOUNDS) {
                lo=fminf(fminf(fminf(q0,q1),fminf(q2,q3)),fminf(fminf(q4,q5),fminf(q6,q7)));
                hi=fmaxf(fmaxf(fmaxf(q0,q1),fmaxf(q2,q3)),fmaxf(fmaxf(q4,q5),fmaxf(q6,q7)));
            }
            float value=(((q0*(1-wx)+q1*wx)*(1-wy)+(q2*(1-wx)+q3*wx)*wy)*(1-wz)
                +((q4*(1-wx)+q5*wx)*(1-wy)+(q6*(1-wx)+q7*wx)*wy)*wz);
            return apply_inlet && xx<0 ? inlet : value;
        }
__device__ void fast_depart(const float* u,const float* v,const float* w,
    float zz,float yy,float xx,int nz,int ny,int nx,
    float half,float halfz,float full,float fullz,float inlet,
    float& zs,float& ys,float& xs)
{
    float k[4][3];
     // k[stage][0,1,2] contains sampled u,v,w.
    float weight[4]={1.0f,2.0f,2.0f,1.0f};
    for(int stage=0;stage<4;stage++) {
        float x=xx,y=yy,z=zz;
        if(stage>0) {
            float distance_xy=stage==3?full:half;
            float distance_z=stage==3?fullz:halfz;
            x-=distance_xy*k[stage-1][0];
            y-=distance_xy*k[stage-1][1];
            z-=distance_z*k[stage-1][2];
        }
        float low,high;
        k[stage][0]=fast_sample(u,z,y,x,nz,ny,nx+1,.5f,.5f,0,inlet,true,low,high,false);
        k[stage][1]=fast_sample(v,z,y,x,nz,ny,nx,.5f,0,.5f,0,true,low,high,false);
        k[stage][2]=fast_sample(w,z,y,x,nz,ny,nx,0,.5f,.5f,0,true,low,high,false);
    }
    float sum[3]={0,0,0};
    for(int stage=0;stage<4;stage++)for(int component=0;component<3;component++)
        sum[component]+=weight[stage]*k[stage][component];
    xs=xx-(full/6)*sum[0];
    ys=yy-(full/6)*sum[1];
    zs=zz-(fullz/6)*sum[2];
}


// Trace each arrival face backward with RK4; interpolate impulse and record eight-corner bounds.
__global__ void advect_forward_kernel(
    const float* a,
    const float* u,
    const float* v,
    const float* w,
    int nx,
    int ny,
    int nz,
    int sx,
    float oz,
    float oy,
    float ox,
    float half,
    float halfz,
    float full,
    float fullz,
    float inlet,
    int kind,
    int corrected,
    float* forward_out,
    float* lower_out,
    float* upper_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float& forward = forward_out[i];
    float& lower = lower_out[i];
    float& upper = upper_out[i];

    float zz=i/(sx*ny)+oz, yy=(i/sx)%ny+oy, xx=i%sx+ox;
    float zs,ys,xs;
    fast_depart(&u[0],&v[0],&w[0],zz,yy,xx,nz,ny,nx,half,halfz,full,fullz,inlet,zs,ys,xs);

    forward=fast_sample(&a[0],zs,ys,xs,nz,ny,sx,oz,oy,ox,kind==0 ? inlet : 0,
                        !corrected,lower,upper, true);
    // Encode inflow mask in bounds: correction must impose inlet there.
    if(xs<0){lower=__int_as_float(0x7f800000);
     upper=lower;
    }
}


// Trace the forward result in reverse, estimate interpolation error, blend and clamp the correction.
__global__ void advect_correct_kernel(
    const float* q_in,
    const float* a,
    const float* u,
    const float* v,
    const float* w,
    int nx,
    int ny,
    int nz,
    int sx,
    float oz,
    float oy,
    float ox,
    float half,
    float halfz,
    float full,
    float fullz,
    float inlet,
    const float* forward,
    const float* lower_in,
    const float* upper_in,
    int kind,
    float blend,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float q = q_in[i];
    float lower = lower_in[i];
    float upper = upper_in[i];
    float& out = out_out[i];

    float zz=i/(sx*ny)+oz, yy=(i/sx)%ny+oy, xx=i%sx+ox;
    float zs,ys,xs;
    fast_depart(&u[0],&v[0],&w[0],zz,yy,xx,nz,ny,nx,half,halfz,full,fullz,inlet,zs,ys,xs);

    float lo,hi;
    float backward=fast_sample(&forward[0],zs,ys,xs,nz,ny,sx,oz,oy,ox,
                                kind==0 ? inlet : 0,true,lo,hi, false);
    float value=forward[i]+0.5f*blend*(q-backward);
    out=isinf(lower) && lower>0 ? (kind==0 ? inlet : 0) :
        (isnan(value)||isnan(lower)||isnan(upper) ? __int_as_float(0x7fffffff) :
         fminf(fmaxf(value,lower),upper));
}


// Compute WALE viscosity from fluid-only velocity derivatives. Solid-neighbour derivatives are one-sided.
__global__ void wale_viscosity_kernel(
    const float* g00_in,
    const float* g11_in,
    const float* g22_in,
    const float* fluid_in,
    const float* uc,
    const float* vc,
    const float* wc,
    const float* mask,
    int nx,
    int ny,
    int nz,
    float h,
    float hz,
    float length2,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float g00 = g00_in[i];
    float g11 = g11_in[i];
    float g22 = g22_in[i];
    float fluid = fluid_in[grid_x + grid_nx*grid_y];
    float& out = out_out[i];

    int x=i%nx,y=(i/nx)%ny,z=i/(nx*ny);
    float g01=0.0f,g02=0.0f,g10=0.0f,g12=0.0f,g20=0.0f,g21=0.0f;
    if(fluid>0.0f) {
        int xm=x>0 ? i-1 : i, xp1=x+1<nx ? i+1 : i;
        int ym=y>0 ? i-nx : i+(ny-1)*nx;
        int yp=y+1<ny ? i+nx : i-(ny-1)*nx;
        int zm=z>0 ? i-nx*ny : i+(nz-1)*nx*ny;
        int zp=z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny;
        int cell=y*nx+x;
        bool ml=(x>0 && mask[cell-1]>0.0f);
        bool mr=(x+1<nx && mask[cell+1]>0.0f);
        bool mb=mask[(y>0 ? y-1 : ny-1)*nx+x]>0.0f;
        bool mt=mask[(y+1<ny ? y+1 : 0)*nx+x]>0.0f;
        if(ml && mr) {
            g01=(vc[xp1]-vc[xm])/(2.0f*h);
            g02=(wc[xp1]-wc[xm])/(2.0f*h);
        } else if(mr) {
            g01=(vc[xp1]-vc[i])/h;
            g02=(wc[xp1]-wc[i])/h;
        } else if(ml) {
            g01=(vc[i]-vc[xm])/h;
            g02=(wc[i]-wc[xm])/h;
        }
        if(mb && mt) {
            g10=(uc[yp]-uc[ym])/(2.0f*h);
            g12=(wc[yp]-wc[ym])/(2.0f*h);
        } else if(mt) {
            g10=(uc[yp]-uc[i])/h;
            g12=(wc[yp]-wc[i])/h;
        } else if(mb) {
            g10=(uc[i]-uc[ym])/h;
            g12=(wc[i]-wc[ym])/h;
        }
        g20=(uc[zp]-uc[zm])/(2.0f*hz);
        g21=(vc[zp]-vc[zm])/(2.0f*hz);
    }

    // Work with a small 3x3 matrix in each thread. These local arrays live
    // in thread-local storage, not as another full domain-sized field.
    float G[3][3]={{g00,g01,g02},{g10,g11,g12},{g20,g21,g22}};
    float largest=0;
     bool finite=true;
    for(int row=0;row<3;row++) for(int col=0;col<3;col++) {
        largest=fmaxf(largest,fabsf(G[row][col]));
        finite=finite && isfinite(G[row][col]);
    }
    out=0;
    if(!finite){out=__int_as_float(0x7fffffff);
    return;
    }
    if(fluid==0 || largest==0)return;
    // Normalisation prevents overflow. Restore the gradient scale at the end.
    for(int row=0;row<3;row++)for(int col=0;col<3;col++)G[row][col]/=largest;
    float G_squared[3][3]={{0,0,0},{0,0,0},{0,0,0}};
    for(int row=0;row<3;row++)for(int col=0;col<3;col++)for(int k=0;k<3;k++)
        G_squared[row][col]+=G[row][k]*G[k][col];
    float trace=(G_squared[0][0]+G_squared[1][1]+G_squared[2][2])/3;
    float strain_norm=0,deviator_norm=0;
    for(int row=0;row<3;row++)for(int col=0;col<3;col++) {
        float S=.5f*(G[row][col]+G[col][row]);
        float Sd=.5f*(G_squared[row][col]+G_squared[col][row]);
        if(row==col)Sd-=trace;
        strain_norm+=S*S;
        deviator_norm+=Sd*Sd;
    }
    // nu_t = (Cw*Delta)^2 * (Sd:Sd)^(3/2)
    //        / [(S:S)^(5/2) + (Sd:Sd)^(5/4)].
    float denominator=strain_norm*strain_norm*sqrtf(strain_norm)
                     +deviator_norm*sqrtf(sqrtf(deviator_norm));
    if(denominator>0)out=length2*largest*deviator_norm*sqrtf(deviator_norm)/denominator;

}


// At each cell solve (I + alpha*dt*g)*m_new = m_advected. Return the CHANGE, not the new field.
__global__ void stretch_impulse_kernel(
    const float* g00_in,
    const float* g01_in,
    const float* g02_in,
    const float* g10_in,
    const float* g11_in,
    const float* g12_in,
    const float* g20_in,
    const float* g21_in,
    const float* g22_in,
    const float* fluid_in,
    float scale,
    const float* a,
    const float* b,
    const float* c,
    int nx,
    int ny,
    int nz,
    float* da_out,
    float* db_out,
    float* dc_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float g00 = g00_in[i];
    float g01 = g01_in[i];
    float g02 = g02_in[i];
    float g10 = g10_in[i];
    float g11 = g11_in[i];
    float g12 = g12_in[i];
    float g20 = g20_in[i];
    float g21 = g21_in[i];
    float g22 = g22_in[i];
    float fluid = fluid_in[grid_x + grid_nx*grid_y];
    float& da = da_out[i];
    float& db = db_out[i];
    float& dc = dc_out[i];

    int x=i%nx,y=(i/nx)%ny,z=i/(nx*ny),ia=(z*ny+y)*(nx+1)+x;
    int yp=y+1<ny?i+nx:i-(ny-1)*nx,zp=z+1<nz?i+nx*ny:i-(nz-1)*nx*ny;
    float ac=0.5f*(a[ia]+a[ia+1]),bc=0.5f*(b[i]+b[yp]),cc=0.5f*(c[i]+c[zp]);
    // The PDE stretch term is -g*m. Backward Euler gives
    // (I + alpha*dt*g)*m_new = m_advected at this one cell.
    float G[3][3]={{g00,g01,g02},{g10,g11,g12},{g20,g21,g22}};
    float matrix[3][3],cofactor[3][3];
    float old_m[3]={ac,bc,cc},change[3];
    for(int row=0;row<3;row++)for(int col=0;col<3;col++)
        matrix[row][col]=(row==col?1.0f:0.0f)+scale*fluid*G[row][col];
    // Cyclic row/column indices generate the signed 2x2 minors.
    for(int row=0;row<3;row++)for(int col=0;col<3;col++) {
        int r1=(row+1)%3,r2=(row+2)%3,c1=(col+1)%3,c2=(col+2)%3;
        cofactor[row][col]=matrix[r1][c1]*matrix[r2][c2]-matrix[r1][c2]*matrix[r2][c1];
    }
    float determinant=0;
    for(int col=0;col<3;col++)determinant+=matrix[0][col]*cofactor[0][col];
    // inverse(matrix) = transpose(cofactor)/determinant.
    for(int row=0;row<3;row++) {
        float value=0;
        for(int col=0;col<3;col++)value+=cofactor[col][row]*old_m[col];
        change[row]=value/determinant-old_m[row];
    }
    da=change[0];
    db=change[1];
    dc=change[2];

}


// Average the cell-centred impulse change onto its adjacent MAC faces.
__global__ void add_stretch_to_faces_kernel(
    const float* field_in,
    const float* d,
    int nx,
    int ny,
    int nz,
    int axis,
    float* updated_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float field = field_in[i];
    float& updated = updated_out[i];

    updated=field;
    if(axis==2){
        int x=i%(nx+1),row=i/(nx+1),j=row*nx+x;
        if(x>0 && x<nx) updated=field+0.5f*(d[j-1]+d[j]);
    }else{
        int y=(i/nx)%ny,z=i/(nx*ny);
        int l=axis==1?(y>0?i-nx:i+(ny-1)*nx):(z>0?i-nx*ny:i+(nz-1)*nx*ny);
        updated=field+0.5f*(d[i]+d[l]);
    }
}


// Normal stresses are tau_xx=2*nu_t*du/dx, tau_yy=2*nu_t*dv/dy and tau_zz=2*nu_t*dw/dz.
__global__ void normal_stress_kernel(
    const float* nu_in,
    const float* u,
    const float* v,
    const float* w,
    long long nx,
    long long ny,
    long long nz,
    float h,
    float hz,
    float* xx_out,
    float* yy_out,
    float* zz_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float nu = nu_in[i];
    float& xx = xx_out[i];
    float& yy = yy_out[i];
    float& zz = zz_out[i];

    long x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
    long ym=(y>0 ? i-nx : i+(ny-1)*nx), yp=(y+1<ny ? i+nx : i-(ny-1)*nx);
    long zm=(z>0 ? i-nx*ny : i+(nz-1)*nx*ny), zp=(z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny);

        long a=(z*ny+y)*(nx+1)+x;
        xx=2*nu*(u[a+1]-u[a])/h;
        yy=2*nu*(v[yp]-v[i])/h;
        zz=2*nu*(w[zp]-w[i])/hz;
}


// Interpolate viscosity to four-cell edges and form tau_xy and tau_xz. Exclude solid neighbours.
__global__ void xy_xz_stress_kernel(
    const float* u,
    const float* v,
    const float* w,
    const float* nu,
    const float* mask,
    long long nx,
    long long ny,
    long long nz,
    float h,
    float hz,
    float* xy_out,
    float* xz_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float& xy = xy_out[i];
    float& xz = xz_out[i];

    long x=i%(nx+1), y=(i/(nx+1))%ny, z=i/((nx+1)*ny);
    xy=0;
     xz=0;
    if(x>0 && x<nx){
        long j=(z*ny+y)*nx+x, jm=(z*ny+(y+ny-1)%ny)*nx+x;
        long km=(((z+nz-1)%nz)*ny+y)*nx+x;
        long uy=(z*ny+(y+ny-1)%ny)*(nx+1)+x;
        long uz=(((z+nz-1)%nz)*ny+y)*(nx+1)+x;
        if(mask[j%(nx*ny)]>0 && mask[(j-1)%(nx*ny)]>0){
            float nuz=0.25f*(nu[j]+nu[j-1]+nu[km]+nu[km-1]);
            xz=nuz*((u[i]-u[uz])/hz+(w[j]-w[j-1])/h);
            if(mask[jm%(nx*ny)]>0 && mask[(jm-1)%(nx*ny)]>0){
                float nuy=0.25f*(nu[j]+nu[j-1]+nu[jm]+nu[jm-1]);
                xy=nuy*((u[i]-u[uy])/h+(v[j]-v[j-1])/h);
            }
        }
    }
}


// Form tau_yz = nu_t*(dv/dz + dw/dy) at y-z edges.
__global__ void yz_stress_kernel(
    const float* v,
    const float* w,
    const float* nu,
    const float* mask,
    long long nx,
    long long ny,
    long long nz,
    float h,
    float hz,
    float* yz_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float& yz = yz_out[i];

    long x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
    long ym=(y>0 ? i-nx : i+(ny-1)*nx), yp=(y+1<ny ? i+nx : i-(ny-1)*nx);
    long zm=(z>0 ? i-nx*ny : i+(nz-1)*nx*ny), zp=(z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny);

        yz=0;
        if(mask[i%(nx*ny)]>0 && mask[ym%(nx*ny)]>0){
            long yzm=(z>0 ? ym-nx*ny : ym+(nz-1)*nx*ny);
            float n=0.25f*(nu[i]+nu[ym]+nu[zm]+nu[yzm]);
            yz=n*((v[i]-v[zm])/hz+(w[i]-w[ym])/h);
        }
}


// Add dt*(d tau_xx/dx + d tau_xy/dy + d tau_xz/dz) to x impulse.
__global__ void add_sgs_x_kernel(
    const float* xx,
    const float* xy,
    const float* xz,
    long long nx,
    long long ny,
    long long nz,
    float h,
    float hz,
    const float* prior_in,
    float dt,
    float* updated_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float prior = prior_in[i];
    float& updated = updated_out[i];

    float out;

                long x=i%(nx+1), y=(i/(nx+1))%ny, z=i/((nx+1)*ny);
                out=0;
                if(x>0 && x<nx){
                    long j=(z*ny+y)*nx+x;
                    long yp=(z*ny+(y+1)%ny)*(nx+1)+x;
                    long zp=(((z+1)%nz)*ny+y)*(nx+1)+x;
                    out=(xx[j]-xx[j-1])/h+(xy[yp]-xy[i])/h+(xz[zp]-xz[i])/hz;
                }

    updated=prior+dt*out;
}


// Add the corresponding stress divergence to y or z impulse.
__global__ void add_sgs_yz_kernel(
    const float* normal,
    const float* xedge,
    const float* yz,
    long long nx,
    long long ny,
    long long nz,
    float h,
    float hz,
    int component,
    const float* prior_in,
    float dt,
    float* updated_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float prior = prior_in[i];
    float& updated = updated_out[i];

    float out;

            long x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
            long ym=(y>0 ? i-nx : i+(ny-1)*nx), yp=(y+1<ny ? i+nx : i-(ny-1)*nx);
            long zm=(z>0 ? i-nx*ny : i+(nz-1)*nx*ny), zp=(z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny);

                long a=(z*ny+y)*(nx+1)+x;
                float dx=(xedge[a+1]-xedge[a])/h;
                out=component==1 ? dx+(normal[i]-normal[ym])/h+(yz[zp]-yz[i])/hz
                                 : dx+(normal[i]-normal[zm])/hz+(yz[yp]-yz[i])/h;

    updated=prior+dt*out;
}


// Face flux differences give cell divergence. masked=1 removes faces strictly inside the cylinder.
__global__ void divergence_kernel(
    const float* a,
    const float* b,
    const float* c,
    const float* ma,
    const float* mb,
    const float* mc,
    long long nx,
    long long ny,
    long long nz,
    float ih,
    float iz,
    int masked,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float& out = out_out[i];

    long x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
    long xm=(x>0 ? i-1 : i), xp1=(x+1<nx ? i+1 : i);
    long ym=(y>0 ? i-nx : i+(ny-1)*nx);
    long yp=(y+1<ny ? i+nx : i-(ny-1)*nx);
    long zm=(z>0 ? i-nx*ny : i+(nz-1)*nx*ny);
    long zp=(z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny);

        long ia=(z*ny+y)*(nx+1)+x, ja=y*(nx+1)+x, j=y*nx+x;
        long jyp=((y+1)%ny)*nx+x;
        float ar=a[ia+1], al=a[ia], bp=b[yp], bm=b[i], cp=c[zp], cm=c[i];
        if(masked){ ar*=ma[ja+1];
         al*=ma[ja];
         bp*=mb[jyp];
         bm*=mb[j];
                    cp*=mc[j];
                     cm*=mc[j];
                     }
        out=(ar-al+bp-bm)*ih+(cp-cm)*iz;
}


// Add scale*dphi/dx to x faces. Boundary x faces are left unchanged.
__global__ void add_phi_x_kernel(
    const float* q_in,
    const float* phi,
    long long nx,
    long long ny,
    float factor,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float q = q_in[i];
    float& out = out_out[i];

    long x=i%(nx+1), row=i/(nx+1);
    out=q;
    if(x>0 && x<nx) out=q+factor*(phi[row*nx+x]-phi[row*nx+x-1]);
}


// Add scale*dphi/dy or scale*dphi/dz to the lower periodic face.
__global__ void add_phi_yz_kernel(
    const float* q_in,
    const float* phi,
    long long nx,
    long long ny,
    long long nz,
    int axis,
    float factor,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float q = q_in[i];
    float& out = out_out[i];

    long x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
    long xm=(x>0 ? i-1 : i), xp1=(x+1<nx ? i+1 : i);
    long ym=(y>0 ? i-nx : i+(ny-1)*nx);
    long yp=(y+1<ny ? i+nx : i-(ny-1)*nx);
    long zm=(z>0 ? i-nx*ny : i+(nz-1)*nx*ny);
    long zp=(z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny);

        out=q+factor*(phi[i]-phi[axis==1 ? ym : zm]);
}


// One red OR black sweep of (I - nu*dt*L)*m = rhs. Threads of one colour never read another same-colour neighbour.
__global__ void diffusion_sweep_kernel(
    const float* q_in,
    const float* field,
    const float* rhs,
    long long nx,
    long long ny,
    long long nz,
    float alpha,
    float ih2,
    float iz2,
    int sign,
    int color,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float q = q_in[i];
    float& out = out_out[i];

    long x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
    long xm=(x>0 ? i-1 : i), xp1=(x+1<nx ? i+1 : i);
    long ym=(y>0 ? i-nx : i+(ny-1)*nx);
    long yp=(y+1<ny ? i+nx : i-(ny-1)*nx);
    long zm=(z>0 ? i-nx*ny : i+(nz-1)*nx*ny);
    long zp=(z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny);

        if (((x+y+z)&1) == color) {
            float off=(field[yp]+field[ym])*ih2 + (field[zp]+field[zm])*iz2;
            float xdiag;
            if (x==0) { off+=field[i+1]*ih2;
             xdiag=(2-sign)*ih2;
             }
            else if (x==nx-1) { off+=field[i-1]*ih2;
             xdiag=ih2;
             }
            else { off+=(field[i-1]+field[i+1])*ih2;
             xdiag=2*ih2;
             }
            out=(rhs[i]+alpha*off)/(1.0f+alpha*(2*ih2+2*iz2+xdiag));
        }
}


// Evaluate the true residual abs(m - nu*dt*L(m) - rhs). This is what controls convergence.
__global__ void diffusion_residual_kernel(
    const float* q_in,
    const float* rhs_in,
    const float* f,
    int nx,
    int ny,
    int nz,
    float alpha,
    float ih2,
    float iz2,
    int sign,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float q = q_in[i];
    float rhs = rhs_in[i];
    float& out = out_out[i];

    int x=i%nx,y=(i/nx)%ny,z=i/(nx*ny);
    int ym=y>0?i-nx:i+(ny-1)*nx,yp=y+1<ny?i+nx:i-(ny-1)*nx;
    int zm=z>0?i-nx*ny:i+(nz-1)*nx*ny,zp=z+1<nz?i+nx*ny:i-(nz-1)*nx*ny;
    float lap=(f[zp]-2*q+f[zm])*iz2+(f[yp]-2*q+f[ym])*ih2
        +(f[x+1<nx?i+1:i]-2*q+(x==0?sign*q:f[i-1]))*ih2;
    out=fabsf(q-alpha*lap-rhs);
}


// Maximum row sum of absolute velocity derivatives: a conservative stretching bound.
__global__ void gradient_norm_kernel(
    const float* g00_in,
    const float* g01_in,
    const float* g02_in,
    const float* g10_in,
    const float* g11_in,
    const float* g12_in,
    const float* g20_in,
    const float* g21_in,
    const float* g22_in,
    const float* fluid_in,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float g00 = g00_in[i];
    float g01 = g01_in[i];
    float g02 = g02_in[i];
    float g10 = g10_in[i];
    float g11 = g11_in[i];
    float g12 = g12_in[i];
    float g20 = g20_in[i];
    float g21 = g21_in[i];
    float g22 = g22_in[i];
    float fluid = fluid_in[grid_x + grid_nx*grid_y];
    float& out = out_out[i];

    if(!isfinite(g00+g01+g02+g10+g11+g12+g20+g21+g22)){out=__int_as_float(0x7fffffff);
    }
    else out=fluid*fmaxf(fmaxf(fabsf(g00)+fabsf(g01)+fabsf(g02),
                       fabsf(g10)+fabsf(g11)+fabsf(g12)),
                       fabsf(g20)+fabsf(g21)+fabsf(g22));
}


// det(I + alpha*dt*g). A small determinant means the local implicit solve is nearly singular.
__global__ void stretch_determinant_kernel(
    const float* g00_in,
    const float* g01_in,
    const float* g02_in,
    const float* g10_in,
    const float* g11_in,
    const float* g12_in,
    const float* g20_in,
    const float* g21_in,
    const float* g22_in,
    const float* fluid_in,
    float scale,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float g00 = g00_in[i];
    float g01 = g01_in[i];
    float g02 = g02_in[i];
    float g10 = g10_in[i];
    float g11 = g11_in[i];
    float g12 = g12_in[i];
    float g20 = g20_in[i];
    float g21 = g21_in[i];
    float g22 = g22_in[i];
    float fluid = fluid_in[grid_x + grid_nx*grid_y];
    float& out = out_out[i];

    float s=scale*fluid;
    float m00=1+s*g00,m01=s*g01,m02=s*g02;
    float m10=s*g10,m11=1+s*g11,m12=s*g12;
    float m20=s*g20,m21=s*g21,m22=1+s*g22;
    float c00=m11*m22-m12*m21,c01=m02*m21-m01*m22,c02=m01*m12-m02*m11;
    float c10=m12*m20-m10*m22,c11=m00*m22-m02*m20,c12=m02*m10-m00*m12;
    float c20=m10*m21-m11*m20,c21=m01*m20-m00*m21,c22=m00*m11-m01*m10;
    float det=m00*c00+m01*c10+m02*c20;
    out=det;
}


// Optional fourth-order cross derivative in regular fluid; fall back near the body and domain ends.
__global__ void higher_order_derivative_kernel(
    const float* q_in,
    const float* f,
    const float* fluid,
    long long nx,
    long long ny,
    long long nz,
    int axis,
    float spacing,
    float* out_out,
    int grid_nx,
    int grid_ny,
    int grid_nz)
{
    // Equivalent to cuda.grid(3) in your Numba code: one thread per grid point.
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (grid_x >= grid_nx || grid_y >= grid_ny || grid_z >= grid_nz) return;

    // Contiguous [z,y,x] storage: x changes fastest.
    long long i = ((long long)grid_z * grid_ny + grid_y) * grid_nx + grid_x;
    float q = q_in[i];
    float& out = out_out[i];

    long x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
    long xm=(x>0 ? i-1 : i), xp1=(x+1<nx ? i+1 : i);
    long ym=(y>0 ? i-nx : i+(ny-1)*nx);
    long yp=(y+1<ny ? i+nx : i-(ny-1)*nx);
    long zm=(z>0 ? i-nx*ny : i+(nz-1)*nx*ny);
    long zp=(z+1<nz ? i+nx*ny : i-(nz-1)*nx*ny);

        long fm, fp, fmm, fpp, jm, jp, jmm, jpp;
        if(axis==2) {
            fm=xm;
             fp=xp1;
            fmm=x>1 ? i-2 : fm;
            fpp=x+2<nx ? i+2 : fp;
            jm=(long)y*nx+(x>0 ? x-1 : x);
            jp=(long)y*nx+(x+1<nx ? x+1 : x);
            jmm=(long)y*nx+(x>1 ? x-2 : x);
            jpp=(long)y*nx+(x+2<nx ? x+2 : x);
        } else if(axis==1) {
            fm=ym;
             fp=yp;
            fmm=(y>=2 ? i-2*nx : i+((y+ny-2)%ny-y)*nx);
            fpp=(y+2<ny ? i+2*nx : i+((y+2)%ny-y)*nx);
            jm=(long)((y+ny-1)%ny)*nx+x;
            jp=(long)((y+1)%ny)*nx+x;
            jmm=(long)((y+ny-2)%ny)*nx+x;
            jpp=(long)((y+2)%ny)*nx+x;
        } else {
            fm=zm;
             fp=zp;
            fmm=(z>=2 ? i-2*nx*ny : i+((z+nz-2)%nz-z)*nx*ny);
            fpp=(z+2<nz ? i+2*nx*ny : i+((z+2)%nz-z)*nx*ny);
            jm=jp=jmm=jpp=(long)y*nx+x;
        }
        bool high = (axis!=2 || (x>=2 && x+2<nx)) &&
                    fluid[jm]>0.5f && fluid[jp]>0.5f &&
                    fluid[jmm]>0.5f && fluid[jpp]>0.5f;
        if (high) out=(f[fmm]-8.0f*f[fm]+8.0f*f[fp]-f[fpp])/(12.0f*spacing);
        else if(axis==2 && x==0 && nx>=3)
            out=(-3.0f*q+4.0f*f[i+1]-f[i+2])/(2.0f*spacing);
        else if(axis==2 && x==nx-1 && nx>=3)
            out=(3.0f*q-4.0f*f[i-1]+f[i-2])/(2.0f*spacing);
        else out=(f[fp]-f[fm])/(2.0f*spacing);
}


static thread_local std::string last_error;
static const double PI = 3.14159265358979323846;
static void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
 }
static void check_fft(cufftResult e) { if(e != CUFFT_SUCCESS) throw std::runtime_error("cuFFT failed, code " + std::to_string((int)e));
 }
static void check_solver(cusolverStatus_t e) { if(e != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER failed, code " + std::to_string((int)e));
 }
// Every stencil launch uses the same easy-to-read 3-D grid.
static dim3 threads3(8,8,4);
static dim3 grid3(int nx,int ny,int nz) { return dim3((nx+7)/8,(ny+7)/8,(nz+3)/4);
 }
static int blocks1(long long n) { return int((n+255)/256);
 }

// This struct is just a box of parameters and pointers; it has no numerical methods.
// Components are indexed 0=x, 1=y, 2=z. All fields stay on the GPU.
struct Solver {
    SolverConfig c{};
    int nx=0,ny=0,nz=0, wall_count=0,wall_x_count=0,fluid_cells=0;
    double h=0,hz=0,nu=0,time=0,last_dt=0,stats_weight=0;
    long long cells=0, faces[3]{},steps=0;
    float* velocity[3]{};
              // accepted u,v,w
    float* impulse[3]{};
               // accepted mx,my,mz
    float* trial_velocity[3]{};
        // candidate u,v,w; not committed until validated
    float* trial_impulse[3]{};
         // candidate mx,my,mz
    float* center[3]{};
                // cell-centred velocity; reused for stretch changes
    float* gradient[9]{};
              // g00,g01,g02,g10,...,g22 (direction first)
    float* phi=nullptr;
     float* trial_phi=nullptr;
    float* nut=nullptr;
     float* fluid=nullptr;
     float* face_mask[3]{};
    float* work=nullptr;
     float* forward=nullptr;
     float* lower=nullptr;
     float* upper=nullptr;
    float* rhs=nullptr;
     float* correction=nullptr;
    float* stress[6]{};
                // xx,yy,zz,xy,xz,yz
    int* wall_face=nullptr;
     int* wall_minus=nullptr;
     int* wall_plus=nullptr;
    float* wall_inverse=nullptr;
     cufftComplex* wall_data=nullptr;
     cufftComplex* wall_force=nullptr;
    std::vector<int> wall_kind_host,wall_index_host;
    cufftHandle poisson_forward=0,poisson_backward=0,wall_fft=0,wall_setup_fft=0;
    cusolverDnHandle_t dense_solver=nullptr;
    float* fft_real=nullptr;
     cufftComplex* fft_spectrum=nullptr;
    double* partial_a=nullptr;
     double* partial_b=nullptr;
    double* line_sum=nullptr;
     double* line_sum2=nullptr;
    std::vector<double> probe_time;
     std::vector<float> probe_value;
    float* particles[6]{};
     float* trial_particles[6]{};
     // z,y,x,mx,my,mz
    float* particle_before[3]{};
     float* particle_weight[3]{};
    int particle_count=0;
    double determinant=1,grad_max=0,sgs_number=0,cd=0,cl=0;
    double accepted_peak=0,accepted_energy=0,accepted_cfl_rate=0;
    bool metrics_valid=false;
    int sweeps=0,retries=0;
    std::vector<void*> allocations;
    ~Solver() {
        if(poisson_forward) cufftDestroy(poisson_forward);
        if(poisson_backward) cufftDestroy(poisson_backward);
        if(wall_fft) cufftDestroy(wall_fft);
        if(wall_setup_fft) cufftDestroy(wall_setup_fft);
        if(dense_solver) cusolverDnDestroy(dense_solver);
        for(void* p:allocations) cudaFree(p);
    }
};
// Memory ownership is kept here so even a failed initialization can clean up.
/* MEMORY HELPER: T means the entry type (float, double or int). Allocate GPU memory, record ownership, and zero it. This is setup plumbing, not a PDE update. */
template<class T> static T* allocate(Solver& s,long long count) {
    T* p=nullptr;
     check(cudaMalloc((void**)&p,size_t(count)*sizeof(T)));
    s.allocations.push_back(p);
     check(cudaMemset(p,0,size_t(count)*sizeof(T)));
     return p;
}
static void release_scratch(Solver& s,void* pointer) {
    check(cudaFree(pointer));
    s.allocations.erase(std::remove(s.allocations.begin(),s.allocations.end(),pointer),s.allocations.end());
}
static void copy_device(float* to,const float* from,long long n) {
    check(cudaMemcpy(to,from,size_t(n)*sizeof(float),cudaMemcpyDeviceToDevice));
}

// A reduction first combines 256 array entries in each block, then combines
// those block results until only one number remains. Only that number goes to CPU.
// operation: 0=max(abs), 1=max, 2=min, 3=sum.
__global__ void reduce_float_kernel(const float* field,long long n,int operation,double* partial) {
    __shared__ double values[256];
    int t=threadIdx.x;
     long long i=(long long)blockIdx.x*256+t;
    double value = operation==2 ? INFINITY : 0.0;
    if(i<n) value = operation==0 ? fabs((double)field[i]) : (double)field[i];
    values[t]=value;
     __syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(t<stride) {
            double a=values[t],b=values[t+stride];
            values[t] = (!isfinite(a) && !isinf(a)) || (!isfinite(b) && !isinf(b)) ? NAN :
                (operation==3 ? a+b : (operation==2 ? fmin(a,b) : fmax(a,b)));
        }
        __syncthreads();
    }
    if(t==0) partial[blockIdx.x]=values[0];
}
__global__ void reduce_double_kernel(const double* field,long long n,int operation,double* partial) {
    __shared__ double values[256];
     int t=threadIdx.x;
    long long i=(long long)blockIdx.x*256+t;
    double value=operation==2 ? INFINITY : 0.0;
    if(i<n) value=field[i];
    values[t]=value;
     __syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(t<stride) { double a=values[t],b=values[t+stride];
            values[t]=isnan(a)||isnan(b)?NAN:(operation==3?a+b:(operation==2?fmin(a,b):fmax(a,b)));
             }
        __syncthreads();
    }
    if(t==0) partial[blockIdx.x]=values[0];
}
static double reduce(Solver& s,const float* f,long long n,int operation) {
    int count=blocks1(n);
     reduce_float_kernel<<<count,256>>>(
    f, n, operation, s.partial_a);
    double* input=s.partial_a;
     double* output=s.partial_b;
    while(count>1) {
        int next=blocks1(count);
         reduce_double_kernel<<<next,256>>>(
    input, count, operation, output);
        std::swap(input,output);
         count=next;
    }
    double value;
     check(cudaMemcpy(&value,input,sizeof(double),cudaMemcpyDeviceToHost));
     return value;
}

// ---------------------------------------------------------------------------
// GEOMETRY: the cylinder axis is z, so its masks are only 2-D.
// ---------------------------------------------------------------------------
/* ONE-TIME SETUP: loop over the x-y geometry on the CPU. Build fluid/solid masks and the list of grid faces that require a normal-wall constraint. These masks are then copied to GPU. */
static void make_geometry(Solver& s) {
    int nx=s.nx,ny=s.ny;
     double xc=s.c.body_x/s.h,yc=0.5*ny,R=0.5*s.c.diameter/s.h;
    std::vector<float> fluid(nx*ny);
    for(int y=0;y<ny;y++) for(int x=0;x<nx;x++) {
        double dx=x+0.5-xc,dy=y+0.5-yc;
        fluid[y*nx+x]=dx*dx+dy*dy<R*R ? 0.0f:1.0f;
        if(fluid[y*nx+x]>0) s.fluid_cells+=s.nz;
    }
    check(cudaMemcpy(s.fluid,fluid.data(),fluid.size()*sizeof(float),cudaMemcpyHostToDevice));
    std::vector<int> wall_face,minus,plus;
    for(int component=0;component<3;component++) {
        int sx=nx+(component==0);
         std::vector<float> mask(sx*ny,1.0f);
        for(int y=0;y<ny;y++) for(int x=0;x<sx;x++) {
            bool left=false,right=false;
            int lo=0,hi=0;
            if(component==0) {lo=y*nx+x-1;
            hi=y*nx+x;
                left=x>0 && fluid[lo]==0;
                right=x<nx && fluid[hi]==0;
                }
            if(component==1) {lo=((y+ny-1)%ny)*nx+x;
            hi=y*nx+x;
            left=fluid[lo]==0;
            right=fluid[hi]==0;
            }
            if(component==2) {lo=hi=y*nx+x;
            left=right=fluid[hi]==0;
            }
            mask[y*sx+x]=left&&right ? 0.0f:1.0f;
            if(component<2 && left!=right) {
                wall_face.push_back(y*sx+x);
                minus.push_back(lo);
                plus.push_back(hi);
                s.wall_kind_host.push_back(component);
                s.wall_index_host.push_back(y*sx+x);
            }
        }
        if(component==0) s.wall_x_count=int(wall_face.size());
        check(cudaMemcpy(s.face_mask[component],mask.data(),mask.size()*sizeof(float),cudaMemcpyHostToDevice));
    }
    s.wall_count=int(wall_face.size());
    if(s.wall_count==0 || s.fluid_cells==0) throw std::runtime_error("Cylinder/grid has no usable wall or fluid cells");
    s.wall_face=allocate<int>(s,s.wall_count);
    s.wall_minus=allocate<int>(s,s.wall_count);
    s.wall_plus=allocate<int>(s,s.wall_count);
    check(cudaMemcpy(s.wall_face,wall_face.data(),s.wall_count*sizeof(int),cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.wall_minus,minus.data(),s.wall_count*sizeof(int),cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.wall_plus,plus.data(),s.wall_count*sizeof(int),cudaMemcpyHostToDevice));
}

// Mirror an interior face across the true circular surface. The body forcing
// is applied to a VELOCITY predictor, not directly to unprojected impulse.
// A separate output buffer prevents concurrent threads from changing samples.
__global__ void immersed_boundary_kernel(const float* input,float* output,int component,
    int nx,int ny,int nz,float xc,float yc,float radius) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    int sx=nx+(component==0);
    if(x>=sx||y>=ny||z>=nz)return;
    int i=(z*ny+y)*sx+x;
    float ox=component==0?0.0f:0.5f,oy=component==1?0.0f:0.5f;
    float dx=x+ox-xc,dy=y+oy-yc,r=sqrtf(dx*dx+dy*dy);
    output[i]=input[i];
    if(r>=radius)return;
    float safe=r>1.e-12f?r:1.0f;
    float fx=xc+(2*radius-r)*(r>1.e-12f?dx/safe:1.0f)-ox;
    float fy=yc+(2*radius-r)*(r>1.e-12f?dy/safe:0.0f)-oy;
    int x0=(int)floorf(fx),y0=(int)floorf(fy);
    float wx=fx-x0,wy=fy-y0;
    // Creation validates that mirror points fit inside the x-y box.
    float sample=(1-wy)*((1-wx)*input[(z*ny+y0)*sx+x0]+wx*input[(z*ny+y0)*sx+x0+1])
                  +wy*((1-wx)*input[(z*ny+y0+1)*sx+x0]+wx*input[(z*ny+y0+1)*sx+x0+1]);
    float rho=r/radius,g=sinf(0.5f*3.14159265358979323846f*rho*rho);
    if(rho>=1-1.e-9f)g=0;
    output[i]=-g*sample;
}
/* CYLINDER BOUNDARY: launch one kernel per velocity component. The kernel samples mirrored exterior velocity and writes interior targets; this host function only arranges those launches. */
static void apply_body(Solver& s,float* fields[3]) {
    for(int component=0;component<3;component++) {
        int sx=s.nx+(component==0);
        immersed_boundary_kernel<<<grid3(sx,s.ny,s.nz),threads3>>>(
    fields[component], s.work, component, s.nx, s.ny, s.nz, float(s.c.body_x/s.h), 0.5f*s.ny,
    float(0.5*s.c.diameter/s.h));
        copy_device(fields[component],s.work,s.faces[component]);
    }
}

// ---------------------------------------------------------------------------
// POISSON: an even x extension reproduces the cell-centred Neumann Laplacian.
// cuFFT handles the hard transform; our code supplies the physical eigenvalues.
// ---------------------------------------------------------------------------
__global__ void extend_even_x_kernel(const float* rhs,float* extended,int nx,int ny,int nz) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=2*nx||y>=ny||z>=nz)return;
    int original_x=x<nx?x:2*nx-1-x;
    extended[(z*ny+y)*(2*nx)+x]=rhs[(z*ny+y)*nx+original_x];
}
__global__ void divide_poisson_modes_kernel(cufftComplex* spectrum,int nx,int ny,int nz,double h,double hz) {
    int kx=blockIdx.x*blockDim.x+threadIdx.x,ky=blockIdx.y*blockDim.y+threadIdx.y,kz=blockIdx.z*blockDim.z+threadIdx.z;
    if(kx>nx||ky>=ny||kz>=nz)return;
    int i=(kz*ny+ky)*(nx+1)+kx;
    double sx=sin(3.14159265358979323846*kx/(2*nx)),sy=sin(3.14159265358979323846*ky/ny),sz=sin(3.14159265358979323846*kz/nz);
    float inverse=(kx==0&&ky==0&&kz==0)?0.0f:float(1/(-4*((sx*sx+sy*sy)/(h*h)+sz*sz/(hz*hz))));
    spectrum[i].x*=inverse;
    spectrum[i].y*=inverse;
}
__global__ void copy_poisson_result_kernel(const float* extended,float* phi,int nx,int ny,int nz) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx||y>=ny||z>=nz)return;
    phi[(z*ny+y)*nx+x]=extended[(z*ny+y)*(2*nx)+x]/float(2LL*nx*ny*nz);
}
/* GLOBAL LINEAR SOLVE: launch even-x packing, call cuFFT forward, divide each mode by the discrete Laplacian eigenvalue, call cuFFT inverse, and unpack/normalise. The CPU does not solve one pressure cell at a time. */
static void poisson(Solver& s,const float* rhs,float* phi) {
    extend_even_x_kernel<<<grid3(2*s.nx,s.ny,s.nz),threads3>>>(
    rhs, s.fft_real, s.nx, s.ny, s.nz);
    check_fft(cufftExecR2C(s.poisson_forward,s.fft_real,s.fft_spectrum));
    divide_poisson_modes_kernel<<<grid3(s.nx+1,s.ny,s.nz),threads3>>>(
    s.fft_spectrum, s.nx, s.ny, s.nz, s.h, s.hz);
    check_fft(cufftExecC2R(s.poisson_backward,s.fft_spectrum,s.fft_real));
    copy_poisson_result_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.fft_real, phi, s.nx, s.ny, s.nz);
}

// Build B^T P B once for each z Fourier mode. P removes divergence, while B
// puts an impulse on a normal wall face. The pseudoinverse removes the closed
// surface null mode instead of weakening the no-penetration constraint.
__global__ void divide_wall_setup_modes_kernel(const cufftDoubleComplex* source,cufftDoubleComplex* target,
    int nx,int ny,int kz,int nz,double h,double hz) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=2*nx||y>=ny)return;
    int i=y*(2*nx)+x;
    double sx=sin(3.14159265358979323846*x/(2*nx)),sy=sin(3.14159265358979323846*y/ny),sz=sin(3.14159265358979323846*kz/nz);
    double inverse=(x==0&&y==0&&kz==0)?0:1/(-4*((sx*sx+sy*sy)/(h*h)+sz*sz/(hz*hz)));
    target[i].x=source[i].x*inverse;
    target[i].y=source[i].y*inverse;
}
__global__ void sample_wall_setup_kernel(const cufftDoubleComplex* potential,const int* minus,const int* plus,
    double* column,int count,int nx,int ny,double h) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=count)return;
    int lo=minus[row],hi=plus[row];
    int lo_ext=(lo/nx)*(2*nx)+lo%nx,hi_ext=(hi/nx)*(2*nx)+hi%nx;
    column[row]=(potential[lo_ext].x-potential[hi_ext].x)/(h*(2LL*nx*ny));
}
/* ONE-TIME WALL SETUP: construct the small response matrix for each spanwise Fourier mode. Use a symmetric pseudoinverse so the closed-surface null mode does not cause leakage. Read this after understanding project(). */
static void build_wall_inverse(Solver& s) {
    int count=s.wall_count,modes=s.nz/2+1;
    long long plane=2LL*s.nx*s.ny;
    auto source=allocate<cufftDoubleComplex>(s,plane);
    auto spectrum=allocate<cufftDoubleComplex>(s,plane);
    auto shifted=allocate<cufftDoubleComplex>(s,plane);
    auto potential=allocate<cufftDoubleComplex>(s,plane);
    auto column_device=allocate<double>(s,count);
    std::vector<double> grams(size_t(modes)*count*count,0),column(count);
    std::vector<int> minus(count),plus(count);
    check(cudaMemcpy(minus.data(),s.wall_minus,count*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(plus.data(),s.wall_plus,count*sizeof(int),cudaMemcpyDeviceToHost));
    check_fft(cufftPlan2d(&s.wall_setup_fft,s.ny,2*s.nx,CUFFT_Z2Z));
    std::vector<cufftDoubleComplex> load(plane);
    // These host loops are one-time geometry setup, not CPU timestepping.
    for(int wall_column=0;wall_column<count;wall_column++) {
        if(wall_column%16==0){std::printf("Wall setup: column %d / %d\n",wall_column+1,count);
        std::fflush(stdout);
        }
        std::fill(load.begin(),load.end(),cufftDoubleComplex{0,0});
        for(int sign=0;sign<2;sign++) {
            int cell=sign==0?minus[wall_column]:plus[wall_column];
            int x=cell%s.nx,y=cell/s.nx;
            double value=(sign==0?1.0:-1.0)/s.h;
            load[y*(2*s.nx)+x].x+=value;
            load[y*(2*s.nx)+2*s.nx-1-x].x+=value;
        }
        check(cudaMemcpy(source,load.data(),plane*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice));
        check_fft(cufftExecZ2Z(s.wall_setup_fft,source,spectrum,CUFFT_FORWARD));
        for(int kz=0;kz<modes;kz++) {
            divide_wall_setup_modes_kernel<<<dim3((2*s.nx+7)/8,(s.ny+7)/8),dim3(8,8)>>>(
    spectrum, shifted, s.nx, s.ny, kz, s.nz, s.h, s.hz);
            check_fft(cufftExecZ2Z(s.wall_setup_fft,shifted,potential,CUFFT_INVERSE));
            sample_wall_setup_kernel<<<blocks1(count),256>>>(
    potential, s.wall_minus, s.wall_plus, column_device, count, s.nx, s.ny, s.h);
            check(cudaMemcpy(column.data(),column_device,count*sizeof(double),cudaMemcpyDeviceToHost));
            for(int row=0;row<count;row++) grams[(size_t(kz)*count+row)*count+wall_column]=column[row];
        }
    }
    check_solver(cusolverDnCreate(&s.dense_solver));
    double* matrix=allocate<double>(s,1LL*count*count);
    double* eigen=allocate<double>(s,count);
    int* info=allocate<int>(s,1);
    int workspace_count=0;
    check_solver(cusolverDnDsyevd_bufferSize(s.dense_solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_UPPER,count,matrix,count,eigen,&workspace_count));
    double* workspace=allocate<double>(s,workspace_count);
    std::vector<double> vectors(count*count),values(count);
    std::vector<float> inverses(size_t(modes)*count*count);
    for(int kz=0;kz<modes;kz++) {
        double* gram=grams.data()+size_t(kz)*count*count;
        for(int row=0;row<count;row++) for(int col=row;col<count;col++) {
            double value=0.5*(gram[row*count+col]+gram[col*count+row])+(row==col?1.0:0.0);
            gram[row*count+col]=gram[col*count+row]=value;
        }
        check(cudaMemcpy(matrix,gram,count*count*sizeof(double),cudaMemcpyHostToDevice));
        check_solver(cusolverDnDsyevd(s.dense_solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_UPPER,count,matrix,count,eigen,workspace,workspace_count,info));
        int status;
        check(cudaMemcpy(&status,info,sizeof(int),cudaMemcpyDeviceToHost));
        if(status!=0)throw std::runtime_error("Wall pseudoinverse eigensolve did not converge");
        check(cudaMemcpy(vectors.data(),matrix,count*count*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(values.data(),eigen,count*sizeof(double),cudaMemcpyDeviceToHost));
        // cuSOLVER returns COLUMN-major eigenvectors. Each eigenvector is a column.
        for(int row=0;row<count;row++) for(int col=0;col<count;col++) {
            double value=0;
            for(int k=0;k<count;k++) if(values[k]>1.e-11)
                value+=vectors[row+k*count]*vectors[col+k*count]/values[k];
            inverses[(size_t(kz)*count+row)*count+col]=float(value);
        }
    }
    s.wall_inverse=allocate<float>(s,inverses.size());
    check(cudaMemcpy(s.wall_inverse,inverses.data(),inverses.size()*sizeof(float),cudaMemcpyHostToDevice));
    s.wall_data=allocate<cufftComplex>(s,1LL*s.nz*count);
    s.wall_force=allocate<cufftComplex>(s,1LL*s.nz*count);
    int n[1]={s.nz},embed[1]={s.nz};
    // Each wall face is a separate z-line. Adjacent z samples are count entries apart.
    check_fft(cufftPlanMany(&s.wall_fft,1,n,embed,count,1,embed,count,1,CUFFT_C2C,count));
    for(void* temporary:{(void*)source,(void*)spectrum,(void*)shifted,(void*)potential,(void*)column_device,
                         (void*)matrix,(void*)eigen,(void*)info,(void*)workspace})release_scratch(s,temporary);
    check_fft(cufftDestroy(s.wall_setup_fft));
    s.wall_setup_fft=0;
    check_solver(cusolverDnDestroy(s.dense_solver));
    s.dense_solver=nullptr;
}

// ---------------------------------------------------------------------------
// PROJECTION: zero divergence AND zero wall-normal face velocity.
// ---------------------------------------------------------------------------
__global__ void boundary_flux_difference_kernel(const float* u,float* difference,int nx,int ny,int nz) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ny*nz)return;
    difference[i]=u[i*(nx+1)]-u[i*(nx+1)+nx];
}
__global__ void balance_outlet_kernel(float* u,int nx,int ny,int nz,float mean_difference) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<ny*nz)u[i*(nx+1)+nx]+=mean_difference;
}
__global__ void collect_wall_velocity_kernel(const float* a,const float* b,const float* phi,
    const int* faces,const int* minus,const int* plus,cufftComplex* residual,int count,int xcount,int nx,int ny,int nz,float h) {
    int wall=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
    if(wall>=count||z>=nz)return;
    int lo=minus[wall]+z*nx*ny,hi=plus[wall]+z*nx*ny;
    float value=wall<xcount?a[z*ny*(nx+1)+faces[wall]]:b[z*ny*nx+faces[wall]];
    residual[z*count+wall]={value-(phi[hi]-phi[lo])/h,0};
}
__global__ void wall_matrix_solve_kernel(const cufftComplex* residual,cufftComplex* force,const float* inverse,int count,int nz) {
    int row=blockIdx.x*blockDim.x+threadIdx.x,kz=blockIdx.y*blockDim.y+threadIdx.y;
    if(row>=count||kz>=nz)return;
    int mode=kz<=nz/2?kz:nz-kz;
    float real=0,imag=0;
    // This is a small dense matrix-vector multiply, not a volume stencil.
    for(int col=0;col<count;col++) {
        float coefficient=inverse[(mode*count+row)*count+col];
        real-=coefficient*residual[kz*count+col].x;
        imag-=coefficient*residual[kz*count+col].y;
    }
    force[kz*count+row]={real,imag};
}
__global__ void apply_wall_force_kernel(float* a,float* b,float* rhs,const cufftComplex* force,
    const int* faces,const int* minus,const int* plus,int count,int xcount,int nx,int ny,int nz,float h) {
    int wall=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
    if(wall>=count||z>=nz)return;
    float f=force[z*count+wall].x/nz;
     // cuFFT inverse is unnormalised.
    if(wall<xcount)a[z*ny*(nx+1)+faces[wall]]+=f;
    else b[z*ny*nx+faces[wall]]+=f;
    // Different wall faces can meet the same cell: add their divergence safely.
    atomicAdd(rhs+z*nx*ny+minus[wall],f/h);
    atomicAdd(rhs+z*nx*ny+plus[wall],-f/h);
}
__global__ void sum_phi_kernel(float* phi,const float* correction,int nx,int ny,int nz) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x<nx&&y<ny&&z<nz){int i=(z*ny+y)*nx+x;
    phi[i]+=correction[i];
    }
}
static void add_gradient(Solver& s,float* fields[3],const float* phi,double scale) {
    add_phi_x_kernel<<<grid3(s.nx+1,s.ny,s.nz),threads3>>>(
    fields[0], phi, s.nx, s.ny, float(scale/s.h), fields[0], s.nx+1, s.ny, s.nz);
    add_phi_yz_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    fields[1], phi, s.nx, s.ny, s.nz, 1, float(scale/s.h), fields[1], s.nx, s.ny, s.nz);
    add_phi_yz_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    fields[2], phi, s.nx, s.ny, s.nz, 0, float(scale/s.hz), fields[2], s.nx, s.ny, s.nz);
}
/* VELOCITY RECOVERY: first remove divergence; then determine extra wall impulses; then project those impulses consistently. Final velocity is impulse minus the combined potential gradient. Two Poisson solves are intentional. */
static void project(Solver& s,float* momentum[3],float* velocity[3],float* phi,double dt) {
    boundary_flux_difference_kernel<<<blocks1(s.ny*s.nz),256>>>(
    momentum[0], s.work, s.nx, s.ny, s.nz);
    double shift=reduce(s,s.work,1LL*s.ny*s.nz,3)/(s.ny*s.nz);
    balance_outlet_kernel<<<blocks1(s.ny*s.nz),256>>>(
    momentum[0], s.nx, s.ny, s.nz, float(shift));
    divergence_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    momentum[0], momentum[1], momentum[2], s.face_mask[0], s.face_mask[1], s.face_mask[2], s.nx, s.ny, s.nz,
    float(1/s.h), float(1/s.hz), 1, s.rhs, s.nx, s.ny, s.nz);
    poisson(s,s.rhs,phi);
    dim3 wall_threads(16,16),wall_grid((s.wall_count+15)/16,(s.nz+15)/16);
    collect_wall_velocity_kernel<<<wall_grid,wall_threads>>>(
    momentum[0], momentum[1], phi, s.wall_face, s.wall_minus, s.wall_plus, s.wall_data, s.wall_count,
    s.wall_x_count, s.nx, s.ny, s.nz, float(s.h));
    check_fft(cufftExecC2C(s.wall_fft,s.wall_data,s.wall_data,CUFFT_FORWARD));
    wall_matrix_solve_kernel<<<wall_grid,wall_threads>>>(
    s.wall_data, s.wall_force, s.wall_inverse, s.wall_count, s.nz);
    if(dt>0) {
        std::vector<cufftComplex> zero(s.wall_count);
        check(cudaMemcpy(zero.data(),s.wall_force,s.wall_count*sizeof(cufftComplex),cudaMemcpyDeviceToHost));
        double sumx=0,sumy=0;
        for(int j=0;j<s.wall_count;j++)(j<s.wall_x_count?sumx:sumy)+=zero[j].x;
        double coefficient=-2*s.h*s.h/(dt*s.c.inlet*s.c.inlet*s.c.diameter*s.nz);
        s.cd=coefficient*sumx;
        s.cl=coefficient*sumy;
    }
    check_fft(cufftExecC2C(s.wall_fft,s.wall_force,s.wall_data,CUFFT_INVERSE));
    check(cudaMemset(s.rhs,0,s.cells*sizeof(float)));
    apply_wall_force_kernel<<<wall_grid,wall_threads>>>(
    momentum[0], momentum[1], s.rhs, s.wall_data, s.wall_face, s.wall_minus, s.wall_plus, s.wall_count,
    s.wall_x_count, s.nx, s.ny, s.nz, float(s.h));
    poisson(s,s.rhs,s.correction);
    sum_phi_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    phi, s.correction, s.nx, s.ny, s.nz);
    for(int component=0;component<3;component++)copy_device(velocity[component],momentum[component],s.faces[component]);
    add_gradient(s,velocity,phi,-1);
}

// ---------------------------------------------------------------------------
// MOMENTUM STAGES: one clearly named host function per numerical operation.
// Host functions launch CUDA kernels; the array arithmetic runs on the GPU.
// ---------------------------------------------------------------------------
/* OLD-TIME DERIVATIVES: launch face-to-centre averaging and velocity-gradient kernels. Launch WALE if enabled. All derivatives and viscosity calculations happen in GPU kernels. */
static void compute_gradients_and_wale(Solver& s) {
    cell_velocity_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.velocity[0], s.velocity[1], s.velocity[2], s.nx, s.ny, s.nz, s.center[0], s.center[1], s.center[2], s.nx,
    s.ny, s.nz);
    velocity_gradients_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.velocity[0], s.velocity[1], s.velocity[2], s.center[0], s.center[1], s.center[2], s.nx, s.ny, s.nz,
    float(s.h), float(s.hz), s.gradient[0], s.gradient[1], s.gradient[2], s.gradient[3], s.gradient[4],
    s.gradient[5], s.gradient[6], s.gradient[7], s.gradient[8], s.nx, s.ny, s.nz);
    if(s.c.high_order && s.c.mode==0) {
        for(int direction=0;direction<3;direction++)for(int component=0;component<3;component++)if(direction!=component) {
            int axis=2-direction;
            higher_order_derivative_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.center[component], s.center[component], s.fluid, s.nx, s.ny, s.nz, axis, float(direction==2?s.hz:s.h),
    s.gradient[3*direction+component], s.nx, s.ny, s.nz);
        }
    }
    if(s.c.les) {
        double delta=std::cbrt(s.h*s.h*s.hz);
        wale_viscosity_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.gradient[0], s.gradient[4], s.gradient[8], s.fluid, s.center[0], s.center[1], s.center[2], s.fluid, s.nx,
    s.ny, s.nz, float(s.h), float(s.hz), float(s.c.wale_cw*s.c.wale_cw*delta*delta), s.nut, s.nx, s.ny, s.nz);
    } else check(cudaMemset(s.nut,0,s.cells*sizeof(float)));
}
__global__ void energy_kernel(const float* u,const float* v,const float* w,const float* fluid,float* energy,int nx,int ny,int nz) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx||y>=ny||z>=nz)return;
    int i=(z*ny+y)*nx+x,iu=(z*ny+y)*(nx+1)+x;
    float uc=0.5f*(u[iu]+u[iu+1]),vc=0.5f*(v[i]+v[(z*ny+(y+1)%ny)*nx+x]),wc=0.5f*(w[i]+w[(((z+1)%nz)*ny+y)*nx+x]);
    energy[i]=0.5f*(uc*uc+vc*vc+wc*wc)*fluid[y*nx+x];
}
struct Metrics {double peak=0,energy=0,cfl_rate=0;
};
static Metrics state_metrics(Solver& s,float* fields[3]) {
    Metrics result;
    double maxima[3];
    for(int component=0;component<3;component++)maxima[component]=reduce(s,fields[component],s.faces[component],0);
    result.peak=max(maxima[0],max(maxima[1],maxima[2]));
    result.cfl_rate=(maxima[0]+maxima[1])/s.h+maxima[2]/s.hz;
    energy_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    fields[0], fields[1], fields[2], s.fluid, s.work, s.nx, s.ny, s.nz);
    result.energy=reduce(s,s.work,s.cells,3)/s.fluid_cells;
    return result;
}
static double minimum_determinant(Solver& s,double dt) {
    stretch_determinant_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.gradient[0], s.gradient[1], s.gradient[2], s.gradient[3], s.gradient[4], s.gradient[5], s.gradient[6],
    s.gradient[7], s.gradient[8], s.fluid, float(s.c.stretch_alpha*dt), s.work, s.nx, s.ny, s.nz);
    return reduce(s,s.work,s.cells,2);
}
/* HOST DECISION USING GPU REDUCTIONS: read a few maxima/minima, then select dt from the configured bounds. This function changes the timestep scalar, not the velocity/impulse arrays. */
static double choose_timestep(Solver& s,const Metrics& old,double retry_cap) {
    gradient_norm_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.gradient[0], s.gradient[1], s.gradient[2], s.gradient[3], s.gradient[4], s.gradient[5], s.gradient[6],
    s.gradient[7], s.gradient[8], s.fluid, s.work, s.nx, s.ny, s.nz);
    s.grad_max=reduce(s,s.work,s.cells,1);
    double nut_max=reduce(s,s.nut,s.cells,1);
    double alpha=s.c.mode==0?s.c.stretch_alpha:0;
    double sgs_rate=2*nut_max*(2/(s.h*s.h)+1/(s.hz*s.hz));
    if(!std::isfinite(s.grad_max)||!std::isfinite(sgs_rate))throw std::runtime_error("Nonfinite gradients or WALE viscosity");
    double dt=s.c.adaptive?std::numeric_limits<double>::infinity():s.c.dt;
    if(s.c.dt_max>0)dt=min(dt,s.c.dt_max);
    if(retry_cap>0)dt=min(dt,retry_cap);
    if(s.c.adaptive) {
        if(s.c.les && sgs_rate>0)dt=min(dt,s.c.dt_safety*s.c.diffusion_limit/sgs_rate);
        if(alpha*s.grad_max>0)dt=min(dt,s.c.dt_safety*s.c.gradient_limit/(alpha*s.grad_max));
        if(s.c.adapt_cfl)dt=min(dt,s.c.dt_safety*s.c.cfl_target/max(old.cfl_rate,1.e-300));
    }
    if(!std::isfinite(dt)||dt<=0||s.time+dt==s.time)throw std::runtime_error("Timestep collapsed or lacks a finite cap");
    if((s.c.les && sgs_rate*dt>s.c.diffusion_limit)||alpha*s.grad_max*dt>=s.c.gradient_limit)
        throw std::runtime_error("Fixed dt exceeds explicit-SGS or stretching bound");
    s.determinant=alpha>0?minimum_determinant(s,dt):1;
    if(!std::isfinite(s.determinant))throw std::runtime_error("Nonfinite stretching determinant");
    if(alpha>0 && s.determinant<=s.c.determinant_limit) {
        if(!s.c.adaptive)throw std::runtime_error("Stretching matrix is nearly singular at fixed dt");
        double hi=dt,lo=0.5*dt;
        while(minimum_determinant(s,lo)<=s.c.determinant_limit) {
            hi=lo;
            lo*=0.5;
            if(lo<=0||s.time+lo==s.time)throw std::runtime_error("Determinant timestep collapsed");
        }
        for(int i=0;i<16;i++) {double mid=0.5*(lo+hi);
        if(minimum_determinant(s,mid)>s.c.determinant_limit)lo=mid;
        else hi=mid;
        }
        dt=s.c.dt_safety*lo;
        s.determinant=minimum_determinant(s,dt);
    }
    s.sgs_number=dt*sgs_rate;
    return dt;
}
/* ADVECTION: the component loop means do x, then y, then z. For each component launch gauge reset, forward RK4 tracing/interpolation, and optionally the bounded backward correction. */
static void semi_lagrangian_transport(Solver& s,double dt,double r) {
    for(int component=0;component<3;component++) {
        int sx=s.nx+(component==0);
        dim3 grid=grid3(sx,s.ny,s.nz);
        float oz=component==2?0.0f:0.5f,oy=component==1?0.0f:0.5f,ox=component==0?0.0f:0.5f;
        // First reset gauge, then advect the reset impulse. Reuse scratch arrays
        // component by component instead of allocating three full sets of bounds.
        gauge_reset_kernel<<<grid,threads3>>>(
    s.impulse[component], s.velocity[component], float(r), s.work, sx, s.ny, s.nz);
        advect_forward_kernel<<<grid,threads3>>>(
    s.work, s.velocity[0], s.velocity[1], s.velocity[2], s.nx, s.ny, s.nz, sx, oz, oy, ox, float(dt/(2*s.h)),
    float(dt/(2*s.hz)), float(dt/s.h), float(dt/s.hz), float(s.c.inlet), component, s.c.maccormack, s.forward,
    s.lower, s.upper, sx, s.ny, s.nz);
        if(s.c.maccormack) {
            advect_correct_kernel<<<grid,threads3>>>(
    s.work, s.work, s.velocity[0], s.velocity[1], s.velocity[2], s.nx, s.ny, s.nz, sx, oz, oy, ox,
    float(-dt/(2*s.h)), float(-dt/(2*s.hz)), float(-dt/s.h), float(-dt/s.hz), float(s.c.inlet), s.forward,
    s.lower, s.upper, component, float(s.c.correction_blend), s.trial_impulse[component], sx, s.ny, s.nz);
        } else copy_device(s.trial_impulse[component],s.forward,s.faces[component]);
    }
}
/* IMPLICIT STRETCH: the kernel solves one 3x3 system per cell, returning the impulse change. A second kernel averages that change back to MAC faces. This is separate from molecular diffusion. */
static void stretch_impulse(Solver& s,double dt) {
    if(s.c.mode!=0 || s.c.stretch_alpha==0)return;
    stretch_impulse_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.gradient[0], s.gradient[1], s.gradient[2], s.gradient[3], s.gradient[4], s.gradient[5], s.gradient[6],
    s.gradient[7], s.gradient[8], s.fluid, float(s.c.stretch_alpha*dt), s.trial_impulse[0], s.trial_impulse[1],
    s.trial_impulse[2], s.nx, s.ny, s.nz, s.center[0], s.center[1], s.center[2], s.nx, s.ny, s.nz);
    for(int component=0;component<3;component++) {
        int sx=s.nx+(component==0);
        add_stretch_to_faces_kernel<<<grid3(sx,s.ny,s.nz),threads3>>>(
    s.trial_impulse[component], s.center[component], s.nx, s.ny, s.nz, 2-component, s.trial_impulse[component],
    sx, s.ny, s.nz);
    }
}
/* IMPLICIT VISCOSITY: preserve a fixed RHS, launch red then black sweeps, calculate the true residual, and repeat. The impulse update is performed in diffusion_sweep_kernel, not as a host-side addition. */
static void molecular_diffusion(Solver& s,double dt) {
    s.sweeps=0;
    for(int component=0;component<3;component++) {
        int sx=s.nx+(component==0),sign=component==0?1:-1;
        float* field=s.trial_impulse[component];
        copy_device(s.work,field,s.faces[component]);
         // fixed RHS
        double norm=max(1.0,reduce(s,s.work,s.faces[component],0));
        bool converged=false;
        for(int sweep=1;sweep<=s.c.be_max_sweeps;sweep++) {
            for(int color=0;color<2;color++) {
                diffusion_sweep_kernel<<<grid3(sx,s.ny,s.nz),threads3>>>(
    field, field, s.work, sx, s.ny, s.nz, float(s.nu*dt), float(1/(s.h*s.h)), float(1/(s.hz*s.hz)), sign, color,
    field, sx, s.ny, s.nz);
            }
            if(sweep%s.c.be_check_every==0 || sweep==s.c.be_max_sweeps) {
                diffusion_residual_kernel<<<grid3(sx,s.ny,s.nz),threads3>>>(
    field, s.work, field, sx, s.ny, s.nz, float(s.nu*dt), float(1/(s.h*s.h)), float(1/(s.hz*s.hz)), sign,
    s.forward, sx, s.ny, s.nz);
                double residual=reduce(s,s.forward,s.faces[component],1)/norm;
                if(residual<=s.c.be_tolerance){converged=true;
                s.sweeps=max(s.sweeps,sweep);
                break;
                }
            }
        }
        if(!converged)throw std::runtime_error("Backward-Euler diffusion did not converge");
    }
}
/* EXPLICIT SGS FORCE: first launch kernels for the six stress components. Then launch kernels that add dt times their divergence to impulse. Molecular viscosity above is implicit; this SGS stage is explicit. */
static void add_wale_stresses(Solver& s,double dt) {
    if(!s.c.les)return;
    // Transport and diffusion no longer need their scratch arrays. Borrow them
    // for stresses; trial_velocity is also unused until the projection stage.
    // This saves six full 3-D allocations without changing the equations.
    s.stress[0]=s.lower;
    s.stress[1]=s.upper;
    s.stress[2]=s.forward;
    s.stress[3]=s.trial_velocity[0];
    s.stress[4]=s.trial_velocity[1];
    s.stress[5]=s.trial_velocity[2];
    normal_stress_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.nut, s.velocity[0], s.velocity[1], s.velocity[2], s.nx, s.ny, s.nz, float(s.h), float(s.hz), s.stress[0],
    s.stress[1], s.stress[2], s.nx, s.ny, s.nz);
    xy_xz_stress_kernel<<<grid3(s.nx+1,s.ny,s.nz),threads3>>>(
    s.velocity[0], s.velocity[1], s.velocity[2], s.nut, s.fluid, s.nx, s.ny, s.nz, float(s.h), float(s.hz),
    s.stress[3], s.stress[4], s.nx+1, s.ny, s.nz);
    yz_stress_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.velocity[1], s.velocity[2], s.nut, s.fluid, s.nx, s.ny, s.nz, float(s.h), float(s.hz), s.stress[5], s.nx,
    s.ny, s.nz);
    add_sgs_x_kernel<<<grid3(s.nx+1,s.ny,s.nz),threads3>>>(
    s.stress[0], s.stress[3], s.stress[4], s.nx, s.ny, s.nz, float(s.h), float(s.hz), s.trial_impulse[0],
    float(dt), s.trial_impulse[0], s.nx+1, s.ny, s.nz);
    for(int component=1;component<3;component++)add_sgs_yz_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.stress[component], s.stress[component==1?3:4], s.stress[5], s.nx, s.ny, s.nz, float(s.h), float(s.hz),
    component, s.trial_impulse[component], float(dt), s.trial_impulse[component], s.nx, s.ny, s.nz);
}
__global__ void inlet_outlet_kernel(float* field,const float* old,int component,int nx,int ny,int nz,float inlet,float courant) {
    int y=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
    if(y>=ny||z>=nz)return;
    int sx=nx+(component==0),row=(z*ny+y)*sx;
    if(component==0)field[row]=inlet;
    field[row+sx-1]=old[row+sx-1]-courant*(old[row+sx-1]-old[row+sx-2]);
}
// Optional Eulerian vorticity confinement uses the same old-time gradients.
__global__ void confinement_force_kernel(const float* g01,const float* g02,const float* g10,const float* g12,
    const float* g20,const float* g21,float* wx,float* wy,float* wz,float* magnitude,int nx,int ny,int nz) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx||y>=ny||z>=nz)return;
    int i=(z*ny+y)*nx+x;
    wx[i]=g12[i]-g21[i];
    wy[i]=g20[i]-g02[i];
    wz[i]=g01[i]-g10[i];
    magnitude[i]=sqrtf(wx[i]*wx[i]+wy[i]*wy[i]+wz[i]*wz[i]);
}
__global__ void confinement_cross_kernel(const float* wx,const float* wy,const float* wz,const float* mag,
    float* fx,float* fy,float* fz,int nx,int ny,int nz,float h,float hz,float epsilon) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx||y>=ny||z>=nz)return;
    int i=(z*ny+y)*nx+x;
    int xm=x>0?i-1:i,xp=x+1<nx?i+1:i,ym=y>0?i-nx:i+(ny-1)*nx,yp=y+1<ny?i+nx:i-(ny-1)*nx,zm=z>0?i-nx*ny:i+(nz-1)*nx*ny,zp=z+1<nz?i+nx*ny:i-(nz-1)*nx*ny;
    float ex=(mag[xp]-mag[xm])/(2*h),ey=(mag[yp]-mag[ym])/(2*h),ez=(mag[zp]-mag[zm])/(2*hz);
    float norm=sqrtf(ex*ex+ey*ey+ez*ez)+1.e-20f;
    ex/=norm;
    ey/=norm;
    ez/=norm;
    fx[i]=epsilon*h*(ey*wz[i]-ez*wy[i]);
    fy[i]=epsilon*h*(ez*wx[i]-ex*wz[i]);
    fz[i]=epsilon*h*(ex*wy[i]-ey*wx[i]);
}
__global__ void scale_field_kernel(const float* input,float* output,long long n,float factor) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)output[i]=factor*input[i];
}
static void add_confinement(Solver& s,double dt) {
    if(s.c.mode!=2)return;
    confinement_force_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.gradient[1], s.gradient[2], s.gradient[3], s.gradient[5], s.gradient[6], s.gradient[7], s.center[0],
    s.center[1], s.center[2], s.work, s.nx, s.ny, s.nz);
    confinement_cross_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.center[0], s.center[1], s.center[2], s.work, s.gradient[0], s.gradient[1], s.gradient[2], s.nx, s.ny,
    s.nz, float(s.h), float(s.hz), float(s.c.vc_epsilon));
    // Forces were stored at cell centres. Multiply by dt before face averaging.
    for(int component=0;component<3;component++) {
        scale_field_kernel<<<blocks1(s.cells),256>>>(
    s.gradient[component], s.center[component], s.cells, float(dt));
        int sx=s.nx+(component==0);
        add_stretch_to_faces_kernel<<<grid3(sx,s.ny,s.nz),threads3>>>(
    s.trial_impulse[component], s.center[component], s.nx, s.ny, s.nz, 2-component, s.trial_impulse[component],
    sx, s.ny, s.nz);
    }
}

// ---------------------------------------------------------------------------
// OPTIONAL PIC/FLIP. The default configured solver uses semi-Lagrangian transport.
// Particles carry impulse; grid velocity moves them. All particle arithmetic is CUDA.
// ---------------------------------------------------------------------------
__device__ void move_outside_cylinder(float& y,float& x,float xc,float yc,float radius) {
    float dx=x-xc,dy=y-yc,r=sqrtf(dx*dx+dy*dy);
    if(r<1.e-7f){dx=1;
     r=1;
    }
    float actual=sqrtf((x-xc)*(x-xc)+(y-yc)*(y-yc));
    float factor=actual<radius+0.02f?(radius+0.02f)/r:1.0f;
    x=xc+dx*factor;
    y=yc+dy*factor;
}
__global__ void particle_move_kernel(const float* pz,const float* py,const float* px,const float* pmx,const float* pmy,const float* pmz,
    float* qz,float* qy,float* qx,float* qmx,float* qmy,float* qmz,const float* u,const float* v,const float* w,
    int n,int nx,int ny,int nz,float h,float hz,float dt,float r,float inlet,float xc,float yc,float radius) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    float z=pz[p],y=py[p],x=px[p],lo,hi;
    float a=fast_sample(u,z,y,x,nz,ny,nx+1,.5f,.5f,0,inlet,true,lo,hi,false);
    float b=fast_sample(v,z,y,x,nz,ny,nx,.5f,0,.5f,0,true,lo,hi,false);
    float c=fast_sample(w,z,y,x,nz,ny,nx,0,.5f,.5f,0,true,lo,hi,false);
    float zs,ys,xs;
    // fast_depart traces backward. Negative dt therefore traces the particle forward.
    fast_depart(u,v,w,z,y,x,nz,ny,nx,-dt/(2*h),-dt/(2*hz),-dt/h,-dt/hz,inlet,zs,ys,xs);
    zs=fmodf(fmodf(zs,nz)+nz,nz);
    ys=fmodf(fmodf(ys,ny)+ny,ny);
    bool recycled=xs<0||xs>=nx;
    if(xs<0)xs=.25f;
    else if(xs>=nx)xs-=nx;
    move_outside_cylinder(ys,xs,xc,yc,radius);
    ys=fmodf(fmodf(ys,ny)+ny,ny);
    qz[p]=zs;
    qy[p]=ys;
    qx[p]=xs;
    qmx[p]=recycled?inlet:r*pmx[p]+(1-r)*a;
    qmy[p]=recycled?0:r*pmy[p]+(1-r)*b;
    qmz[p]=recycled?0:r*pmz[p]+(1-r)*c;
}
__global__ void particle_deposit_kernel(const float* pz,const float* py,const float* px,const float* values,
    float* sum,float* weight,const float* mask,int n,int component,int nx,int ny,int nz) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    float ox=component==0?0:.5f,oy=component==1?0:.5f,oz=component==2?0:.5f;
    int sx=nx+(component==0);
    float x=fminf(fmaxf(px[p]-ox,0),float(sx-1)),y=py[p]-oy,z=pz[p]-oz;
    int x0=(int)floorf(x),yb=(int)floorf(y),zb=(int)floorf(z);
    float wx=x-x0,wy=y-yb,wz=z-zb;
    int x1=min(x0+1,sx-1),y0=(yb%ny+ny)%ny,z0=(zb%nz+nz)%nz;
    // Three short loops deposit to the eight nearby grid corners.
    for(int dz=0;dz<2;dz++)for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++) {
        int ix=dx?x1:x0,iy=(y0+dy)%ny,iz=(z0+dz)%nz;
        float w=(dx?wx:1-wx)*(dy?wy:1-wy)*(dz?wz:1-wz);
        int face=iy*sx+ix,index=(iz*ny+iy)*sx+ix;
        if(w>0 && mask[face]>0.5f){atomicAdd(sum+index,w*values[p]);
        atomicAdd(weight+index,w);
        }
    }
}
__global__ void particle_normalize_kernel(float* sum,const float* weights,const float* fallback,long long n) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)sum[i]=weights[i]>1.e-6f?sum[i]/weights[i]:fallback[i];
}
__global__ void particle_update_kernel(float* particle_m,const float* pz,const float* py,const float* px,
    const float* after,const float* before,int n,int component,int nx,int ny,int nz,float flip,float inlet) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    float z=pz[p],y=py[p],x=px[p],lo,hi;
    int sx=nx+(component==0);
    float ox=component==0?0:.5f,oy=component==1?0:.5f,oz=component==2?0:.5f;
    float pic=fast_sample(after,z,y,x,nz,ny,sx,oz,oy,ox,component==0?inlet:0,true,lo,hi,false);
    float old=fast_sample(before,z,y,x,nz,ny,sx,oz,oy,ox,component==0?inlet:0,true,lo,hi,false);
    // delta interpolation is linear; after-before is equivalent inside the domain.
    particle_m[p]=flip*(particle_m[p]+pic-old)+(1-flip)*pic;
}
static void particle_transport(Solver& s,double dt,double r) {
    particle_move_kernel<<<blocks1(s.particle_count),256>>>(
    s.particles[0], s.particles[1], s.particles[2], s.particles[3], s.particles[4], s.particles[5],
    s.trial_particles[0], s.trial_particles[1], s.trial_particles[2], s.trial_particles[3],
    s.trial_particles[4], s.trial_particles[5], s.velocity[0], s.velocity[1], s.velocity[2], s.particle_count,
    s.nx, s.ny, s.nz, float(s.h), float(s.hz), float(dt), float(r), float(s.c.inlet), float(s.c.body_x/s.h),
    .5f*s.ny, float(.5*s.c.diameter/s.h));
    for(int component=0;component<3;component++) {
        check(cudaMemset(s.trial_impulse[component],0,s.faces[component]*sizeof(float)));
        check(cudaMemset(s.particle_weight[component],0,s.faces[component]*sizeof(float)));
        particle_deposit_kernel<<<blocks1(s.particle_count),256>>>(
    s.trial_particles[0], s.trial_particles[1], s.trial_particles[2], s.trial_particles[3+component],
    s.trial_impulse[component], s.particle_weight[component], s.face_mask[component], s.particle_count,
    component, s.nx, s.ny, s.nz);
        particle_normalize_kernel<<<blocks1(s.faces[component]),256>>>(
    s.trial_impulse[component], s.particle_weight[component], s.impulse[component], s.faces[component]);
        copy_device(s.particle_before[component],s.trial_impulse[component],s.faces[component]);
    }
}
static void update_particle_impulse(Solver& s) {
    for(int component=0;component<3;component++)particle_update_kernel<<<blocks1(s.particle_count),256>>>(
    s.trial_particles[3+component], s.trial_particles[0], s.trial_particles[1], s.trial_particles[2],
    s.trial_impulse[component], s.particle_before[component], s.particle_count, component, s.nx, s.ny, s.nz,
    float(s.c.flip_fraction), float(s.c.inlet));
}

// ---------------------------------------------------------------------------
// ACCEPT/REJECT: a failed trial never replaces the accepted physical arrays.
// ---------------------------------------------------------------------------
__global__ void centreline_statistics_kernel(const float* u,double* sum,double* sum2,int nx,int ny,int nz,float yc,double weight) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=nx)return;
    float yf=yc-.5f;
    int j0=(int)floorf(yf);
    float wy=yf-j0;
    int j1=(j0+1)%ny;
    j0=(j0%ny+ny)%ny;
    double mean=0,second=0;
    for(int z=0;z<nz;z++) {
        int a=(z*ny+j0)*(nx+1)+x,b=(z*ny+j1)*(nx+1)+x;
        double value=.5*((1-wy)*(u[a]+u[a+1])+wy*(u[b]+u[b+1]));
        mean+=value;
        second+=value*value;
    }
    sum[x]+=weight*mean/nz;
    sum2[x]+=weight*second/nz;
}
/* START HERE: this is the complete time-step algorithm. Each numbered block launches the next numerical stage. After all checks pass, pointers are swapped to commit the candidate without copying full arrays. */
static void advance_one_step(Solver& s) {
    Metrics old;
    if(s.metrics_valid){old.peak=s.accepted_peak;
    old.energy=s.accepted_energy;
    old.cfl_rate=s.accepted_cfl_rate;
    }
    else old=state_metrics(s,s.velocity);
    if(!std::isfinite(old.peak)||!std::isfinite(old.energy)||!std::isfinite(old.cfl_rate))throw std::runtime_error("Accepted state is nonfinite");
    double retry_cap=0,dt=0;
    for(int attempt=0;attempt<=s.c.max_retries;attempt++) {
        // 1. Evaluate gradients and WALE from the OLD accepted velocity.
        compute_gradients_and_wale(s);
        dt=choose_timestep(s,old,retry_cap);
        double r=s.c.mode==0?s.c.retention:0;
        // 2. Reset gauge and transport impulse, with SL or the optional particles.
        if(s.c.particle_advection)particle_transport(s,dt,r);
        else semi_lagrangian_transport(s,dt,r);
        // 3. Local implicit stretching (only geometric impulse mode).
        stretch_impulse(s,dt);
        // 4. Backward-Euler molecular viscosity; explicit WALE stress force.
        molecular_diffusion(s,dt);
        add_wale_stresses(s,dt);
        add_confinement(s,dt);
        // 5. Impose boundary conditions on a velocity predictor. Add the retained
        // potential back afterwards so the stored impulse stays consistent.
        double old_phi_scale=s.c.mode==0?r:dt/max(s.last_dt,dt);
        add_gradient(s,s.trial_impulse,s.phi,-old_phi_scale);
        for(int component=0;component<3;component++)inlet_outlet_kernel<<<dim3((s.ny+15)/16,(s.nz+15)/16),dim3(16,16)>>>(
    s.trial_impulse[component], s.velocity[component], component, s.nx, s.ny, s.nz, float(s.c.inlet),
    float(s.c.inlet*dt/s.h));
        apply_body(s,s.trial_impulse);
        add_gradient(s,s.trial_impulse,s.phi,old_phi_scale);
        // 6. Two Poisson solves plus the normal-wall constraint solve.
        project(s,s.trial_impulse,s.trial_velocity,s.trial_phi,dt);
        Metrics next=state_metrics(s,s.trial_velocity);
        double speed_ratio=next.peak/max(old.peak,s.c.inlet),energy_ratio=next.energy/max(old.energy,.5*s.c.inlet*s.c.inlet);
        bool finite=std::isfinite(next.peak)&&std::isfinite(next.energy)&&std::isfinite(next.cfl_rate);
        bool cfl_ok=!s.c.adapt_cfl || dt*next.cfl_rate<s.c.cfl_target;
        if(finite&&cfl_ok&&speed_ratio<=s.c.speed_growth&&energy_ratio<=s.c.energy_growth) {
            // 7. Commit: swap pointers rather than copying full arrays.
            if(s.c.particle_advection)update_particle_impulse(s);
            for(int component=0;component<3;component++) {
                std::swap(s.velocity[component],s.trial_velocity[component]);
                std::swap(s.impulse[component],s.trial_impulse[component]);
            }
            if(s.c.particle_advection)for(int field=0;field<6;field++)std::swap(s.particles[field],s.trial_particles[field]);
            std::swap(s.phi,s.trial_phi);
            s.retries=attempt;
            s.accepted_peak=next.peak;
            s.accepted_energy=next.energy;
            s.accepted_cfl_rate=next.cfl_rate;
            s.metrics_valid=true;
            break;
        }
        if(!s.c.adaptive||attempt==s.c.max_retries)throw std::runtime_error("Trial rejected: velocity/energy growth or CFL bound failed; accepted state was not replaced");
        double severity=finite?max(speed_ratio/s.c.speed_growth,energy_ratio/s.c.energy_growth):std::numeric_limits<double>::infinity();
        double factor=max(.1,min(.8,.8/severity));
        if(s.c.adapt_cfl&&finite)factor=min(factor,s.c.dt_safety*s.c.cfl_target/max(dt*next.cfl_rate,1.e-300));
        retry_cap=dt*factor;
    }
    double old_time=s.time;
    s.time+=dt;
    s.last_dt=dt;
    s.steps++;
    int probe_x=int(std::nearbyint((s.c.body_x+s.c.probe_x*s.c.diameter)/s.h-.5));
    int probe_index=(s.nz/2*s.ny+s.ny/2)*s.nx+probe_x;
    float value;
    check(cudaMemcpy(&value,s.velocity[1]+probe_index,sizeof(float),cudaMemcpyDeviceToHost));
    s.probe_time.push_back(s.time);
    s.probe_value.push_back(value);
    if(s.time>s.c.stats_start) {
        double weight=s.time-max(old_time,s.c.stats_start);
        s.stats_weight+=weight;
        centreline_statistics_kernel<<<blocks1(s.nx),256>>>(
    s.velocity[0], s.line_sum, s.line_sum2, s.nx, s.ny, s.nz, .5f*s.ny, weight);
    }
    check(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// INITIALIZATION AND REPORTING. These functions are also in C++, not Python.
// ---------------------------------------------------------------------------
static double random_unit(std::mt19937& generator) {
    // NumPy RandomState's 53-bit uniform construction using the same MT19937 seed.
    unsigned long a=generator()>>5,b=generator()>>6;
    return (a*67108864.0+b)/9007199254740992.0;
}
__global__ void seed_particle_values_kernel(float* pz,float* py,float* px,float* mx,float* my,float* mz,
    const float* a,const float* b,const float* c,int n,int nx,int ny,int nz,float inlet,float xc,float yc,float radius) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    float z=pz[p],y=py[p],x=px[p];
    move_outside_cylinder(y,x,xc,yc,radius);
    y=fmodf(fmodf(y,ny)+ny,ny);
    py[p]=y;
    px[p]=x;
    float lo,hi;
    mx[p]=fast_sample(a,z,y,x,nz,ny,nx+1,.5f,.5f,0,inlet,true,lo,hi,false);
    my[p]=fast_sample(b,z,y,x,nz,ny,nx,.5f,0,.5f,0,true,lo,hi,false);
    mz[p]=fast_sample(c,z,y,x,nz,ny,nx,0,.5f,.5f,0,true,lo,hi,false);
}
static void seed_particles(Solver& s) {
    std::vector<float> fluid(s.nx*s.ny);
    check(cudaMemcpy(fluid.data(),s.fluid,fluid.size()*sizeof(float),cudaMemcpyDeviceToHost));
    const float offsets[8][3]={{.25f,.25f,.25f},{.75f,.75f,.75f},{.25f,.75f,.75f},{.75f,.25f,.25f},{.25f,.25f,.75f},{.75f,.75f,.25f},{.75f,.25f,.75f},{.25f,.75f,.25f}};
    long long particles=1LL*s.fluid_cells*s.c.particles_per_cell;
    if(particles>2147483647)throw std::runtime_error("Too many particles for 32-bit indexing");
    s.particle_count=int(particles);
    for(int field=0;field<6;field++){s.particles[field]=allocate<float>(s,particles);
    s.trial_particles[field]=allocate<float>(s,particles);
    }
    for(int component=0;component<3;component++){s.particle_before[component]=allocate<float>(s,s.faces[component]);
    s.particle_weight[component]=allocate<float>(s,s.faces[component]);
    }
    std::vector<float> z(particles),y(particles),x(particles);
    int index=0;
    for(int iz=0;iz<s.nz;iz++)for(int iy=0;iy<s.ny;iy++)for(int ix=0;ix<s.nx;ix++)if(fluid[iy*s.nx+ix]>0)
        for(int phase=0;phase<s.c.particles_per_cell;phase++){z[index]=iz+offsets[phase][0];
        y[index]=iy+offsets[phase][1];
        x[index]=ix+offsets[phase][2];
        index++;
        }
    check(cudaMemcpy(s.particles[0],z.data(),particles*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.particles[1],y.data(),particles*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.particles[2],x.data(),particles*sizeof(float),cudaMemcpyHostToDevice));
    seed_particle_values_kernel<<<blocks1(particles),256>>>(
    s.particles[0], s.particles[1], s.particles[2], s.particles[3], s.particles[4], s.particles[5],
    s.impulse[0], s.impulse[1], s.impulse[2], s.particle_count, s.nx, s.ny, s.nz, float(s.c.inlet),
    float(s.c.body_x/s.h), .5f*s.ny, float(.5*s.c.diameter/s.h));
}
/* SOLVER STARTUP: validate parameters, allocate fields, create geometry/wall matrices/FFT plans, initialise velocity and recover a divergence-free initial state. This runs once. */
static void initialize(Solver& s,const SolverConfig& config) {
    s.c=config;
    s.nx=config.nx;
    s.ny=config.ny;
    s.nz=config.nz;
    if(s.nx<4||s.ny<4||s.nz<4||s.ny%2||s.nz%2)throw std::runtime_error("Need nx>=4 and even ny,nz>=4 for periodic red-black sweeps");
    if(config.lx<=0||config.ly<=0||config.lz<=0||config.diameter<=0||config.inlet<=0||config.reynolds<=0||config.dt<=0)throw std::runtime_error("Lengths, inlet, Re and dt must be positive");
    s.h=config.lx/s.nx;
    s.hz=config.lz/s.nz;
    s.nu=config.inlet*config.diameter/config.reynolds;
    if(std::abs(config.ly/s.ny-s.h)>1.e-10*s.h)throw std::runtime_error("x and y spacings must be equal in this uniform-grid solver");
    if(config.mode<0||config.mode>2||config.retention<=0||config.retention>1||config.be_check_every<1||config.be_max_sweeps<1||config.max_retries<0||config.be_tolerance<=0)throw std::runtime_error("Invalid mode, gauge retention or convergence settings");
    if(config.correction_blend<0||config.correction_blend>1||config.flip_fraction<0||config.flip_fraction>1||config.dt_safety<=0||config.dt_safety>=1||config.cfl_target<=0||config.diffusion_limit<=0||config.gradient_limit<=0||config.gradient_limit>=1||config.determinant_limit<=0||config.wale_cw<=0)throw std::runtime_error("Invalid transport or stability settings");
    if(config.particle_advection && (config.mode!=0 || (config.particles_per_cell!=1&&config.particles_per_cell!=2&&config.particles_per_cell!=4&&config.particles_per_cell!=8)))throw std::runtime_error("PIC/FLIP needs impulse mode and 1,2,4 or 8 particles per cell");
    if(config.les && config.two_d)throw std::runtime_error("Disable WALE for the two-dimensional option");
    double R=.5*config.diameter;
    if(config.body_x-2*R<=s.h||config.body_x+2*R>=config.lx-s.h||config.ly/2-2*R<=s.h)throw std::runtime_error("Cylinder mirror stencil does not fit in domain");
    int probe_x=int(std::nearbyint((config.body_x+config.probe_x*config.diameter)/s.h-.5));
    if(probe_x<0||probe_x>=s.nx)throw std::runtime_error("Wake probe outside domain");
    s.cells=1LL*s.nx*s.ny*s.nz;
    for(int component=0;component<3;component++)s.faces[component]=1LL*(s.nx+(component==0))*s.ny*s.nz;
    if(s.faces[0]>2147483647)throw std::runtime_error("Grid too large for current 32-bit stencil indexing");
    for(int component=0;component<3;component++) {
        s.velocity[component]=allocate<float>(s,s.faces[0]);
        s.impulse[component]=allocate<float>(s,s.faces[component]);
        s.trial_velocity[component]=allocate<float>(s,s.faces[0]);
        s.trial_impulse[component]=allocate<float>(s,s.faces[component]);
        s.center[component]=allocate<float>(s,s.cells);
        s.face_mask[component]=allocate<float>(s,s.faces[component]/s.nz);
    }
    for(int i=0;i<9;i++)s.gradient[i]=allocate<float>(s,s.cells);
    s.phi=allocate<float>(s,s.cells);
    s.trial_phi=allocate<float>(s,s.cells);
    s.nut=allocate<float>(s,s.cells);
    s.fluid=allocate<float>(s,s.nx*s.ny);
    s.work=allocate<float>(s,s.faces[0]);
    s.forward=allocate<float>(s,s.faces[0]);
    s.lower=allocate<float>(s,s.faces[0]);
    s.upper=allocate<float>(s,s.faces[0]);
    s.rhs=allocate<float>(s,s.cells);
    s.correction=allocate<float>(s,s.cells);
    s.partial_a=allocate<double>(s,blocks1(s.faces[0]));
    s.partial_b=allocate<double>(s,blocks1(s.faces[0]));
    s.line_sum=allocate<double>(s,s.nx);
    s.line_sum2=allocate<double>(s,s.nx);
    s.fft_real=allocate<float>(s,2*s.cells);
    s.fft_spectrum=allocate<cufftComplex>(s,1LL*(s.nx+1)*s.ny*s.nz);
    make_geometry(s);
    build_wall_inverse(s);
    // Forward and inverse transforms execute sequentially, so they can share
    // one cuFFT workspace instead of holding two large workspaces in VRAM.
    check_fft(cufftCreate(&s.poisson_forward));
    check_fft(cufftCreate(&s.poisson_backward));
    check_fft(cufftSetAutoAllocation(s.poisson_forward,0));
    check_fft(cufftSetAutoAllocation(s.poisson_backward,0));
    size_t forward_bytes=0,backward_bytes=0;
    check_fft(cufftMakePlan3d(s.poisson_forward,s.nz,s.ny,2*s.nx,CUFFT_R2C,&forward_bytes));
    check_fft(cufftMakePlan3d(s.poisson_backward,s.nz,s.ny,2*s.nx,CUFFT_C2R,&backward_bytes));
    auto fft_workspace=allocate<unsigned char>(s,max(size_t(1),max(forward_bytes,backward_bytes)));
    check_fft(cufftSetWorkArea(s.poisson_forward,fft_workspace));
    check_fft(cufftSetWorkArea(s.poisson_backward,fft_workspace));
    std::vector<float> u(s.faces[0],float(config.inlet)),v(s.cells),w(s.cells);
    std::mt19937 generator((unsigned)config.seed);
    for(long long i=0;i<s.cells;i++)v[i]=float(config.perturbation*config.inlet*(2*random_unit(generator)-1));
    for(long long i=0;i<s.cells;i++)w[i]=float(config.perturbation*config.inlet*(2*random_unit(generator)-1));
    if(config.two_d)for(int z=0;z<s.nz;z++)for(int j=0;j<s.nx*s.ny;j++){v[z*s.nx*s.ny+j]=v[j];
    w[z*s.nx*s.ny+j]=0;
    }
    check(cudaMemcpy(s.impulse[0],u.data(),s.faces[0]*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.impulse[1],v.data(),s.cells*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.impulse[2],w.data(),s.cells*sizeof(float),cudaMemcpyHostToDevice));
    apply_body(s,s.impulse);
    project(s,s.impulse,s.velocity,s.phi,0);
    if(config.particle_advection)seed_particles(s);
    check(cudaDeviceSynchronize());
}
__global__ void squared_divergence_kernel(const float* divergence,const float* fluid,float* squares,int nx,int ny,int nz) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x<nx&&y<ny&&z<nz){int i=(z*ny+y)*nx+x;
    squares[i]=divergence[i]*divergence[i]*fluid[y*nx+x];
    }
}
__global__ void wall_slip_kernel(const float* u,const float* v,const int* faces,float* values,int count,int xcount,int nx,int ny,int nz) {
    int wall=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
    if(wall>=count||z>=nz)return;
    values[z*count+wall]=wall<xcount?fabsf(u[z*ny*(nx+1)+faces[wall]]):fabsf(v[z*ny*nx+faces[wall]]);
}
/* DIAGNOSTICS ONLY: evaluate norms and transfer small scalar results to the host. No accepted momentum field is advanced by this reporting function. */
static void report(Solver& s,SolverReport& r) {
    r={};
    Metrics m=state_metrics(s,s.velocity);
    compute_gradients_and_wale(s);
    r.steps=s.steps;
    r.time=s.time;
    r.dt=s.last_dt;
    r.cfl=s.last_dt*m.cfl_rate;
    r.energy=m.energy;
    r.peak_velocity=m.peak;
    for(int component=0;component<3;component++)r.peak_impulse=max(r.peak_impulse,reduce(s,s.impulse[component],s.faces[component],0));
    divergence_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.velocity[0], s.velocity[1], s.velocity[2], s.face_mask[0], s.face_mask[1], s.face_mask[2], s.nx, s.ny,
    s.nz, float(1/s.h), float(1/s.hz), 0, s.rhs, s.nx, s.ny, s.nz);
    squared_divergence_kernel<<<grid3(s.nx,s.ny,s.nz),threads3>>>(
    s.rhs, s.fluid, s.work, s.nx, s.ny, s.nz);
    r.divergence_rms=std::sqrt(reduce(s,s.work,s.cells,3)/s.fluid_cells);
    wall_slip_kernel<<<dim3((s.wall_count+15)/16,(s.nz+15)/16),dim3(16,16)>>>(
    s.velocity[0], s.velocity[1], s.wall_face, s.work, s.wall_count, s.wall_x_count, s.nx, s.ny, s.nz);
    r.wall_slip=reduce(s,s.work,1LL*s.wall_count*s.nz,1)/s.c.inlet;
    r.nut_ratio_mean=reduce(s,s.nut,s.cells,3)/(s.fluid_cells*s.nu);
    r.nut_ratio_max=reduce(s,s.nut,s.cells,1)/s.nu;
    r.min_determinant=s.determinant;
    r.gradient_dt=s.grad_max*s.last_dt;
    r.sgs_number=s.sgs_number;
    r.cd_wall=s.cd;
    r.cl_wall=s.cl;
    r.probe_velocity=s.probe_value.empty()?0:s.probe_value.back();
    r.averaging_duration=s.stats_weight;
    r.diffusion_sweeps=s.sweeps;
    r.retries=s.retries;
    check(cudaGetLastError());
}
// Q = 0.5*(||rotation||^2-||strain||^2) = -0.5*trace(G*G).
// This is a visualization query only; it never changes the accepted flow.
__device__ void q_cell_velocity(const float* u,const float* v,const float* w,int nx,int ny,int nz,int x,int y,int z,float& a,float& b,float& c) {
    x=max(0,min(nx-1,x));
    y=(y+ny)%ny;
    z=(z+nz)%nz;
    int i=(z*ny+y)*nx+x,iu=(z*ny+y)*(nx+1)+x;
    a=.5f*(u[iu]+u[iu+1]);
    b=.5f*(v[i]+v[(z*ny+(y+1)%ny)*nx+x]);
    c=.5f*(w[i]+w[(((z+1)%nz)*ny+y)*nx+x]);
}
__global__ void q_volume_kernel(const float* u,const float* v,const float* w,const float* fluid,float* q,float* speed,
    int nx,int ny,int nz,int x0,int y0,int width,int height,float h,float hz,float diameter,float inlet) {
    int ix=blockIdx.x*blockDim.x+threadIdx.x,iy=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(ix>=width||iy>=height||z>=nz)return;
    int x=x0+ix,y=y0+iy,i=(z*height+iy)*width+ix;
    if(fluid[y*nx+x]==0){q[i]=-1.e30f;
    speed[i]=0;
    return;
    }
    float a,b,c;
     q_cell_velocity(u,v,w,nx,ny,nz,x,y,z,a,b,c);
    speed[i]=sqrtf(a*a+b*b+c*c)/inlet;
    float axp,bxp,cxp,axm,bxm,cxm,ayp,byp,cyp,aym,bym,cym,azp,bzp,czp,azm,bzm,czm;
    int xp=min(nx-1,x+1),xm=max(0,x-1);
    q_cell_velocity(u,v,w,nx,ny,nz,xp,y,z,axp,bxp,cxp);
    q_cell_velocity(u,v,w,nx,ny,nz,xm,y,z,axm,bxm,cxm);
    q_cell_velocity(u,v,w,nx,ny,nz,x,y+1,z,ayp,byp,cyp);
    q_cell_velocity(u,v,w,nx,ny,nz,x,y-1,z,aym,bym,cym);
    q_cell_velocity(u,v,w,nx,ny,nz,x,y,z+1,azp,bzp,czp);
    q_cell_velocity(u,v,w,nx,ny,nz,x,y,z-1,azm,bzm,czm);
    float ixh=1/(max(1,xp-xm)*h),iyh=.5f/h,izh=.5f/hz;
    float g00=(axp-axm)*ixh,g01=(bxp-bxm)*ixh,g02=(cxp-cxm)*ixh;
    float g10=(ayp-aym)*iyh,g11=(byp-bym)*iyh,g12=(cyp-cym)*iyh;
    float g20=(azp-azm)*izh,g21=(bzp-bzm)*izh,g22=(czp-czm)*izh;
    q[i]=-.5f*(g00*g00+g11*g11+g22*g22+2*(g01*g10+g02*g20+g12*g21))*diameter*diameter/(inlet*inlet);
}

// Each exported function catches C++ exceptions and returns an error code.
// Exceptions must never cross a ctypes boundary into Python.
SOLVER_EXPORT int solver_create(const SolverConfig* config,void** handle) {
    try {if(!config||!handle)throw std::runtime_error("Null create argument");
    *handle=nullptr;
        std::unique_ptr<Solver> s(new Solver);
        initialize(*s,*config);
        *handle=s.release();
        return 0;
    }catch(const std::exception& e){last_error=e.what();
    return -1;
    }
}
SOLVER_EXPORT int solver_advance(void* handle,int count,SolverReport* result) {
    try {if(!handle||!result||count<0)throw std::runtime_error("Invalid advance arguments");
    auto& s=*(Solver*)handle;
        for(int step=0;step<count;step++)advance_one_step(s);
        if(count==0){report(s,*result);
        return 0;
        }
        // The per-step report is cheap. Full diagnostics are queried separately.
        *result={};
        result->steps=s.steps;
        result->time=s.time;
        result->dt=s.last_dt;
        result->cfl=s.last_dt*s.accepted_cfl_rate;
        result->energy=s.accepted_energy;
        result->peak_velocity=s.accepted_peak;
        result->peak_impulse=NAN;
        result->divergence_rms=NAN;
        result->wall_slip=NAN;
        result->nut_ratio_mean=NAN;
        result->nut_ratio_max=NAN;
        result->min_determinant=s.determinant;
        result->gradient_dt=s.grad_max*s.last_dt;
        result->sgs_number=s.sgs_number;
        result->cd_wall=s.cd;
        result->cl_wall=s.cl;
        result->probe_velocity=s.probe_value.back();
        result->averaging_duration=s.stats_weight;
        result->diffusion_sweeps=s.sweeps;
        result->retries=s.retries;
        return 0;
    }catch(const std::exception& e){last_error=e.what();
    return -1;
    }
}
SOLVER_EXPORT int solver_report(void* handle,SolverReport* result) {
    try {if(!handle||!result)throw std::runtime_error("Null report argument");
    report(*(Solver*)handle,*result);
    return 0;
    }catch(const std::exception& e){last_error=e.what();
    return -1;
    }
}
SOLVER_EXPORT int solver_copy_field(void* handle,int field,float* destination,long long count) {
    try {if(!handle||!destination||field<0||field>7)throw std::runtime_error("Invalid field request");
    auto& s=*(Solver*)handle;
        float* pointer=field<3?s.velocity[field]:(field<6?s.impulse[field-3]:(field==6?s.phi:s.nut));
        long long expected=field<6?s.faces[field%3]:s.cells;
        if(count!=expected)throw std::runtime_error("Field buffer has wrong size");
        if(field==7)compute_gradients_and_wale(s);
        check(cudaMemcpy(destination,pointer,count*sizeof(float),cudaMemcpyDeviceToHost));
        return 0;
    }catch(const std::exception& e){last_error=e.what();
    return -1;
    }
}
SOLVER_EXPORT int solver_copy_line(void* handle,double* mean,double* rms,int count) {
    try {if(!handle||!mean||!rms)throw std::runtime_error("Null line argument");
    auto& s=*(Solver*)handle;
    if(count!=s.nx)throw std::runtime_error("Line buffer has wrong size");
        std::vector<double> sums(count),squares(count);
        check(cudaMemcpy(sums.data(),s.line_sum,count*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(squares.data(),s.line_sum2,count*sizeof(double),cudaMemcpyDeviceToHost));
        for(int x=0;x<count;x++){double m=s.stats_weight>0?sums[x]/s.stats_weight:NAN;
        mean[x]=m/s.c.inlet;
        rms[x]=s.stats_weight>0?std::sqrt(max(0.0,squares[x]/s.stats_weight-m*m))/s.c.inlet:NAN;
        }return 0;
    }catch(const std::exception& e){last_error=e.what();
    return -1;
    }
}
SOLVER_EXPORT int solver_q_volume(void* handle,int x0,int y0,int width,int height,float* q,float* speed,long long count) {
    try {if(!handle||!q||!speed)throw std::runtime_error("Null Q argument");
    auto& s=*(Solver*)handle;
        if(x0<0||y0<0||width<1||height<1||x0+width>s.nx||y0+height>s.ny||count!=1LL*width*height*s.nz)throw std::runtime_error("Q region/buffer mismatch");
        // Reuse FFT real storage when the crop fits. No extra persistent Q volumes.
        float* dq=s.fft_real;
        float* ds=s.fft_real+s.cells;
        q_volume_kernel<<<grid3(width,height,s.nz),threads3>>>(
    s.velocity[0], s.velocity[1], s.velocity[2], s.fluid, dq, ds, s.nx, s.ny, s.nz, x0, y0, width, height,
    float(s.h), float(s.hz), float(s.c.diameter), float(s.c.inlet));
        check(cudaMemcpy(q,dq,count*sizeof(float),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(speed,ds,count*sizeof(float),cudaMemcpyDeviceToHost));
        return 0;
    }catch(const std::exception& e){last_error=e.what();
    return -1;
    }
}
SOLVER_EXPORT int solver_destroy(void* handle) {
    try {delete (Solver*)handle;
    return 0;
    }catch(const std::exception& e){last_error=e.what();
    return -1;
    }
}
SOLVER_EXPORT const char* solver_last_error() {return last_error.c_str();
}
SOLVER_EXPORT int solver_copy_crop(void* handle,int field,int x0,int y0,int width,int height,float* destination,long long count) {
    try {if(!handle||!destination||field<0||field>7)throw std::runtime_error("Invalid crop request");
    auto& s=*(Solver*)handle;
        int sx=s.nx+(field==0||field==3);
        if(x0<0||y0<0||width<1||height<1||x0+width>sx||y0+height>s.ny||count!=1LL*width*height*s.nz)throw std::runtime_error("Crop region/buffer mismatch");
        float* pointer=field<3?s.velocity[field]:(field<6?s.impulse[field-3]:(field==6?s.phi:s.nut));
        if(field==7)compute_gradients_and_wale(s);
        for(int z=0;z<s.nz;z++)check(cudaMemcpy2D(destination+z*width*height,width*sizeof(float),pointer+(z*s.ny+y0)*sx+x0,sx*sizeof(float),width*sizeof(float),height,cudaMemcpyDeviceToHost));
        return 0;
    }catch(const std::exception& e){last_error=e.what();
    return -1;
    }
}
