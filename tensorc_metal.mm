#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <tensorc/ndtensor.h>
#include "tensorc_device_internal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

@interface TCMetalBuffer : NSObject
@property(nonatomic) size_t count;
@property(nonatomic) float *pointer;
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, strong) id<MTLCommandBuffer> lastCommand;
@end
@implementation TCMetalBuffer
@end

typedef struct {
    uint32_t batch, channels, height, width, filters, kernel_h, kernel_w;
    uint32_t out_h, out_w, stride_h, stride_w, pad_h, pad_w;
} TCMetalConvDesc;
typedef struct { uint32_t batch, channels, height, width, pool_h, pool_w; }
    TCMetalPoolDesc;

static id<MTLDevice> tc_metal_device;
static id<MTLCommandQueue> tc_metal_queue;
static NSMutableDictionary<NSValue *, TCMetalBuffer *> *tc_metal_buffers;
static NSMutableArray<TCMetalBuffer *> *tc_metal_pool;
static id<MTLLibrary> tc_metal_library;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *tc_metal_pipelines;
static size_t tc_metal_threshold = 16384;
static int tc_metal_state;
static int tc_metal_init(void);
static TCMetalBuffer *tc_metal_info(const float *pointer);
static id<MTLCommandBuffer> tc_metal_command(void);

static NSString *tc_metal_shader_source(void) {
    return @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct BDesc { int ndim; int a_ndim; int b_ndim; ulong out_shape[8]; ulong out_strides[8]; ulong a_shape[8]; ulong a_strides[8]; ulong b_shape[8]; ulong b_strides[8]; };\n"
           "inline ulong bindex(uint i, constant BDesc& d, bool left) { ulong ix=0; int shift=d.ndim-(left?d.a_ndim:d.b_ndim); for(int q=d.ndim-1;q>=0;--q){ ulong c=(i/d.out_strides[q])%d.out_shape[q]; int r=q-shift; if(r>=0 && (left?d.a_shape[r]:d.b_shape[r])!=1) ix+=c*(left?d.a_strides[r]:d.b_strides[r]); } return ix; }\n"
           "kernel void binary_f(device const float*a [[buffer(0)]],device const float*b [[buffer(1)]],device float*o [[buffer(2)]],constant BDesc&d [[buffer(3)]],constant uint&n [[buffer(4)]],constant uint&mul [[buffer(5)]],uint i [[thread_position_in_grid]]){if(i<n){ulong ia=bindex(i,d,true),ib=bindex(i,d,false);o[i]=mul?a[ia]*b[ib]:a[ia]+b[ib];}}\n"
           "kernel void binary_b(device const float*g [[buffer(0)]],device const float*a [[buffer(1)]],device const float*b [[buffer(2)]],device atomic_float*ga [[buffer(3)]],device atomic_float*gb [[buffer(4)]],constant BDesc&d [[buffer(5)]],constant uint&n [[buffer(6)]],constant uint&mul [[buffer(7)]],constant uint&wa [[buffer(8)]],constant uint&wb [[buffer(9)]],uint i [[thread_position_in_grid]]){if(i<n){ulong ia=bindex(i,d,true),ib=bindex(i,d,false);float v=g[i];if(wa)atomic_fetch_add_explicit(&ga[ia],mul?v*b[ib]:v,memory_order_relaxed);if(wb)atomic_fetch_add_explicit(&gb[ib],mul?v*a[ia]:v,memory_order_relaxed);}}\n"
           "kernel void relu_f(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint&n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<n)y[i]=max(x[i],0.0f);}\n"
           "kernel void relu_b(device const float*x [[buffer(0)]],device const float*g [[buffer(1)]],device atomic_float*gx [[buffer(2)]],constant uint&n [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<n)atomic_fetch_add_explicit(&gx[i],x[i]>0.0f?g[i]:0.0f,memory_order_relaxed);}\n"
           "kernel void softmax_f(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint&rows [[buffer(2)]],constant uint&cols [[buffer(3)]],uint r [[thread_position_in_grid]]){if(r<rows){float m=x[r*cols];for(uint j=1;j<cols;++j)m=max(m,x[r*cols+j]);float s=0;for(uint j=0;j<cols;++j){y[r*cols+j]=exp(x[r*cols+j]-m);s+=y[r*cols+j];}for(uint j=0;j<cols;++j)y[r*cols+j]/=s;}}\n"
           "kernel void softmax_b(device const float*y [[buffer(0)]],device const float*g [[buffer(1)]],device atomic_float*gx [[buffer(2)]],constant uint&rows [[buffer(3)]],constant uint&cols [[buffer(4)]],uint r [[thread_position_in_grid]]){if(r<rows){float d=0;for(uint j=0;j<cols;++j)d+=g[r*cols+j]*y[r*cols+j];for(uint j=0;j<cols;++j)atomic_fetch_add_explicit(&gx[r*cols+j],y[r*cols+j]*(g[r*cols+j]-d),memory_order_relaxed);}}\n"
           "kernel void ce_f(device const float*p [[buffer(0)]],device const float*t [[buffer(1)]],device float*out [[buffer(2)]],constant uint&n [[buffer(3)]],constant uint&rows [[buffer(4)]],uint i [[thread_position_in_grid]]){if(i==0){float loss=0;for(uint j=0;j<n;++j)loss-=t[j]*log(max(p[j],1e-7f));out[0]=loss/rows;}}\n"
           "kernel void ce_b(device const float*p [[buffer(0)]],device const float*t [[buffer(1)]],device const float*g [[buffer(2)]],device atomic_float*gp [[buffer(3)]],device atomic_float*gt [[buffer(4)]],constant uint&n [[buffer(5)]],constant uint&rows [[buffer(6)]],constant uint&wp [[buffer(7)]],constant uint&wt [[buffer(8)]],uint i [[thread_position_in_grid]]){if(i<n){if(wp)atomic_fetch_add_explicit(&gp[i],-g[0]*t[i]/max(p[i],1e-7f)/rows,memory_order_relaxed);if(wt)atomic_fetch_add_explicit(&gt[i],-g[0]*log(max(p[i],1e-7f))/rows,memory_order_relaxed);}}\n"
           "kernel void copy_f(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint&n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<n)y[i]=x[i];}\n"
           "kernel void accum_f(device const float*x [[buffer(0)]],device atomic_float*y [[buffer(1)]],constant uint&n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<n)atomic_fetch_add_explicit(&y[i],x[i],memory_order_relaxed);}\n"
           "kernel void zero_f(device float*x [[buffer(0)]],constant uint&n [[buffer(1)]],uint i [[thread_position_in_grid]]){if(i<n)x[i]=0.0f;}\n"
           "kernel void sgd_f(device float*p [[buffer(0)]],device const float*g [[buffer(1)]],constant uint&n [[buffer(2)]],constant float&lr [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<n)p[i]-=lr*g[i];}\n"
           "kernel void adam_f(device float*p [[buffer(0)]],device const float*g [[buffer(1)]],device float*m [[buffer(2)]],device float*v [[buffer(3)]],constant uint&n [[buffer(4)]],constant float&lr [[buffer(5)]],constant float&b1 [[buffer(6)]],constant float&b2 [[buffer(7)]],constant float&c1 [[buffer(8)]],constant float&c2 [[buffer(9)]],constant float&e [[buffer(10)]],uint i [[thread_position_in_grid]]){if(i<n){float gi=g[i];m[i]=b1*m[i]+(1.0f-b1)*gi;v[i]=b2*v[i]+(1.0f-b2)*gi*gi;p[i]-=lr*(m[i]/c1)/(sqrt(v[i]/c2)+e);}}\n"
           "kernel void matmul_b(device const float*a [[buffer(0)]],device const float*b [[buffer(1)]],device atomic_float*c [[buffer(2)]],constant uint&m [[buffer(3)]],constant uint&k [[buffer(4)]],constant uint&n [[buffer(5)]],constant uint&mode [[buffer(6)]],uint i [[thread_position_in_grid]]){uint rows=mode==1?m:k,cols=mode==1?k:n;if(i<rows*cols){uint r=i/cols,q=i%cols;float sum=0;if(mode==1){for(uint s=0;s<n;++s)sum+=a[r*n+s]*b[q*n+s];}else{for(uint s=0;s<m;++s)sum+=a[s*k+r]*b[s*n+q];}atomic_fetch_add_explicit(&c[i],sum,memory_order_relaxed);}}\n"
           "struct ConvDesc{uint batch,channels,height,width,filters,kh,kw,oh,ow,sh,sw,ph,pw;};\n"
           "kernel void conv_f(device const float*x [[buffer(0)]],device const float*w [[buffer(1)]],device float*y [[buffer(2)]],constant ConvDesc&d [[buffer(3)]],uint i [[thread_position_in_grid]]){uint total=d.batch*d.filters*d.oh*d.ow;if(i<total){uint ow=i%d.ow,oh=(i/d.ow)%d.oh,oc=(i/(d.ow*d.oh))%d.filters,n=i/(d.ow*d.oh*d.filters);float sum=0;for(uint c=0;c<d.channels;++c)for(uint kh=0;kh<d.kh;++kh)for(uint kw=0;kw<d.kw;++kw){int ih=int(oh*d.sh+kh)-int(d.ph),iw=int(ow*d.sw+kw)-int(d.pw);if(ih>=0&&iw>=0&&ih<int(d.height)&&iw<int(d.width)){uint xi=((n*d.channels+c)*d.height+uint(ih))*d.width+uint(iw);uint wi=((oc*d.channels+c)*d.kh+kh)*d.kw+kw;sum+=x[xi]*w[wi];}}y[i]=sum;}}\n"
           "kernel void conv_dx(device const float*x [[buffer(0)]],device const float*w [[buffer(1)]],device const float*gy [[buffer(2)]],device float*gx [[buffer(3)]],constant ConvDesc&d [[buffer(4)]],uint i [[thread_position_in_grid]]){uint total=d.batch*d.channels*d.height*d.width;if(i<total){uint iw=i%d.width,ih=(i/d.width)%d.height,c=(i/(d.width*d.height))%d.channels,n=i/(d.width*d.height*d.channels);float sum=0;for(uint oc=0;oc<d.filters;++oc)for(uint kh=0;kh<d.kh;++kh)for(uint kw=0;kw<d.kw;++kw){int oh=int(ih+d.ph)-int(kh),ow=int(iw+d.pw)-int(kw);if(oh>=0&&ow>=0&&oh%int(d.sh)==0&&ow%int(d.sw)==0){oh/=int(d.sh);ow/=int(d.sw);if(oh<int(d.oh)&&ow<int(d.ow)){uint yi=((n*d.filters+oc)*d.oh+uint(oh))*d.ow+uint(ow);uint wi=((oc*d.channels+c)*d.kh+kh)*d.kw+kw;sum+=gy[yi]*w[wi];}}}gx[i]+=sum;}}\n"
           "kernel void conv_dw(device const float*x [[buffer(0)]],device const float*gy [[buffer(1)]],device float*gw [[buffer(2)]],constant ConvDesc&d [[buffer(3)]],uint i [[thread_position_in_grid]]){uint total=d.filters*d.channels*d.kh*d.kw;if(i<total){uint kw=i%d.kw,kh=(i/d.kw)%d.kh,c=(i/(d.kw*d.kh))%d.channels,oc=i/(d.kw*d.kh*d.channels);float sum=0;for(uint n=0;n<d.batch;++n)for(uint oh=0;oh<d.oh;++oh)for(uint ow=0;ow<d.ow;++ow){int ih=int(oh*d.sh+kh)-int(d.ph),iw=int(ow*d.sw+kw)-int(d.pw);if(ih>=0&&iw>=0&&ih<int(d.height)&&iw<int(d.width)){uint xi=((n*d.channels+c)*d.height+uint(ih))*d.width+uint(iw);uint yi=((n*d.filters+oc)*d.oh+oh)*d.ow+ow;sum+=x[xi]*gy[yi];}}gw[i]+=sum;}}\n"
           "struct PoolDesc{uint batch,channels,height,width,pool_h,pool_w;};\n"
           "kernel void pool_f(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant PoolDesc&d [[buffer(2)]],uint i [[thread_position_in_grid]]){uint ohn=d.height/d.pool_h,own=d.width/d.pool_w,total=d.batch*d.channels*ohn*own;if(i<total){uint ow=i%own,oh=(i/own)%ohn,c=(i/(own*ohn))%d.channels,n=i/(own*ohn*d.channels);float best=-INFINITY;for(uint h=0;h<d.pool_h;++h)for(uint w=0;w<d.pool_w;++w){uint ix=((n*d.channels+c)*d.height+oh*d.pool_h+h)*d.width+ow*d.pool_w+w;best=max(best,x[ix]);}y[i]=best;}}\n"
           "kernel void pool_b(device const float*x [[buffer(0)]],device const float*gy [[buffer(1)]],device float*gx [[buffer(2)]],constant PoolDesc&d [[buffer(3)]],uint i [[thread_position_in_grid]]){uint ohn=d.height/d.pool_h,own=d.width/d.pool_w,total=d.batch*d.channels*ohn*own;if(i<total){uint ow=i%own,oh=(i/own)%ohn,c=(i/(own*ohn))%d.channels,n=i/(own*ohn*d.channels);uint besti=0;float best=-INFINITY;for(uint h=0;h<d.pool_h;++h)for(uint w=0;w<d.pool_w;++w){uint ix=((n*d.channels+c)*d.height+oh*d.pool_h+h)*d.width+ow*d.pool_w+w;if(x[ix]>best){best=x[ix];besti=ix;}}gx[besti]+=gy[i];}}\n";
}

static id<MTLComputePipelineState> tc_metal_pipeline(NSString *name) {
    id<MTLComputePipelineState> state = tc_metal_pipelines[name];
    if (state) return state;
    NSError *error = nil;
    id<MTLFunction> function = [tc_metal_library newFunctionWithName:name];
    if (!function) return nil;
    state = [tc_metal_device newComputePipelineStateWithFunction:function error:&error];
    if (!state) return nil;
    tc_metal_pipelines[name] = state;
    return state;
}

static int tc_metal_run(NSString *name, const float *const *pointers,
                        NSUInteger pointer_count, const void *const *bytes,
                        const NSUInteger *byte_sizes, NSUInteger byte_count,
                        size_t count) {
    if (!tc_metal_init()) return 0;
    id<MTLComputePipelineState> pipeline = tc_metal_pipeline(name);
    id<MTLCommandBuffer> command = tc_metal_command();
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!pipeline || !command || !encoder) return 0;
    NSMutableArray<NSValue *> *keys = [NSMutableArray array];
    for (NSUInteger i = 0; i < pointer_count; ++i) {
        if (!pointers[i]) {
            [encoder setBuffer:nil offset:0 atIndex:i];
            continue;
        }
        TCMetalBuffer *info = tc_metal_info(pointers[i]);
        if (!info) { [encoder endEncoding]; return 0; }
        [encoder setBuffer:info.buffer offset:0 atIndex:i];
        [keys addObject:[NSValue valueWithPointer:pointers[i]]];
    }
    for (NSUInteger i = 0; i < byte_count; ++i)
        [encoder setBytes:bytes[i] length:byte_sizes[i]
                  atIndex:pointer_count + i];
    [encoder setComputePipelineState:pipeline];
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup < 256
                           ? pipeline.maxTotalThreadsPerThreadgroup : 256;
    if (!width) width = 1;
    [encoder dispatchThreadgroups:MTLSizeMake((count + width - 1) / width, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    for (NSValue *key in keys) tc_metal_buffers[key].lastCommand = command;
    [command commit];
    return 1;
}

static int tc_metal_init(void) {
    if (tc_metal_state) return tc_metal_state > 0;
    @autoreleasepool {
        tc_metal_device = MTLCreateSystemDefaultDevice();
        tc_metal_queue = [tc_metal_device newCommandQueue];
        if (!tc_metal_device || !tc_metal_queue) {
            tc_metal_state = -1;
            return 0;
        }
        tc_metal_buffers = [NSMutableDictionary dictionary];
        tc_metal_pool = [NSMutableArray array];
        tc_metal_pipelines = [NSMutableDictionary dictionary];
        NSError *error = nil;
        tc_metal_library = [tc_metal_device newLibraryWithSource:
            tc_metal_shader_source() options:nil error:&error];
        if (!tc_metal_library) {
            NSLog(@"TensorC Metal shader compile failed: %@", error);
            tc_metal_state = -1;
            return 0;
        }
        const char *env = getenv("TC_METAL_MIN_ELEMS");
        if (env && *env) tc_metal_threshold = (size_t)strtoull(env, NULL, 10);
        tc_metal_state = 1;
    }
    return 1;
}

static TCMetalBuffer *tc_metal_info(const float *pointer) {
    return tc_metal_buffers[[NSValue valueWithPointer:pointer]];
}

static int tc_metal_wait(TCMetalBuffer *info) {
    if (!info.lastCommand) return 1;
    [info.lastCommand waitUntilCompleted];
    return info.lastCommand.status == MTLCommandBufferStatusCompleted;
}

static id<MTLCommandBuffer> tc_metal_command(void) {
    return tc_metal_init() ? [tc_metal_queue commandBuffer] : nil;
}

int tc_metal_available(void) { return tc_metal_init(); }
int tc__device_available(void) { return tc_metal_available(); }
size_t tc__device_min_elems(void) { return tc_metal_threshold; }
size_t tc__device_min_gemm_work(void) {
    return tc_metal_threshold < 4096 ? tc_metal_threshold : 4096;
}

float *tc__device_alloc(size_t count) {
    if (!tc_metal_init() || count == 0 || count > SIZE_MAX / sizeof(float))
        return NULL;
    @autoreleasepool {
        for (NSUInteger i = 0; i < tc_metal_pool.count; ++i) {
            TCMetalBuffer *candidate = tc_metal_pool[i];
            if (candidate.count == count &&
                candidate.lastCommand.status == MTLCommandBufferStatusCompleted) {
                [tc_metal_pool removeObjectAtIndex:i];
                candidate.lastCommand = nil;
                tc_metal_buffers[[NSValue valueWithPointer:candidate.pointer]] = candidate;
                return candidate.pointer;
            }
        }
        TCMetalBuffer *info = [TCMetalBuffer new];
        info.buffer = [tc_metal_device newBufferWithLength:count * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        if (!info.buffer) return NULL;
        info.pointer = (float *)info.buffer.contents;
        info.count = count;
        tc_metal_buffers[[NSValue valueWithPointer:info.pointer]] = info;
        return info.pointer;
    }
}

void tc__device_free(float *pointer, size_t count) {
    if (!pointer || !tc_metal_init()) return;
    @autoreleasepool {
        NSValue *key = [NSValue valueWithPointer:pointer];
        TCMetalBuffer *info = tc_metal_buffers[key];
        if (!info || info.count != count) return;
        [tc_metal_buffers removeObjectForKey:key];
        [tc_metal_pool addObject:info];
    }
}

int tc__device_copy_h2d(float *dst, const float *src, size_t count) {
    TCMetalBuffer *info = tc_metal_info(dst);
    if (!info || !src || count > info.count || !tc_metal_wait(info)) return 0;
    memcpy(dst, src, count * sizeof(float));
    return 1;
}

int tc__device_copy_d2h(float *dst, const float *src, size_t count) {
    TCMetalBuffer *info = tc_metal_info(src);
    if (!info || !dst || count > info.count || !tc_metal_wait(info)) return 0;
    memcpy(dst, src, count * sizeof(float));
    return 1;
}

static int tc_metal_matmul(const float *a, const float *b, float *c,
                           size_t m, size_t k, size_t n,
                           BOOL transpose_a, BOOL transpose_b,
                           size_t a_rows, size_t a_cols,
                           size_t b_rows, size_t b_cols, float beta) {
    TCMetalBuffer *ai = tc_metal_info(a), *bi = tc_metal_info(b);
    TCMetalBuffer *ci = tc_metal_info(c);
    if (!tc_metal_init() || !ai || !bi || !ci ||
        a_rows > SIZE_MAX / sizeof(float) / a_cols ||
        b_rows > SIZE_MAX / sizeof(float) / b_cols ||
        ci.count < m * n)
        return 0;
    MPSMatrixDescriptor *ad = [MPSMatrixDescriptor
        matrixDescriptorWithRows:a_rows columns:a_cols
        rowBytes:a_cols * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *bd = [MPSMatrixDescriptor
        matrixDescriptorWithRows:b_rows columns:b_cols
        rowBytes:b_cols * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *cd = [MPSMatrixDescriptor
        matrixDescriptorWithRows:m columns:n
        rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
    if (!ad || !bd || !cd) return 0;
    MPSMatrix *am = [[MPSMatrix alloc] initWithBuffer:ai.buffer offset:0
                                          descriptor:ad];
    MPSMatrix *bm = [[MPSMatrix alloc] initWithBuffer:bi.buffer offset:0
                                          descriptor:bd];
    MPSMatrix *cm = [[MPSMatrix alloc] initWithBuffer:ci.buffer offset:0
                                          descriptor:cd];
    MPSMatrixMultiplication *op = [[MPSMatrixMultiplication alloc]
        initWithDevice:tc_metal_device transposeLeft:transpose_a
        transposeRight:transpose_b resultRows:m resultColumns:n
        interiorColumns:k alpha:1.0 beta:beta];
    id<MTLCommandBuffer> command = tc_metal_command();
    if (!am || !bm || !cm || !op || !command) return 0;
    [op encodeToCommandBuffer:command leftMatrix:am rightMatrix:bm
                 resultMatrix:cm];
    ai.lastCommand = bi.lastCommand = ci.lastCommand = command;
    [command commit];
    return 1;
}

int tc__device_matmul_nn(const float *a, const float *b, float *c,
                         size_t m, size_t k, size_t n) {
    return tc_metal_matmul(a, b, c, m, k, n, NO, NO, m, k, k, n, 0.0f);
}
int tc__device_matmul_nt(const float *g, const float *b, float *da,
                         size_t m, size_t k, size_t n) {
    if(m>UINT32_MAX||k>UINT32_MAX||n>UINT32_MAX)return 0;
    const uint32_t rows=(uint32_t)m,inner=(uint32_t)k,cols=(uint32_t)n,mode=1;
    const float*p[]={g,b,da};const void*v[]={&rows,&inner,&cols,&mode};
    const NSUInteger z[]={sizeof(rows),sizeof(inner),sizeof(cols),sizeof(mode)};
    return tc_metal_run(@"matmul_b",p,3,v,z,4,m*k);
}
int tc__device_matmul_tn(const float *a, const float *g, float *db,
                         size_t m, size_t k, size_t n) {
    if(m>UINT32_MAX||k>UINT32_MAX||n>UINT32_MAX)return 0;
    const uint32_t rows=(uint32_t)m,inner=(uint32_t)k,cols=(uint32_t)n,mode=2;
    const float*p[]={a,g,db};const void*v[]={&rows,&inner,&cols,&mode};
    const NSUInteger z[]={sizeof(rows),sizeof(inner),sizeof(cols),sizeof(mode)};
    return tc_metal_run(@"matmul_b",p,3,v,z,4,k*n);
}

static int tc_metal_conv_desc(TCMetalConvDesc *d, size_t batch,
                              size_t channels, size_t height, size_t width,
                              size_t filters, size_t kh, size_t kw,
                              size_t oh, size_t ow, size_t sh, size_t sw,
                              size_t ph, size_t pw) {
    const size_t values[]={batch,channels,height,width,filters,kh,kw,oh,ow,sh,sw,ph,pw};
    for (size_t i=0;i<sizeof(values)/sizeof(values[0]);++i)
        if (values[i]>UINT32_MAX) return 0;
    *d=(TCMetalConvDesc){(uint32_t)batch,(uint32_t)channels,(uint32_t)height,
        (uint32_t)width,(uint32_t)filters,(uint32_t)kh,(uint32_t)kw,
        (uint32_t)oh,(uint32_t)ow,(uint32_t)sh,(uint32_t)sw,
        (uint32_t)ph,(uint32_t)pw};
    return 1;
}

int tc__device_conv2d_forward(const float*x,const float*w,float*y,size_t batch,
    size_t channels,size_t height,size_t width,size_t filters,size_t kh,
    size_t kw,size_t oh,size_t ow,size_t sh,size_t sw,size_t ph,size_t pw) {
    TCMetalConvDesc d;
    if (!tc_metal_conv_desc(&d,batch,channels,height,width,filters,kh,kw,oh,ow,sh,sw,ph,pw)) return 0;
    const size_t count=batch*filters*oh*ow;
    if(count>UINT32_MAX)return 0;
    const float*p[]={x,w,y};const void*v[]={&d};const NSUInteger z[]={sizeof(d)};
    return tc_metal_run(@"conv_f",p,3,v,z,1,count);
}

int tc__device_conv2d_backward(const float*x,const float*w,const float*gy,
    float*gx,float*gw,size_t batch,size_t channels,size_t height,size_t width,
    size_t filters,size_t kh,size_t kw,size_t oh,size_t ow,size_t sh,size_t sw,
    size_t ph,size_t pw,int want_x,int want_w) {
    TCMetalConvDesc d;
    if(!tc_metal_conv_desc(&d,batch,channels,height,width,filters,kh,kw,oh,ow,sh,sw,ph,pw))return 0;
    if(want_x){if(!gx||batch*channels*height*width>UINT32_MAX)return 0;const float*p[]={x,w,gy,gx};const void*v[]={&d};const NSUInteger z[]={sizeof(d)};if(!tc_metal_run(@"conv_dx",p,4,v,z,1,batch*channels*height*width))return 0;}
    if(want_w){if(!gw||filters*channels*kh*kw>UINT32_MAX)return 0;const float*p[]={x,gy,gw};const void*v[]={&d};const NSUInteger z[]={sizeof(d)};if(!tc_metal_run(@"conv_dw",p,3,v,z,1,filters*channels*kh*kw))return 0;}
    return 1;
}

static int tc_metal_pool_desc(TCMetalPoolDesc *d,size_t batch,size_t channels,
                              size_t height,size_t width,size_t ph,size_t pw) {
    const size_t values[]={batch,channels,height,width,ph,pw};
    for(size_t i=0;i<6;++i)if(values[i]>UINT32_MAX)return 0;
    *d=(TCMetalPoolDesc){(uint32_t)batch,(uint32_t)channels,(uint32_t)height,
                         (uint32_t)width,(uint32_t)ph,(uint32_t)pw};
    return 1;
}
int tc__device_maxpool2d_forward(const float*x,float*y,size_t batch,
    size_t channels,size_t height,size_t width,size_t ph,size_t pw) {
    TCMetalPoolDesc d;if(!tc_metal_pool_desc(&d,batch,channels,height,width,ph,pw))return 0;
    const size_t count=batch*channels*(height/ph)*(width/pw);if(count>UINT32_MAX)return 0;
    const float*p[]={x,y};const void*v[]={&d};const NSUInteger z[]={sizeof(d)};
    return tc_metal_run(@"pool_f",p,2,v,z,1,count);
}
int tc__device_maxpool2d_backward(const float*x,const float*gy,float*gx,
    size_t batch,size_t channels,size_t height,size_t width,size_t ph,size_t pw) {
    TCMetalPoolDesc d;if(!tc_metal_pool_desc(&d,batch,channels,height,width,ph,pw))return 0;
    const size_t count=batch*channels*(height/ph)*(width/pw);if(count>UINT32_MAX)return 0;
    const float*p[]={x,gy,gx};const void*v[]={&d};const NSUInteger z[]={sizeof(d)};
    return tc_metal_run(@"pool_b",p,3,v,z,1,count);
}

/* Operations without a Metal kernel yet return 0 so the graph uses its CPU path. */
static int tc_metal_binary(const float*a,const float*b,float*out,size_t n,
                           const tc_device_broadcast_desc*d,int mul) {
    if (!d || n > UINT32_MAX) return 0;
    const uint32_t count=(uint32_t)n, operation=(uint32_t)mul;
    const float *p[]={a,b,out}; const void *v[]={d,&count,&operation};
    const NSUInteger sizes[]={sizeof(*d),sizeof(count),sizeof(operation)};
    return tc_metal_run(@"binary_f",p,3,v,sizes,3,n);
}
int tc__device_add_forward(const float*a,const float*b,float*o,size_t n,const tc_device_broadcast_desc*d){return tc_metal_binary(a,b,o,n,d,0);}
int tc__device_mul_forward(const float*a,const float*b,float*o,size_t n,const tc_device_broadcast_desc*d){return tc_metal_binary(a,b,o,n,d,1);}
static int tc_metal_binary_backward(const float*g,const float*a,const float*b,float*ga,float*gb,size_t n,const tc_device_broadcast_desc*d,int wa,int wb,int mul){
    if (!d||!wa||!wb||!ga||!gb||n>UINT32_MAX) return 0;
    const uint32_t count=(uint32_t)n,operation=(uint32_t)mul,yes=1;
    const float*p[]={g,a,b,ga,gb}; const void*v[]={d,&count,&operation,&yes,&yes};
    const NSUInteger sizes[]={sizeof(*d),sizeof(count),sizeof(operation),sizeof(yes),sizeof(yes)};
    return tc_metal_run(@"binary_b",p,5,v,sizes,5,n);
}
int tc__device_add_backward(const float*g,float*a,float*b,size_t n,const tc_device_broadcast_desc*d,int wa,int wb){return tc_metal_binary_backward(g,g,g,a,b,n,d,wa,wb,0);}
int tc__device_mul_backward(const float*g,const float*a,const float*b,float*ga,float*gb,size_t n,const tc_device_broadcast_desc*d,int wa,int wb){return tc_metal_binary_backward(g,a,b,ga,gb,n,d,wa,wb,1);}
int tc__device_relu_forward(const float*x,float*y,size_t n){if(n>UINT32_MAX)return 0;const uint32_t c=(uint32_t)n;const float*p[]={x,y};const void*v[]={&c};const NSUInteger z[]={sizeof(c)};return tc_metal_run(@"relu_f",p,2,v,z,1,n);}
int tc__device_relu_backward(const float*x,const float*g,float*gx,size_t n){if(n>UINT32_MAX)return 0;const uint32_t c=(uint32_t)n;const float*p[]={x,g,gx};const void*v[]={&c};const NSUInteger z[]={sizeof(c)};return tc_metal_run(@"relu_b",p,3,v,z,1,n);}
int tc__device_sigmoid_forward(const float*x,float*y,size_t n){(void)x;(void)y;(void)n;return 0;}
int tc__device_sigmoid_backward(const float*y,const float*g,float*gx,size_t n){(void)y;(void)g;(void)gx;(void)n;return 0;}
int tc__device_tanh_forward(const float*x,float*y,size_t n){(void)x;(void)y;(void)n;return 0;}
int tc__device_tanh_backward(const float*y,const float*g,float*gx,size_t n){(void)y;(void)g;(void)gx;(void)n;return 0;}
int tc__device_softmax_forward(const float*x,float*y,size_t r,size_t c){if(r>UINT32_MAX||c>UINT32_MAX)return 0;const uint32_t rows=(uint32_t)r,cols=(uint32_t)c;const float*p[]={x,y};const void*v[]={&rows,&cols};const NSUInteger z[]={sizeof(rows),sizeof(cols)};return tc_metal_run(@"softmax_f",p,2,v,z,2,r);}
int tc__device_softmax_backward(const float*y,const float*g,float*gx,size_t r,size_t c){if(r>UINT32_MAX||c>UINT32_MAX)return 0;const uint32_t rows=(uint32_t)r,cols=(uint32_t)c;const float*p[]={y,g,gx};const void*v[]={&rows,&cols};const NSUInteger z[]={sizeof(rows),sizeof(cols)};return tc_metal_run(@"softmax_b",p,3,v,z,2,r);}
int tc__device_layer_norm_forward(const float*x,float*y,size_t r,size_t w){(void)x;(void)y;(void)r;(void)w;return 0;}
int tc__device_layer_norm_backward(const float*x,const float*g,float*gx,size_t r,size_t w){(void)x;(void)g;(void)gx;(void)r;(void)w;return 0;}
int tc__device_rms_norm_forward(const float*x,float*y,size_t r,size_t w){(void)x;(void)y;(void)r;(void)w;return 0;}
int tc__device_rms_norm_backward(const float*x,const float*g,float*gx,size_t r,size_t w){(void)x;(void)g;(void)gx;(void)r;(void)w;return 0;}
int tc__device_attention_forward(const float*q,const float*k,const float*v,float*o,size_t b,size_t qs,size_t ks,size_t d,size_t vd,int c){(void)q;(void)k;(void)v;(void)o;(void)b;(void)qs;(void)ks;(void)d;(void)vd;(void)c;return 0;}
int tc__device_attention_backward(const float*q,const float*k,const float*v,const float*o,float*dq,float*dk,float*dv,size_t b,size_t qs,size_t ks,size_t d,size_t vd,int c){(void)q;(void)k;(void)v;(void)o;(void)dq;(void)dk;(void)dv;(void)b;(void)qs;(void)ks;(void)d;(void)vd;(void)c;return 0;}
int tc__device_permute_forward(const float*i,float*o,size_t n,const tc_device_permute_desc*d){(void)i;(void)o;(void)n;(void)d;return 0;}
int tc__device_permute_backward(const float*g,float*gx,size_t n,const tc_device_permute_desc*d){(void)g;(void)gx;(void)n;(void)d;return 0;}
int tc__device_copy_d2d(float*d,const float*s,size_t n){if(n>UINT32_MAX)return 0;const uint32_t c=(uint32_t)n;const float*p[]={s,d};const void*v[]={&c};const NSUInteger z[]={sizeof(c)};return tc_metal_run(@"copy_f",p,2,v,z,1,n);}
int tc__device_accumulate(float*d,const float*s,size_t n){if(n>UINT32_MAX)return 0;const uint32_t c=(uint32_t)n;const float*p[]={s,d};const void*v[]={&c};const NSUInteger z[]={sizeof(c)};return tc_metal_run(@"accum_f",p,2,v,z,1,n);}
int tc__device_zero_fill(float*d,size_t n){if(n>UINT32_MAX)return 0;const uint32_t c=(uint32_t)n;const float*p[]={d};const void*v[]={&c};const NSUInteger z[]={sizeof(c)};return tc_metal_run(@"zero_f",p,1,v,z,1,n);}
int tc__device_cross_entropy_forward(const float*p,const float*t,float*out,size_t n,size_t r){if(n>UINT32_MAX||r>UINT32_MAX||r==0)return 0;const uint32_t count=(uint32_t)n,rows=(uint32_t)r;const float*b[]={p,t,out};const void*v[]={&count,&rows};const NSUInteger z[]={sizeof(count),sizeof(rows)};return tc_metal_run(@"ce_f",b,3,v,z,2,1);}
int tc__device_cross_entropy_backward(const float*p,const float*t,const float*g,float*gp,float*gt,size_t n,size_t r,int wp,int wt){if(n>UINT32_MAX||r>UINT32_MAX)return 0;const uint32_t count=(uint32_t)n,rows=(uint32_t)r,wantp=(uint32_t)wp,wantt=(uint32_t)wt;const float*b[]={p,t,g,gp,gt};const void*v[]={&count,&rows,&wantp,&wantt};const NSUInteger z[]={sizeof(count),sizeof(rows),sizeof(wantp),sizeof(wantt)};return tc_metal_run(@"ce_b",b,5,v,z,4,n);}
int tc__device_embedding_forward(const float*t,const float*i,float*o,size_t n,size_t d){(void)t;(void)i;(void)o;(void)n;(void)d;return 0;}
int tc__device_embedding_backward(const float*g,const float*i,float*gt,size_t n,size_t d){(void)g;(void)i;(void)gt;(void)n;(void)d;return 0;}
int tc__device_sparse_ce_forward(const float*p,const float*i,float*o,size_t r,size_t c){(void)p;(void)i;(void)o;(void)r;(void)c;return 0;}
int tc__device_sparse_ce_backward(const float*p,const float*i,const float*g,float*gp,size_t r,size_t c){(void)p;(void)i;(void)g;(void)gp;(void)r;(void)c;return 0;}
int tc__device_sgd_step(float*p,const float*g,size_t n,float lr){if(n>UINT32_MAX)return 0;const uint32_t c=(uint32_t)n;const float rate=lr;const float*b[]={p,g};const void*v[]={&c,&rate};const NSUInteger z[]={sizeof(c),sizeof(rate)};return tc_metal_run(@"sgd_f",b,2,v,z,2,n);}
int tc__device_adam_step(float*p,const float*g,float*m,float*v,size_t n,float lr,float b1,float b2,float c1,float c2,float e){if(n>UINT32_MAX)return 0;const uint32_t c=(uint32_t)n;const float rate=lr,x1=b1,x2=b2,y1=c1,y2=c2,eps=e;const float*b[]={p,g,m,v};const void*x[]={&c,&rate,&x1,&x2,&y1,&y2,&eps};const NSUInteger z[]={sizeof(c),sizeof(rate),sizeof(x1),sizeof(x2),sizeof(y1),sizeof(y2),sizeof(eps)};return tc_metal_run(@"adam_f",b,4,x,z,7,n);}
