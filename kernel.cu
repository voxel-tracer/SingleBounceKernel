
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <bitset>
#include <fstream>
#include <time.h>
#include <sstream>

#define STATS
//#define RUSSIAN_ROULETTE
//#define SHADOW
#define TEXTURES

//#define PRIMARY_PERFECT
//#define PRIMARY0
//#define PRIMARY1
#define PRIMARY2

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "../cuda-raytracing-optimized/camera.h"
#include "../cuda-raytracing-optimized/scene_materials.h"
#include "../cuda-raytracing-optimized/intersections.h"
#include "../cuda-raytracing-optimized/staircase_scene.h"

#ifdef STATS
#include <cooperative_groups.h>
using namespace cooperative_groups;
#endif


#define EPSILON 0.01f

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define FLAG_SPECULAR   1
#define FLAG_INSIDE     2
#define FLAG_DONE       4
#define FLAGS(p)        ((p.specular ? FLAG_SPECULAR:0) | (p.inside ? FLAG_INSIDE:0) | (p.done ? FLAG_DONE:0))

#ifdef STATS
#define NUM_RAYS_PRIMARY                0
#define NUM_RAYS_PRIMARY_NOHITS         1
#define NUM_RAYS_PRIMARY_BBOX_NOHITS    2
#define NUM_RAYS_SECONDARY              3
#define NUM_RAYS_SECONDARY_NOHIT        4
#define NUM_RAYS_SECONDARY_BBOX_NOHIT   5
#define NUM_RAYS_SHADOWS                6
#define NUM_RAYS_SHADOWS_BBOX_NOHITS    7
#define NUM_RAYS_SHADOWS_NOHITS         8
#define NUM_RAYS_LOW_POWER              9
#define NUM_RAYS_EXCEED_MAX_BOUNCE      10
#define NUM_RAYS_RUSSIAN_KILL           11
#define NUM_RAYS_NAN                    12
#define NUM_RAYS_MAX_TRAVERSED_NODES    13
#define METRIC_ACTIVE                   14
#define METRIC_ACTIVE_ITER              15
#define METRIC_LEAF                     16
#define METRIC_LEAF_ITER                17
#define NUM_RAYS_SIZE                   18

char* statNames[NUM_RAYS_SIZE] = {
    " primary             : ",
    " primary no hit      : ",
    " primary bb nohit    : ",
    " secondary           : ",
    " secondary no hit    : ",
    " secondary bb nohit  : ",
    " shadows             : ",
    " shadows nohit       : ",
    " shadows bb nohit    : ",
    " power < 0.01        : ",
    " exceeded max bounce : ",
    " russiand roulette   : ",
    " max travers. nodes  : ",
    " *** NANs ***        : ",
    " active metric       : ",
    " active.iterations   : ",
    " leaf metric         : ",
    " leaf.iterations     : ",
};
#endif

struct saved_path {
    vec3 origin;
    vec3 rayDir;
    vec3 attenuation; // only needed to visually confirm the rendering is correct

    uint8_t flags;
    rand_state rng;

    __host__ saved_path() {}
    __device__ saved_path(const path& p) : origin(p.origin), rayDir(p.rayDir), flags(FLAGS(p)), attenuation(p.attenuation), rng(p.rng) {}

    __device__ bool isDone() const { return flags & FLAG_DONE; }
    __device__ bool isSpecular() const { return flags & FLAG_SPECULAR; }
    __device__ bool isInside() const { return flags & FLAG_INSIDE; }
};

struct RenderContext {
    int nx, ny, ns;
    camera cam;
    saved_path* paths;
    vec3* colors;

    sphere light = sphere(vec3(52.514355, 715.686951, -272.620972), 50);
    vec3 lightColor = vec3(1, 1, 1) * 80;
    triangle* tris;
    material* materials;
    bbox bounds;
    bvh_node* bvh;
    uint32_t firstLeafIdx;
    uint32_t numPrimitivesPerLeaf = 5; //TODO load this from bin file

#ifdef TEXTURES
    float** tex_data;
    int* tex_width;
    int* tex_height;
#endif


#ifdef STATS
    uint64_t* stats;
    __device__ void incStat(int type) const {
        atomicAdd(stats + type, 1);
    }
    __device__ void incStat(int type, int value) const {
        atomicAdd(stats + type, value);
    }
    __device__ void maxStat(int type, uint64_t value) const {
        atomicMax(stats + type, value);
    }

    void initStats() {
        CUDA(cudaMallocManaged((void**)&stats, NUM_RAYS_SIZE * sizeof(uint64_t)));
        memset(stats, 0, NUM_RAYS_SIZE * sizeof(uint64_t));
    }
    void resetStats() {
        memset(stats, 0, NUM_RAYS_SIZE * sizeof(uint64_t));
    }
    void printStats() const {
        std::cerr << "num rays:\n";
        for (auto i = 0; i < METRIC_ACTIVE; i++) {
            if (stats[i] > 0)
                std::cerr << statNames[i] << std::fixed << stats[i] << std::endl;
        }
        // print traversal metrics
        if (stats[METRIC_ACTIVE_ITER] > 0)
            std::cerr << statNames[METRIC_ACTIVE] << (stats[METRIC_ACTIVE] * 100.0 / (stats[METRIC_ACTIVE_ITER] * 32)) << std::endl;
        if (stats[METRIC_LEAF_ITER] > 0)
            std::cerr << statNames[METRIC_LEAF] << (stats[METRIC_LEAF] * 100.0 / (stats[METRIC_LEAF_ITER] * 32)) << std::endl;
    }
#else
    __device__ void incStat(int type) const {}
    __device__ void incStat(int type, int value) const {}
    __device__ void maxStat(int type, uint64_t value) const {}
    void initStats() {}
    void resetStats() {}
    void printStats() const {}
#endif
};

enum OBJ_ID {
    NONE,
    TRIMESH,
    PLANE,
    LIGHT
};

__device__ float hitBvh(const ray& r, const RenderContext& context, float t_min, tri_hit& rec, bool isShadow) {
    bool down = true;
    int idx = 1;
    float closest = FLT_MAX;
    unsigned int bitStack = 0;

    while (true) {
#ifdef STATS
        /*if (isShadow)*/ {
            auto g = coalesced_threads();
            if (g.thread_rank() == 0) {
                context.incStat(METRIC_ACTIVE_ITER);
                context.incStat(METRIC_ACTIVE, g.size());
            }
        }
#endif

        if (down) {
            bvh_node node = context.bvh[idx];
            if (hit_bbox(node.min(), node.max(), r, closest)) {
                if (idx >= context.firstLeafIdx) { // leaf node
#ifdef STATS
                    /*if (isShadow)*/ {
                        auto g = coalesced_threads();
                        if (g.thread_rank() == 0) {
                            context.incStat(METRIC_LEAF_ITER);
                            context.incStat(METRIC_LEAF, g.size());
                        }
                    }
#endif

                    int first = (idx - context.firstLeafIdx) * context.numPrimitivesPerLeaf;
                    for (auto i = 0; i < context.numPrimitivesPerLeaf; i++) {
                        const triangle tri = context.tris[first + i];
                        if (isinf(tri.v[0].x()))
                            break; // we reached the end of the primitives buffer
                        float u, v;
                        float hitT = triangleHit(tri, r, t_min, closest, u, v);
                        if (hitT < FLT_MAX) {
                            if (isShadow) return 0.0f;

                            closest = hitT;
                            rec.triId = first + i;
                            rec.u = u;
                            rec.v = v;
                        }
                    }
                    down = false;
                }
                else { // internal node
                 // current -> left or right
                    const int childIdx = signbit(r.direction()[node.split_axis()]); // 0=left, 1=right
                    bitStack = (bitStack << 1) + childIdx; // push current child idx in the stack
                    idx = (idx << 1) + childIdx;
                }
            }
            else { // ray didn't intersect the node, backtrack
                down = false;
            }
        }
        else if (idx == 1) { // we backtracked up to the root node
            break;
        }
        else { // back tracking
            const int currentChildIdx = bitStack & 1;
            if ((idx & 1) == currentChildIdx) { // node == current child, visit sibling
                idx += -2 * currentChildIdx + 1; // node = node.sibling
                down = true;
            }
            else { // we visited both siblings, backtrack
                bitStack = bitStack >> 1;
                idx = idx >> 1; // node = node.parent
            }
        }
    }

    return closest;
}

__device__ float hitMesh(const ray& r, const RenderContext& context, float t_min, float t_max, tri_hit& rec, bool primary, bool isShadow) {
    if (!hit_bbox(context.bounds.min, context.bounds.max, r, t_max)) {
#ifdef STATS
        if (isShadow) context.incStat(NUM_RAYS_SHADOWS_BBOX_NOHITS);
        else context.incStat(primary ? NUM_RAYS_PRIMARY_BBOX_NOHITS : NUM_RAYS_SECONDARY_BBOX_NOHIT);
#endif
        return FLT_MAX;
    }

    return hitBvh(r, context, t_min, rec, isShadow);
}

__device__ bool hit(const RenderContext& context, const path& p, bool isShadow, intersection& inters) {
    const ray r = isShadow ? ray(p.origin, p.shadowDir) : ray(p.origin, p.rayDir);
    tri_hit triHit;
    bool primary = p.bounce == 0;
    inters.objId = NONE;
    if ((inters.t = hitMesh(r, context, EPSILON, FLT_MAX, triHit, primary, isShadow)) < FLT_MAX) {
        if (isShadow) return true; // we don't need to compute the intersection details for shadow rays

        inters.objId = TRIMESH;
        triangle tri = context.tris[triHit.triId];
        inters.meshID = tri.meshID;
        inters.normal = unit_vector(cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[0]));
        inters.texCoords[0] = (triHit.u * tri.texCoords[1 * 2 + 0] + triHit.v * tri.texCoords[2 * 2 + 0] + (1 - triHit.u - triHit.v) * tri.texCoords[0 * 2 + 0]);
        inters.texCoords[1] = (triHit.u * tri.texCoords[1 * 2 + 1] + triHit.v * tri.texCoords[2 * 2 + 1] + (1 - triHit.u - triHit.v) * tri.texCoords[0 * 2 + 1]);
    }
    else {
        if (isShadow) return false; // shadow rays only care about the main triangle mesh

        if (p.specular && sphereHit(context.light, r, EPSILON, FLT_MAX) < FLT_MAX) { // specular rays should intersect with the light
            inters.objId = LIGHT;
            return true; // we don't need to compute p and update normal to face the ray
        }
    }

    if (inters.objId != NONE) {
        inters.p = r.point_at_parameter(inters.t);
        if (dot(r.direction(), inters.normal) > 0.0f)
            inters.normal = -inters.normal; // ensure normal is always facing the ray
        return true;
    }

    return false;
}

#ifdef SHADOW
__device__ bool generateShadowRay(const RenderContext& context, path& p, const intersection& inters) {
    // create a random direction towards the light
    // coord system for sampling
    const vec3 sw = unit_vector(context.light.center - p.origin);
    const vec3 su = unit_vector(cross(fabs(sw.x()) > 0.01f ? vec3(0, 1, 0) : vec3(1, 0, 0), sw));
    const vec3 sv = cross(sw, su);

    // sample sphere by solid angle
    const float cosAMax = sqrt(1.0f - context.light.radius * context.light.radius / (p.origin - context.light.center).squared_length());
    if (isnan(cosAMax)) return false; // if the light radius is too big and it reaches the model, this will be null

    const float eps1 = rnd(p.rng);
    const float eps2 = rnd(p.rng);
    const float cosA = 1.0f - eps1 + eps1 * cosAMax;
    const float sinA = sqrt(1.0f - cosA * cosA);
    const float phi = 2 * M_PI * eps2;
    const vec3 l = su * cosf(phi) * sinA + sv * sinf(phi) * sinA + sw * cosA;

    const float dotl = dot(l, inters.normal);
    if (dotl <= 0)
        return false;

    p.shadowDir = unit_vector(l);
    const float omega = 2 * M_PI * (1.0f - cosAMax);
    p.lightContribution = p.attenuation * context.lightColor * dotl * omega / M_PI;

    return true;
}
#endif

// only handles a single bounce, as such doesn't track when a path exceeds max depth
__device__ void colorBounce(const RenderContext& context, path& p) {
        intersection inters;
#ifdef STATS
        bool primary = p.bounce == 0;
        context.incStat(primary ? NUM_RAYS_PRIMARY : NUM_RAYS_SECONDARY);
        if (p.attenuation.length() < 0.01f) context.incStat(NUM_RAYS_LOW_POWER);
#endif
        if (!hit(context, p, false, inters)) {
#ifdef STATS
            if (primary) context.incStat(NUM_RAYS_PRIMARY_NOHITS);
            else context.incStat(NUM_RAYS_SECONDARY_NOHIT);
#endif
            p.color += p.attenuation * vec3(0.5f, 0.5f, 0.5f);
            p.done = true;
            return;
        }

        if (inters.objId == LIGHT) {
            // only specular rays can intersect the light
            // ray hit the light, compute its contribution and add it to the path's color
#ifdef SHADOW
            // we should uncomment this line, but we need to compute light contribution properly
            // p.color += p.attenuation * context.lightColor;
#else
            p.color += p.attenuation * context.lightColor;
#endif
            p.done = true;
            return;
        }

        inters.inside = p.inside;
        scatter_info scatter(inters);
        if (inters.objId == TRIMESH) {
            const material& mat = context.materials[inters.meshID];
#ifdef TEXTURES
            vec3 albedo;
            if (mat.texId != -1) {
                int texId = mat.texId;
                int width = context.tex_width[texId];
                int height = context.tex_height[texId];
                float tu = inters.texCoords[0];
                tu = tu - floorf(tu);
                float tv = inters.texCoords[1];
                tv = tv - floorf(tv);
                const int tx = (width - 1) * tu;
                const int ty = (height - 1) * tv;
                const int tIdx = ty * width + tx;
                albedo = vec3(
                    context.tex_data[texId][tIdx * 3 + 0],
                    context.tex_data[texId][tIdx * 3 + 1],
                    context.tex_data[texId][tIdx * 3 + 2]);
            }
            else {
                albedo = mat.color;
            }
#else
            vec3 albedo(0.5f, 0.5f, 0.5f);
#endif
            material_scatter(scatter, inters, p.rayDir, context.materials[inters.meshID], albedo, p.rng);
        }
        else
            floor_diffuse_scatter(scatter, inters, p.rayDir, p.rng);

        p.origin += scatter.t * p.rayDir;
        p.rayDir = scatter.wi;
        p.attenuation *= scatter.throughput;
        p.specular = scatter.specular;
        p.inside = scatter.refracted ? !p.inside : p.inside;

        //p.color = p.attenuation; // debug
#ifdef SHADOW
        // trace shadow ray for diffuse rays
        if (!p.specular && generateShadowRay(context, p, inters)) {
            context.incStat(NUM_RAYS_SHADOWS);
            if (!hit(context, p, true, inters)) {
                context.incStat(NUM_RAYS_SHADOWS_NOHITS);
                // intersection point is illuminated by the light
                p.color += p.lightContribution;
            }
        }
#endif
#ifdef RUSSIAN_ROULETTE
        // russian roulette
        if (p.bounce > 3) {
            float m = max(p.attenuation);
            if (rnd(p.rng) > m) {
                context.incStat(NUM_RAYS_RUSSIAN_KILL);
                p.done = true;
                return;
            }
            p.attenuation *= 1 / m;
        }
#endif
}

__global__ void bounce(const RenderContext context, int bounce, bool save) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= context.nx) || (y >= context.ny)) return;

    path p;
    p.pixelId = y * context.nx + x;
    vec3 color(0, 0, 0);
    for (int s = 0; s < context.ns; s++) {
        saved_path sp = context.paths[p.pixelId * context.ns + s];
        if (sp.isDone()) continue;

        p.origin = sp.origin;
        p.rayDir = sp.rayDir;
        p.rng = sp.rng;
        p.attenuation = sp.attenuation;
        p.color = vec3();
        p.specular = sp.isSpecular();
        p.inside = sp.isInside();
        p.done = false;
        p.bounce = bounce; // only needed by stats

        colorBounce(context, p);
        color += p.color;
#ifdef STATS
        if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif
        if (save) {
            // all samples for same pixel are saved in consecutive order
            context.paths[p.pixelId * context.ns + s] = saved_path(p);
        }
    }

    if (save) {
        context.colors[p.pixelId] += color / float(context.ns);
    }
}

__global__ void primary(const RenderContext context) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= context.nx) || (y >= context.ny)) return;

    path p;
    p.pixelId = y * context.nx + x;
    p.rng = (wang_hash(p.pixelId) * 336343633) | 1;

    for (int s = 0; s < context.ns; s++) {
        float u = float(x + rnd(p.rng)) / float(context.nx);
        float v = float(y + rnd(p.rng)) / float(context.ny);
        ray r = get_ray(context.cam, u, v, p.rng);
        p.origin = r.origin();
        p.rayDir = r.direction();
        p.attenuation = vec3(1, 1, 1);
        p.specular = false;
        p.inside = false;
        p.done = false;
        // all samples for same pixel are saved in consecutive order
        context.paths[p.pixelId * context.ns + s] = saved_path(p);
    }
}

#ifdef PRIMARY0
// original primaryBounce() kernel: each thread handles all samples of the same pixel
__global__ void primaryBounce0(const RenderContext context, bool save) {
#ifdef PERFECT_PRIMARY
    int x = ((threadIdx.x + blockIdx.x * blockDim.x) / 32) * 32; // all rays in the same warp have the same pixelId
#else
    int x = threadIdx.x + blockIdx.x * blockDim.x;
#endif // PERFECT_PRIMARY


    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= context.nx) || (y >= context.ny)) return;

    path p;
    vec3 color(0, 0, 0);
    p.pixelId = y * context.nx + x;
    p.rng = (wang_hash(p.pixelId) * 336343633) | 1;

    for (int s = 0; s < context.ns; s++) {
        float u = float(x + rnd(p.rng)) / float(context.nx);
        float v = float(y + rnd(p.rng)) / float(context.ny);
        ray r = get_ray(context.cam, u, v, p.rng);
        p.origin = r.origin();
        p.rayDir = r.direction();
        p.attenuation = vec3(1, 1, 1);
        p.color = vec3();
        p.specular = false;
        p.inside = false;
        p.done = false;
        p.bounce = 0;

        colorBounce(context, p);
        color += p.color;
#ifdef STATS
        if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif

        if (save) {
            // all samples for same pixel are saved in consecutive order
            context.paths[p.pixelId * context.ns + s] = saved_path(p);
        }
    }

    if (save) {
        context.colors[p.pixelId] += color / float(context.ns);
    }
}
#endif // PRIMARY0

#ifdef PRIMARY1
// start one thread per sample, each consecutive 32 samples (warp) are from the same pixel
__global__ void primaryBounce1(const RenderContext context, bool save) {
    int xs = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((xs >= (context.nx * context.ns)) || (y >= context.ny)) return;
    int x = xs / context.ns;
    int s = xs % context.ns;

    path p;
    vec3 color(0, 0, 0);
    p.pixelId = y * context.nx + x;
    int sampleId = p.pixelId * context.ns + s;
    p.rng = (wang_hash(sampleId) * 336343633) | 1;

    float u = float(x + rnd(p.rng)) / float(context.nx);
    float v = float(y + rnd(p.rng)) / float(context.ny);
    ray r = get_ray(context.cam, u, v, p.rng);
    p.origin = r.origin();
    p.rayDir = r.direction();
    p.attenuation = vec3(1, 1, 1);
    p.color = vec3();
    p.specular = false;
    p.inside = false;
    p.done = false;
    p.bounce = 0;

    colorBounce(context, p);
    color += p.color;
#ifdef STATS
    if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif

    // following code is not used, but leave it here for now in case compiler is smart enough to undo all our work if we don't save anything
    if (save) {
        // all samples for same pixel are saved in consecutive order
        context.paths[sampleId] = saved_path(p);
        context.colors[sampleId] = color;
    }
}
#endif // PRIMARY1


#ifdef PRIMARY2
// start 32 threads per pixel, then each warp keeps looping until all samples of the pixel are processed
__global__ void primaryBounce2(const RenderContext context, bool save) {
    int xs = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((xs >= (context.nx * 32)) || (y >= context.ny)) return;
    int x = xs / 32;
    int s = xs % 32;

    path p;
    vec3 color(0, 0, 0);
    p.pixelId = y * context.nx + x;
    int sampleId = p.pixelId * context.ns + s;
    p.rng = (wang_hash(sampleId) * 336343633) | 1;

    for (; s < context.ns; s+= 32) {
        float u = float(x + rnd(p.rng)) / float(context.nx);
        float v = float(y + rnd(p.rng)) / float(context.ny);
        ray r = get_ray(context.cam, u, v, p.rng);
        p.origin = r.origin();
        p.rayDir = r.direction();
        p.attenuation = vec3(1, 1, 1);
        p.color = vec3();
        p.specular = false;
        p.inside = false;
        p.done = false;
        p.bounce = 0;

        colorBounce(context, p);
        color += p.color;
#ifdef STATS
        if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif

        if (save) {
            // all samples for same pixel are saved in consecutive order
            context.paths[sampleId] = saved_path(p);
            context.colors[sampleId] = color;
        }
    }
}
#endif // PRIMARY2

bool initRenderContext(RenderContext& context, int nx, int ny, int ns, bool save) {
    camera cam = setup_camera(nx, ny);

    scene sc;
    if (!load_scene(sc)) {
        std::cerr << "Failed to load scene" << std::endl;
        return false;
    }

    kernel_scene ksc;
    if (!setup_kernel_scene(sc, ksc)) {
        std::cerr << "Failed to setup kernel scene" << std::endl;
        return false;
    }

    context.nx = nx;
    context.ny = ny;
    context.ns = ns;
    context.cam = cam;
    context.numPrimitivesPerLeaf = ksc.numPrimitivesPerLeaf;

    if (save) {
        uint32_t numpaths = context.nx * context.ny * context.ns;
        CUDA(cudaMallocManaged((void**)&context.paths, numpaths * sizeof(saved_path)));

#ifdef PRIMARY0
        CUDA(cudaMallocManaged((void**)&context.colors, context.nx * context.ny * sizeof(vec3)));
        memset(context.colors, 0, context.nx * context.ny * sizeof(vec3));
#endif
#if defined(PRIMARY1) || defined(PRIMARY2)
        // to simplify primary1 kernel, store each color sample separately
        uint32_t size = context.nx * context.ny * context.ns * sizeof(vec3);
        CUDA(cudaMallocManaged((void**)&context.colors, size));
        memset(context.colors, 0, size);
#endif
    }

    CUDA(cudaMalloc((void**)&context.tris, ksc.m->numTris * sizeof(triangle)));
    CUDA(cudaMemcpy(context.tris, ksc.m->tris, ksc.m->numTris * sizeof(triangle), cudaMemcpyHostToDevice));
    CUDA(cudaMalloc((void**)&context.bvh, ksc.m->numBvhNodes * sizeof(bvh_node)));
    CUDA(cudaMemcpy(context.bvh, ksc.m->bvh, ksc.m->numBvhNodes * sizeof(bvh_node), cudaMemcpyHostToDevice));
    context.firstLeafIdx = ksc.m->numBvhNodes / 2;
    context.bounds = ksc.m->bounds;

    CUDA(cudaMalloc((void**)&context.materials, ksc.numMaterials * sizeof(material)));
    CUDA(cudaMemcpy(context.materials, ksc.materials, ksc.numMaterials * sizeof(material), cudaMemcpyHostToDevice));
#ifdef TEXTURES
    if (sc.numTextures > 0) {
        int* tex_width = new int[sc.numTextures];
        int* tex_height = new int[sc.numTextures];
        float** tex_data = new float* [sc.numTextures];

        for (auto i = 0; i < sc.numTextures; i++) {
            const stexture& tex = sc.textures[i];
            tex_width[i] = tex.width;
            tex_height[i] = tex.height;
            CUDA(cudaMalloc((void**)&tex_data[i], tex.width * tex.height * 3 * sizeof(float)));
            CUDA(cudaMemcpy(tex_data[i], tex.data, tex.width * tex.height * 3 * sizeof(float), cudaMemcpyHostToDevice));
        }
        // copy tex_width to device
        CUDA(cudaMalloc((void**)&context.tex_width, sc.numTextures * sizeof(int)));
        CUDA(cudaMemcpy(context.tex_width, tex_width, sc.numTextures * sizeof(int), cudaMemcpyHostToDevice));
        // copy tex_height to device
        CUDA(cudaMalloc((void**)&context.tex_height, sc.numTextures * sizeof(int)));
        CUDA(cudaMemcpy(context.tex_height, tex_height, sc.numTextures * sizeof(int), cudaMemcpyHostToDevice));
        // copy tex_data to device
        CUDA(cudaMalloc((void**)&context.tex_data, sc.numTextures * sizeof(float*)));
        CUDA(cudaMemcpy(context.tex_data, tex_data, sc.numTextures * sizeof(float*), cudaMemcpyHostToDevice));

        delete[] tex_width;
        delete[] tex_height;
        delete[] tex_data;
    }
#endif

    context.initStats();

    return true;
}

void save(const std::string output, const RenderContext& context) {
    std::fstream out(output, std::ios::out | std::ios::binary);
    const char* HEADER = "SBK_00.01";
    out.write(HEADER, sizeof(HEADER));
    uint32_t numpaths = context.nx * context.ny * context.ns;
    out.write((char*)&numpaths, sizeof(uint32_t));
    out.write((char*)context.paths, sizeof(saved_path) * numpaths);
    out.close();
}

bool load(const std::string input, RenderContext & context) {
    std::fstream in(input, std::ios::in | std::ios::binary);
    const char* HEADER = "SBK_00.01";
    char* header = new char[sizeof(HEADER)];
    in.read(header, sizeof(HEADER));
    if (!strcmp(HEADER, header)) {
        std::cerr << "invalid header " << header << std::endl;
        return false;
    }

    uint32_t numpaths;
    in.read((char*)&numpaths, sizeof(uint32_t));
    if (numpaths != (context.nx * context.ny * context.ns)) {
        std::cerr << "numpaths doesn't match file. expected " << (context.nx * context.ny * context.ns) << ", but found " << numpaths << std::endl;
        return false;
    }

    in.read((char*)context.paths, sizeof(saved_path) * numpaths);

    in.close();

    return true;
}

std::string filename(int bounce, int ns) {
    std::stringstream str;
    str << "bounce." << ns << "." << bounce << ".sbk";
    return str.str();
}

#ifdef PRIMARY0
void iterate(RenderContext &context, int tx, int ty, bool savePaths) {
    dim3 blocks(context.nx / tx + 1, context.ny / ty + 1);
    dim3 threads(tx, ty);

    clock_t time;

    time = clock();
    primaryBounce0 <<<blocks, threads >>> (context, true);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    time = clock() - time;
    std::cerr << "bounce took " << (double)time / CLOCKS_PER_SEC << " seconds.\n";
    context.printStats();
    if (savePaths) {
        save(filename(0, context.ns), context);
    }

    for (auto i = 1; i < 8; i++) {
        time = clock();
        context.resetStats();
        bounce <<<blocks, threads >>> (context, i, true);
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());
        time = clock() - time;
        std::cerr << "bounce " << i << " took " << (double)time / CLOCKS_PER_SEC << " seconds.\n";
        context.printStats();
        if (savePaths) {
            save(filename(i, context.ns), context);
        }
    }
}
#endif

void fromfile(int bnc, RenderContext &context, int tx, int ty) {
    if (bnc > 0) {
        load(filename(bnc - 1, context.ns), context);
    }

    clock_t time;

    time = clock();
    if (bnc == 0) {
#ifdef PRIMARY0
        dim3 blocks((context.nx + tx - 1) / tx, (context.ny + ty - 1) / ty);
        dim3 threads(tx, ty);
        primaryBounce0 <<<blocks, threads >>> (context, false);
#endif
#ifdef PRIMARY1
        tx = 32; ty = 2;
        dim3 blocks((context.nx * context.ns + tx - 1) / tx, (context.ny + ty - 1) / ty);
        dim3 threads(tx, ty);
        primaryBounce1 <<<blocks, threads >>> (context, false);
#endif
#ifdef PRIMARY2
        tx = 32; ty = 2;
        dim3 blocks((context.nx * 32 + tx - 1) / tx, (context.ny + ty - 1) / ty);
        dim3 threads(tx, ty);
        primaryBounce2 <<<blocks, threads >>> (context, false);
#endif
    }
    else {
        dim3 blocks(context.nx / tx + 1, context.ny / ty + 1);
        dim3 threads(tx, ty);
        bounce <<<blocks, threads >>> (context, bnc, false);
    }
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    time = clock() - time;
    std::cerr << "bounce took " << (double)time / CLOCKS_PER_SEC << " seconds.\n";
    context.printStats();
}

int main(int argc, char** argv)
{
    bool perf = false;
    int nx = perf ? 160 : 320;
    int ny = perf ? 200 : 400;
    int ns = perf ? 4 : 64;
#ifdef PRIMARY_PERFECT
    int tx = 32;
    int ty = 2;
#else
    int tx = 8;
    int ty = 8;
#endif // PRIMARY_PERFECT

    bool save = false;

    RenderContext context;
    if (!initRenderContext(context, nx, ny, ns, save)) {
        return -1;
    }

    if (argc > 1) {
        int bnc = strtol(argv[1], NULL, 10);
        fromfile(bnc, context, tx, ty);
    } else {
#ifdef PRIMARY0
        iterate(context, tx, ty, save);
#endif
    }

    if (save) {
#ifdef PRIMARY0
        writePPM(nx, ny, context.colors);
#endif
#if defined(PRIMARY1) || defined(PRIMARY2)
        writePPM(nx, ny, ns, context.colors);
#endif

        CUDA(cudaFree(context.paths));
        CUDA(cudaFree(context.colors));
    }

    return 0;
}
