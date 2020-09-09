
#include <stdio.h>
#include <stdint.h>
#include <bitset>
#include <time.h>
#include <algorithm>

//#define MARK_TRIANGLES

#define STATS
//#define RUSSIAN_ROULETTE
//#define SHADOW
#define TEXTURES

//#define PRIMARY_PERFECT
//#define PRIMARY0
#define PRIMARY1
//#define PRIMARY2

#define SAVE_BITSTACK

#define DUAL_NODES

#define STB_IMAGE_IMPLEMENTATION
#include "../cuda-raytracing-optimized/stb_image.h"

#include "sbk.h"

#include "../cuda-raytracing-optimized/camera.h"
#include "../cuda-raytracing-optimized/scene_materials.h"
#include "../cuda-raytracing-optimized/intersections.h"
#include "../cuda-raytracing-optimized/staircase_scene.h"

#ifdef STATS
#include <cooperative_groups.h>
using namespace cooperative_groups;
#endif

#define EPSILON 0.01f

#ifdef STATS
//#define COHERENCE

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
#define METRIC_NUM_INTERNAL             14
#define METRIC_NUM_LEAVES               15
#define METRIC_NUM_LEAF_HITS            16
#define METRIC_MAX_NUM_INTERNAL         17
#define METRIC_MAX_NUM_LEAVES           18
#define METRIC_NUM_HIGH_LEAVES          19
#define METRIC_ACTIVE                   20
#define METRIC_ACTIVE_ITER              21
#define METRIC_LEAF                     22
#define METRIC_LEAF_ITER                23
#define NUM_RAYS_SIZE                   24

char* statNames[NUM_RAYS_SIZE] = {
    " primary                     : ",
    " primary no hit              : ",
    " primary bb nohit            : ",
    " secondary                   : ",
    " secondary no hit            : ",
    " secondary bb nohit          : ",
    " shadows                     : ",
    " shadows nohit               : ",
    " shadows bb nohit            : ",
    " power < 0.01                : ",
    " exceeded max bounce         : ",
    " russiand roulette           : ",
    " *** NANs ***                : ",
    " max travers. nodes          : ",
    " num internal nodes          : ",
    " num leaf nodes              : ",
    " num leaf hits               : ",
    " max num internal            : ",
    " max num leaves              : ",
    " num paths with large leaves : ",
    " active metric               : ",
    " active.iterations           : ",
    " leaf metric                 : ",
    " leaf.iterations             : ",
};
#endif

struct RenderContext {
    int nx, ny, ns;
    camera cam;
    saved_path* paths;
    vec3* colors;

    sphere light = sphere(vec3(52.514355, 715.686951, -272.620972), 50);
    vec3 lightColor = vec3(1, 1, 1) * 80;
    triangle* tris;
#ifdef MARK_TRIANGLES
    float* triMarkers;
#endif // MARK_TRIANGLES
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

    uint32_t numpaths() const { return nx * ny * ns; }

#ifdef COHERENCE
    uint64_t* nodesCount;

    __device__ void countNode(int idx) const {
        atomicAdd(nodesCount + idx, 1);
    }
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
#ifdef COHERENCE
        CUDA(cudaMallocManaged((void**)&nodesCount, firstLeafIdx * sizeof(uint64_t)));
        memset(nodesCount, 0, firstLeafIdx * sizeof(uint64_t));
#endif
    }
    void resetStats() {
        memset(stats, 0, NUM_RAYS_SIZE * sizeof(uint64_t));
#ifdef COHERENCE
        memset(nodesCount, 0, firstLeafIdx * sizeof(uint64_t));
#endif // COHERENCE

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
#ifdef COHERENCE
        {
            uint64_t total = 0;
            uint64_t unique = 0;
            for (auto idx = 0; idx < firstLeafIdx; idx++) {
                if (nodesCount[idx] > 0) {
                    total += nodesCount[idx];
                    unique++;
                }
            }
            float coherence = (float)(total) / unique;
            std::cerr << " coherence           : " << coherence << "(" << total << "/" << unique << ")" << std::endl;
        }
#endif // COHERENCE
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

#ifdef DUAL_NODES

__device__ void pop_bitstack(unsigned int& bitStack, int& idx) {
    int m = __ffsll(bitStack) - 1;
    bitStack = (bitStack >> m) ^ 1;
    idx = (idx >> m) ^ 1;
}

#ifdef PATH_DBG
__device__ float hitBvh(const ray& r, const RenderContext& context, float t_min, tri_hit& rec, bool isShadow, bool isDebug, uint64_t sampleId) {
#else
__device__ float hitBvh(const ray & r, const RenderContext & context, float t_min, tri_hit & rec, bool isShadow) {
#endif // PATH_DBG
    int idx = 1;
    float closest = FLT_MAX;
    unsigned int bitStack = 1;
#ifdef STATS
    uint64_t numLeaves = 0;
    uint64_t numInternal = 0;
#endif // STATS

    while (idx) {
#ifdef STATS
        /*if (isShadow)*/ {
            auto g = coalesced_threads();
            if (g.thread_rank() == 0) {
                context.incStat(METRIC_ACTIVE_ITER);
                context.incStat(METRIC_ACTIVE, g.size());
            }
        }
#endif
        if (idx < context.firstLeafIdx) { // internal node
#ifdef STATS
            numInternal += 2;
#endif // STATS

            // load both children nodes
            int idx2 = idx << 1;
#ifdef COHERENCE
            context.countNode(idx2);
            context.countNode(idx2 + 1);
#endif // COHERENCE
            bvh_node left = context.bvh[idx2];
            float leftHit = hit_bbox_dist(left.min(), left.max(), r, closest);
            bool traverseLeft = leftHit < closest;
            bvh_node right = context.bvh[idx2 + 1];
            float rightHit = hit_bbox_dist(right.min(), right.max(), r, closest);
            bool traverseRight = rightHit < closest;
            bool swap = rightHit < leftHit;
            if (traverseLeft && traverseRight) {
                idx = idx2 + (swap ? 1 : 0);
                bitStack = (bitStack << 1) + 1;
            }
            else if (traverseLeft || traverseRight) {
                idx = idx2 + (swap ? 1 : 0);
                bitStack = bitStack << 1;
            }
            else {
                pop_bitstack(bitStack, idx);
            }
        }
        else { // leaf node
#ifdef STATS
            /*if (isShadow)*/ {
                auto g = coalesced_threads();
                if (g.thread_rank() == 0) {
                    context.incStat(METRIC_LEAF_ITER);
                    context.incStat(METRIC_LEAF, g.size());
                }
            }
            numLeaves++;
#endif
            int first = (idx - context.firstLeafIdx) * context.numPrimitivesPerLeaf;
#ifdef STATS
#ifdef PATH_DBG
            //if (isDebug) printf("%d\n", first);
#endif // PATH_DBG
            bool found = false;
#endif // STATS
            for (auto i = 0; i < context.numPrimitivesPerLeaf; i++) {
                const triangle tri = context.tris[first + i];
                if (isinf(tri.v[0].x()))
                    break; // we reached the end of the primitives buffer
                float u, v;
                float hitT = triangleHit(tri, r, t_min, closest, u, v);
                if (hitT < closest) {
                    if (isShadow) return 0.0f;
#ifdef STATS
                    found = true;
#endif // STATS
                    closest = hitT;
                    rec.triId = first + i;
#ifdef SAVE_BITSTACK
                    rec.bitstack = bitStack;
#endif // SAVE_BITSTACK
                    rec.u = u;
                    rec.v = v;
                }
            }
            pop_bitstack(bitStack, idx);
#ifdef STATS
            if (found) context.incStat(METRIC_NUM_LEAF_HITS);
#endif // STATS

        }
    }

#ifdef STATS
    context.incStat(METRIC_NUM_INTERNAL, numInternal);
    context.incStat(METRIC_NUM_LEAVES, numLeaves);

    context.maxStat(METRIC_MAX_NUM_LEAVES, numLeaves);
    context.maxStat(METRIC_MAX_NUM_INTERNAL, numInternal);

    if (numLeaves > 199) context.incStat(METRIC_NUM_HIGH_LEAVES);
#ifdef COLOR_NUM_NODES
    rec.numNodes = /*numInternal +*/ numLeaves;
#endif // COLOR_NUM_NODES
#endif // STATS

    return closest;
}
#else
__device__ float hitBvh(const ray& r, const RenderContext& context, float t_min, tri_hit& rec, bool isShadow, bool isDebug) {
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
#ifdef COHERENCE
            context.countNode(idx);
#endif // COHERENCE
            bvh_node node = context.bvh[idx];
            if (hit_bbox(node.min(), node.max(), r, closest)) {
                if (idx >= context.firstLeafIdx) { // leaf node
#ifdef PATH_DBG
                    if (isDebug) printf("LEAF HIT\n");
#endif // PATH_DBG
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
                            //rec.bitstack = bitStack;
                        }
                    }
                    down = false;
                }
                else { // internal node
#ifdef PATH_DBG
                    if (isDebug) printf("DOWN ");
#endif // PATH_DBG
                    // current -> left or right
                    const int childIdx = signbit(r.direction()[node.split_axis()]); // 0=left, 1=right
                    bitStack = (bitStack << 1) + childIdx; // push current child idx in the stack
                    idx = (idx << 1) + childIdx;
                }
            }
            else { // ray didn't intersect the node, backtrack
#ifdef PATH_DBG
                if (idx >= context.firstLeafIdx) { // leaf node
                    if (isDebug) printf("LEAF MISS\n");
                } else {
                    if (isDebug) printf("NODE MISS\n");
                }
#endif // PATH_DBG
                down = false;
            }
        }
        else if (idx == 1) { // we backtracked up to the root node
            break;
        }
        else { // back tracking
            const int currentChildIdx = bitStack & 1;
            if ((idx & 1) == currentChildIdx) { // node == current child, visit sibling
#ifdef PATH_DBG
                if (isDebug) printf("RIGHT ");
#endif // PATH_DBG
                idx += -2 * currentChildIdx + 1; // node = node.sibling
                down = true;
            }
            else { // we visited both siblings, backtrack
#ifdef PATH_DBG
                if (isDebug) printf("BACK ");
#endif // PATH_DBG
                bitStack = bitStack >> 1;
                idx = idx >> 1; // node = node.parent
            }
        }
    }

    return closest;
}
#endif

#ifdef PATH_DBG
__device__ float hitMesh(const ray& r, const RenderContext& context, float t_min, float t_max, tri_hit& rec, bool primary, bool isShadow, bool isDebug, uint64_t sampleId) {
#else
__device__ float hitMesh(const ray & r, const RenderContext & context, float t_min, float t_max, tri_hit & rec, bool primary, bool isShadow) {
#endif // PATH_DBG
    if (!hit_bbox(context.bounds.min, context.bounds.max, r, t_max)) {
#ifdef STATS
        if (isShadow) context.incStat(NUM_RAYS_SHADOWS_BBOX_NOHITS);
        else context.incStat(primary ? NUM_RAYS_PRIMARY_BBOX_NOHITS : NUM_RAYS_SECONDARY_BBOX_NOHIT);
#endif
        return FLT_MAX;
    }

#ifdef PATH_DBG
    return hitBvh(r, context, t_min, rec, isShadow, isDebug, sampleId);
#else
    return hitBvh(r, context, t_min, rec, isShadow);
#endif // PATH_DBG
}

__device__ bool hit(const RenderContext& context, const path& p, bool isShadow, intersection& inters) {
    const ray r = isShadow ? ray(p.origin, p.shadowDir) : ray(p.origin, p.rayDir);
    tri_hit triHit;
    bool primary = p.bounce == 0;
    inters.objId = NONE;
#ifdef PATH_DBG
    inters.t = hitMesh(r, context, EPSILON, FLT_MAX, triHit, primary, isShadow, p.dbg, p.sampleId);
#else
    inters.t = hitMesh(r, context, EPSILON, FLT_MAX, triHit, primary, isShadow);
#endif // PATH_DBG

    if (inters.t < FLT_MAX) {
        if (isShadow) return true; // we don't need to compute the intersection details for shadow rays

        inters.objId = TRIMESH;
        triangle tri = context.tris[triHit.triId];
        inters.meshID = tri.meshID;
        inters.triID = triHit.triId;
        inters.normal = unit_vector(cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[0]));
        inters.texCoords[0] = (triHit.u * tri.texCoords[1 * 2 + 0] + triHit.v * tri.texCoords[2 * 2 + 0] + (1 - triHit.u - triHit.v) * tri.texCoords[0 * 2 + 0]);
        inters.texCoords[1] = (triHit.u * tri.texCoords[1 * 2 + 1] + triHit.v * tri.texCoords[2 * 2 + 1] + (1 - triHit.u - triHit.v) * tri.texCoords[0 * 2 + 1]);
        inters.bitstack = triHit.bitstack;
#ifdef COLOR_NUM_NODES
        inters.numNodes = triHit.numNodes;
#endif // COLOR_NUM_NODES
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

#ifdef MARK_TRIANGLES
__device__ vec3 hsv2rgb(float h, float s, float v) {
    int h_i = (h * 6);
    float f = h * 6 - h_i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    if (h_i == 0) return vec3(v, t, p);
    if (h_i == 1) return vec3(q, v, p);
    if (h_i == 2) return vec3(p, v, t);
    if (h_i == 3) return vec3(p, q, v);
    if (h_i == 4) return vec3(t, p, v);
    return vec3(v, p, q);
}
#endif // MARK_TRIANGLES

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
#ifdef MARK_TRIANGLES
            float marker = context.triMarkers[inters.triID];
            if (marker == 0.0f) {
                p.color = vec3(0, 0, 0);
            } else if (marker < 0.5) {
                float t = marker * 2;
                p.color = (1.0f - t) * vec3(0, 0, 1) + t * vec3(0, 1, 0);
            } else {
                float t = (marker - 0.5f) * 2;
                p.color = (1.0f - t) * vec3(0, 1, 0) + t * vec3(1, 0, 0);
            }
#elif defined(COLOR_NUM_NODES)
            float marker = inters.numNodes < 300 ? inters.numNodes / 300.0f : 1.0f;
            if (marker == 0.0f) {
                p.color = vec3(0, 0, 0);
            }
            else if (marker < 0.5) {
                float t = marker * 2;
                p.color = (1.0f - t) * vec3(0, 0, 1) + t * vec3(0, 1, 0);
            }
            else {
                float t = (marker - 0.5f) * 2;
                p.color = (1.0f - t) * vec3(0, 1, 0) + t * vec3(1, 0, 0);
            }
#else
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
#endif // MARK_TRIANGLES
        }
        else
            floor_diffuse_scatter(scatter, inters, p.rayDir, p.rng);

        p.origin += scatter.t * p.rayDir;
        p.rayDir = scatter.wi;
        p.attenuation *= scatter.throughput;
        p.specular = scatter.specular;
        p.inside = scatter.refracted ? !p.inside : p.inside;
        p.bitstack = inters.bitstack;
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
    uint32_t id = y * context.nx + x;
    for (int s = 0; s < context.ns; s++) {
        const uint32_t pid = id * context.ns + s;
#ifdef PATH_DBG
        p.dbg = pid == (128 * context.nx + 100) * context.ns + 10;
#endif
        saved_path sp = context.paths[pid];
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
#ifdef STATS
        if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif
        if (save) {
            // all samples for same pixel are saved in consecutive order
            context.paths[pid] = saved_path(p, sp.sampleId);
            context.colors[sp.sampleId] = p.color;
        }
    }
}

__global__ void bounceSorted(const RenderContext context, int bounce, bool save) {
    int xs = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((xs >= (context.nx * context.ns)) || (y >= context.ny)) return;
    //int x = xs / context.ns;
    //int s = xs % context.ns;

    uint32_t pid = y * context.nx * context.ns + xs;
    saved_path sp = context.paths[pid];
    if (sp.isDone()) return;

    path p;
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
#ifdef STATS
    if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif
    if (save) {
        // all samples for same pixel are saved in consecutive order
        context.paths[pid] = saved_path(p, sp.sampleId);
        context.colors[sp.sampleId] = p.color;
    }
}

__global__ void primary(const RenderContext context) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= context.nx) || (y >= context.ny)) return;

    path p;
    uint32_t pixelId = y * context.nx + x;
    p.rng = (wang_hash(pixelId) * 336343633) | 1;

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
        uint32_t sampleId = pixelId * context.ns + s;
        context.paths[sampleId] = saved_path(p, sampleId);
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
    uint32_t pixelId = y * context.nx + x;
    p.rng = (wang_hash(pixelId) * 336343633) | 1;

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
#ifdef STATS
        if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif

        if (save) {
            // all samples for same pixel are saved in consecutive order
            uint32_t sampleId = pixelId * context.ns + s;
            context.paths[sampleId] = saved_path(p, sampleId);
            context.colors[sampleId] = p.color;
        }
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
    uint32_t pixelId = y * context.nx + x;
    int sampleId = pixelId * context.ns + s;
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
#ifdef PATH_DBG
    p.sampleId = sampleId;
    p.dbg = sampleId == 7402054;
#endif // PATH_DBG

    colorBounce(context, p);
#ifdef STATS
    if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif

    // following code is not used, but leave it here for now in case compiler is smart enough to undo all our work if we don't save anything
    if (save) {
        // all samples for same pixel are saved in consecutive order
        context.paths[sampleId] = saved_path(p, sampleId);
        context.colors[sampleId] = p.color;
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
    uint32_t pixelId = y * context.nx + x;
    int sampleId = pixelId * context.ns + s;
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
#ifdef STATS
        if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif

        if (save) {
            // all samples for same pixel are saved in consecutive order
            context.paths[sampleId] = saved_path(p);
            context.colors[sampleId] = p.color;
        }
    }
}
#endif // PRIMARY2

#ifdef MARK_TRIANGLES
float computeMarker(const triangle& t) {
    // compute bounds volume
    vec3 tmin = min(t.v[0], min(t.v[1], t.v[2]));
    vec3 tmax = max(t.v[0], max(t.v[1], t.v[2]));
    vec3 boundsExtent = tmax - tmin;
    float bVolume = 1.0f;
    if (boundsExtent.x() > 0.0001f) bVolume *= boundsExtent.x();
    if (boundsExtent.y() > 0.0001f) bVolume *= boundsExtent.y();
    if (boundsExtent.z() > 0.0001f) bVolume *= boundsExtent.z();

    // compute triangle area
    float tArea = cross(t.v[2] - t.v[0], t.v[1] - t.v[0]).length() / 2;

    float value = 1.0f - tArea / bVolume;
    float threshold = 0.9f;
    if (value < threshold) return 0.0f;
    return (value - threshold) / (1 - threshold);
}
#endif // MARK_TRIANGLES

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

    uint32_t numpaths = context.nx * context.ny * context.ns;
    CUDA(cudaMallocManaged((void**)&context.paths, numpaths * sizeof(saved_path)));

#ifdef MARK_TRIANGLES
    {
        float* triMarkers = new float[ksc.m->numTris];
        for (size_t i = 0; i < ksc.m->numTris; i++) {
            triMarkers[i] = computeMarker(ksc.m->tris[i]);
        }
        CUDA(cudaMalloc((void**)&context.triMarkers, ksc.m->numTris * sizeof(float)));
        CUDA(cudaMemcpy(context.triMarkers, triMarkers, ksc.m->numTris * sizeof(float), cudaMemcpyHostToDevice));
        delete[] triMarkers;
    }
#endif // MARK_TRIANGLES

    if (save) {
        // store each color sample separately
        uint32_t size = context.numpaths() * sizeof(vec3);
        CUDA(cudaMallocManaged((void**)&context.colors, size));
        memset(context.colors, 0, size);
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

#ifdef PRIMARY0
#ifdef SAVE_BITSTACK
void iterate(RenderContext& context, int tx, int ty, bool savePaths) {
    dim3 blocks(context.nx / tx + 1, context.ny / ty + 1);
    dim3 threads(tx, ty);

    clock_t time;

    // when storing bitstack we need to make a copy of the paths until we collect their bitstacks in the next bounce then we can save them
    saved_path* paths = NULL;
    if (savePaths) {
        paths = new saved_path[context.numpaths()];
    }

    time = clock();
    primaryBounce0 << <blocks, threads >> > (context, savePaths);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    time = clock() - time;
    std::cerr << "bounce took " << (double)time / CLOCKS_PER_SEC << " seconds.\n";
    context.printStats();
    if (savePaths) {
        // just copy the paths but do not save them
        memcpy((void*)paths, (void*)context.paths, context.numpaths() * sizeof(saved_path));
    }

    // even though we only save 7 bounces, we need to trace the 8th bounce to collect the bitstacks
    for (auto i = 1; i <= 8; i++) {
        time = clock();
        context.resetStats();
        bounce << <blocks, threads >> > (context, i, savePaths);
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());
        time = clock() - time;
        std::cerr << "bounce " << i << " took " << (double)time / CLOCKS_PER_SEC << " seconds.\n";
        context.printStats();
        if (savePaths) {
            // copy bitstacks from context.paths to paths
            for (auto j = 0; j < context.numpaths(); j++) {
                paths[j].bitstack = context.paths[j].bitstack;
            }
            // now we can save previous bounce's paths
            save(filename(i - 1, context.ns, false), paths, context.numpaths());
            // copy current paths to local variable
            memcpy((void*)paths, (void*)context.paths, context.numpaths() * sizeof(saved_path));
        }
    }
}
#else
void iterate(RenderContext& context, int tx, int ty, bool savePaths) {
    dim3 blocks(context.nx / tx + 1, context.ny / ty + 1);
    dim3 threads(tx, ty);

    clock_t time;

    time = clock();
    primaryBounce0 << <blocks, threads >> > (context, savePaths);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    time = clock() - time;
    std::cerr << "bounce took " << (double)time / CLOCKS_PER_SEC << " seconds.\n";
    context.printStats();
    if (savePaths) {
        save(filename(0, context.ns, false), context.paths, context.numpaths());
    }

    for (auto i = 1; i < 1; i++) {
        time = clock();
        context.resetStats();
        bounce << <blocks, threads >> > (context, i, savePaths);
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());
        time = clock() - time;
        std::cerr << "bounce " << i << " took " << (double)time / CLOCKS_PER_SEC << " seconds.\n";
        context.printStats();
        if (savePaths) {
            save(filename(i, context.ns, false), context.paths, context.numpaths());
        }
    }
}
#endif // SAVE_BITSTACK
#endif

void fromfile(int bnc, RenderContext &context, int tx, int ty, bool save, bool sorted) {
    if (bnc > 0) {
        uint32_t numpaths = context.nx * context.ny * context.ns;
        load(filename(bnc - 1, context.ns, sorted), context.paths, numpaths);
    }

    clock_t time;

    time = clock();
    if (bnc == 0) {
#ifdef PRIMARY0
        dim3 blocks((context.nx + tx - 1) / tx, (context.ny + ty - 1) / ty);
        dim3 threads(tx, ty);
        primaryBounce0 <<<blocks, threads >>> (context, save);
#endif
#ifdef PRIMARY1
        tx = 32; ty = 2;
        dim3 blocks((context.nx * context.ns + tx - 1) / tx, (context.ny + ty - 1) / ty);
        dim3 threads(tx, ty);
        primaryBounce1 <<<blocks, threads >>> (context, save);
#endif
#ifdef PRIMARY2
        tx = 32; ty = 2;
        dim3 blocks((context.nx * 32 + tx - 1) / tx, (context.ny + ty - 1) / ty);
        dim3 threads(tx, ty);
        primaryBounce2 <<<blocks, threads >>> (context, save);
#endif
    }
    else {
        if (sorted) {
            tx = 32; ty = 2;
            dim3 blocks((context.nx * context.ns + tx - 1) / tx, (context.ny + ty - 1) / ty);
            dim3 threads(tx, ty);
            bounceSorted <<<blocks, threads >>> (context, bnc, save);
        } else {
            dim3 blocks(context.nx / tx + 1, context.ny / ty + 1);
            dim3 threads(tx, ty);
            bounce <<<blocks, threads >>> (context, bnc, save);
        }
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
    int nx = perf ? 160 : 1280;
    int ny = perf ? 200 : 1600;
    int ns = perf ? 4 : 4;
#ifdef PRIMARY_PERFECT
    int tx = 32;
    int ty = 2;
#else
    int tx = 8;
    int ty = 8;
#endif // PRIMARY_PERFECT

    bool save = false;
    bool sorted = false;
    int bnc = -1;
    if (argc > 1) {
        bnc = strtol(argv[1], NULL, 10);
        for (auto i = 2; i < argc; i++) {
            if (!strcmp("--save", argv[i])) save = true;
            if (!strcmp("--sorted", argv[i])) sorted = true;
        }
    }

    RenderContext context;
    if (!initRenderContext(context, nx, ny, ns, save)) {
        return -1;
    }

    if (bnc >= 0) {
        fromfile(bnc, context, tx, ty, save, sorted);
    } else {
#ifdef PRIMARY0
        iterate(context, tx, ty, save);
#endif
    }

    if (save) {
        writePPM(nx, ny, ns, context.colors);
        CUDA(cudaFree(context.colors));
    }

    CUDA(cudaFree(context.paths));

    return 0;
}
