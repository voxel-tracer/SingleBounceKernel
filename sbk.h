#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../cuda-raytracing-optimized/helper_structs.h"

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

struct saved_path {
    vec3 origin;
    vec3 rayDir;
    vec3 attenuation; // only needed to visually confirm the rendering is correct

    uint32_t sampleId;
    uint8_t flags;
    rand_state rng;
#ifdef SAVE_BITSTACK
    unsigned int bitstack = 0;
#endif // SAVE_BITSTACK


    __host__ saved_path() {}
    __device__ saved_path(const path& p, uint32_t sampleId) : origin(p.origin), rayDir(p.rayDir), flags(FLAGS(p)), attenuation(p.attenuation), rng(p.rng), sampleId(sampleId) {
#ifdef SAVE_BITSTACK
        bitstack = p.bitstack;
#endif // SAVE_BITSTACK
    }

    __device__ bool isDone() const { return flags & FLAG_DONE; }
    __device__ bool isSpecular() const { return flags & FLAG_SPECULAR; }
    __device__ bool isInside() const { return flags & FLAG_INSIDE; }
};

// SBK_00.02: added sampleId
// SBK_00.03: added bitstack
bool load(const std::string input, saved_path* paths, uint32_t expectedNumPaths) {
    std::fstream in(input, std::ios::in | std::ios::binary);
    const char* HEADER = "SBK_00.03";
    char* header = new char[sizeof(HEADER)];
    in.read(header, sizeof(HEADER));
    if (!strcmp(HEADER, header)) {
        std::cerr << "invalid header " << header << std::endl;
        return false;
    }

    uint32_t numpaths;
    in.read((char*)&numpaths, sizeof(uint32_t));
    if (numpaths != expectedNumPaths) {
        std::cerr << "numpaths doesn't match file. expected " << expectedNumPaths << ", but found " << numpaths << std::endl;
        return false;
    }

    in.read((char*)paths, sizeof(saved_path) * numpaths);

    in.close();

    return true;
}

void save(const std::string output, const saved_path* paths, uint32_t numpaths) {
    std::fstream out(output, std::ios::out | std::ios::binary);
    const char* HEADER = "SBK_00.03";
    out.write(HEADER, sizeof(HEADER));
    out.write((char*)&numpaths, sizeof(uint32_t));
    out.write((char*)paths, sizeof(saved_path) * numpaths);
    out.close();
}

std::string filename(int bounce, int ns, bool sorted) {
    std::stringstream str;
    str << "bounce." << ns << "." << bounce << ".sbk";
    if (sorted) str << ".sorted";
    return str.str();
}
