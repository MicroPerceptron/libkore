// The only native boundary of the optional Metal field runtime. All three
// buffers are owned here; no caller input/resource/source pointers survive a
// call. ARC retains Objective-C objects inside the C++ state.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <new>

struct AxMetalInfo {
    uint8_t* inputs;
    uint8_t* outputs;
    uint8_t* resources;
    size_t device_count;
    char device_name[256];
};

namespace {
struct FieldState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLBuffer> inputs = nil;
    id<MTLBuffer> outputs = nil;
    id<MTLBuffer> resources = nil;
    // Retain a submitted command even after an exception. Rust poisons the
    // field on execution errors, so its CPU mappings cannot then be reused.
    id<MTLCommandBuffer> inflight = nil;
    uint32_t width = 0;
    uint32_t height = 0;
    MTLSize group = MTLSizeMake(1, 1, 1);
};

void set_error(char* dst, size_t capacity, const char* message) noexcept {
    if (dst && capacity != 0) {
        std::snprintf(dst, capacity, "%s", message ? message : "Unknown Metal failure");
    }
}

void set_ns_error(char* dst, size_t capacity, const char* context, NSError* error) {
    if (!dst || capacity == 0) return;
    const char* detail = error ? error.localizedDescription.UTF8String : nullptr;
    std::snprintf(dst, capacity, "%s%s%s", context, detail ? ": " : "", detail ? detail : "");
}

void* create_field(const uint8_t* source_bytes, size_t source_length,
                   uint32_t width, uint32_t height,
                   size_t input_bytes, size_t output_bytes, size_t resource_bytes,
                   AxMetalInfo* info, char* error, size_t error_capacity) {
    std::unique_ptr<FieldState> state(new (std::nothrow) FieldState);
    if (!state) {
        set_error(error, error_capacity, "Cannot allocate Metal field state");
        return nullptr;
    }
    state->device = MTLCreateSystemDefaultDevice();
    if (!state->device) {
        set_error(error, error_capacity, "No native Metal device is visible");
        return nullptr;
    }
    MTLCompileOptions* options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        options.languageVersion = MTLLanguageVersion3_2;
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
        set_error(error, error_capacity, "The Metal field runtime requires macOS 15 or newer");
        return nullptr;
    }
    NSString* source = [[NSString alloc] initWithBytes:source_bytes length:source_length
                                             encoding:NSUTF8StringEncoding];
    if (!source || !options) {
        set_error(error, error_capacity, "Cannot create Metal source or compile options");
        return nullptr;
    }
    NSError* native_error = nil;
    id<MTLLibrary> library = [state->device newLibraryWithSource:source
        options:options error:&native_error];
    if (!library) {
        set_ns_error(error, error_capacity, "Metal field shader compilation failed", native_error);
        return nullptr;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"ax_field_main"];
    if (!function) {
        set_error(error, error_capacity, "Metal field entry ax_field_main was not found");
        return nullptr;
    }
    state->pipeline = [state->device newComputePipelineStateWithFunction:function error:&native_error];
    if (!state->pipeline) {
        set_ns_error(error, error_capacity, "Metal field pipeline creation failed", native_error);
        return nullptr;
    }
    state->queue = [state->device newCommandQueue];
    if (!state->queue) {
        set_error(error, error_capacity, "Metal field command queue creation failed");
        return nullptr;
    }
    const NSUInteger max_buffer = state->device.maxBufferLength;
    // A nonempty allocation is needed even for a logically empty binding.
    const size_t input_length = std::max<size_t>(input_bytes, 8);
    const size_t output_length = std::max<size_t>(output_bytes, 8);
    const size_t resource_length = std::max<size_t>(resource_bytes, 8);
    if (input_length > max_buffer || output_length > max_buffer || resource_length > max_buffer) {
        set_error(error, error_capacity, "Metal field buffer exceeds this device's maximum length");
        return nullptr;
    }
    state->inputs = [state->device newBufferWithLength:input_length options:MTLResourceStorageModeShared];
    state->outputs = [state->device newBufferWithLength:output_length options:MTLResourceStorageModeShared];
    state->resources = [state->device newBufferWithLength:resource_length options:MTLResourceStorageModeShared];
    if (!state->inputs || !state->outputs || !state->resources ||
        !state->inputs.contents || !state->outputs.contents || !state->resources.contents) {
        set_error(error, error_capacity, "Metal shared field buffer allocation failed");
        return nullptr;
    }
    // All bytes start initialized, including alignment padding. Subsequent
    // calls update only the caller INPUT slots and declared resource payloads.
    std::memset(state->inputs.contents, 0, input_length);
    std::memset(state->outputs.contents, 0, output_length);
    std::memset(state->resources.contents, 0, resource_length);
    const NSUInteger max_threads = state->pipeline.maxTotalThreadsPerThreadgroup;
    const MTLSize device_max = state->device.maxThreadsPerThreadgroup;
    const NSUInteger group_x = std::min<NSUInteger>({16, max_threads, device_max.width});
    const NSUInteger group_y = group_x == 0 ? 0 :
        std::min<NSUInteger>({8, max_threads / group_x, device_max.height});
    if (group_x == 0 || group_y == 0) {
        set_error(error, error_capacity, "Metal field has no supported threadgroup geometry");
        return nullptr;
    }
    state->width = width;
    state->height = height;
    state->group = MTLSizeMake(group_x, group_y, 1);
    info->inputs = static_cast<uint8_t*>(state->inputs.contents);
    info->outputs = static_cast<uint8_t*>(state->outputs.contents);
    info->resources = static_cast<uint8_t*>(state->resources.contents);
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    info->device_count = devices.count;
    const char* name = state->device.name.UTF8String;
    set_error(info->device_name, sizeof(info->device_name), name ? name : "Metal device");
    return state.release();
}

int run_field(FieldState* state, double* gpu_ms, char* error, size_t error_capacity) {
    if (state->inflight) {
        set_error(error, error_capacity, "Metal field has an unresolved prior command");
        return 0;
    }
    id<MTLCommandBuffer> command = [state->queue commandBuffer];
    if (!command) {
        set_error(error, error_capacity, "Metal field command buffer creation failed");
        return 0;
    }
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) {
        set_error(error, error_capacity, "Metal field command encoder creation failed");
        return 0;
    }
    [encoder setComputePipelineState:state->pipeline];
    [encoder setBuffer:state->inputs offset:0 atIndex:0];
    [encoder setBuffer:state->outputs offset:0 atIndex:1];
    [encoder setBuffer:state->resources offset:0 atIndex:2];
    const MTLSize groups = MTLSizeMake(
        (NSUInteger(state->width) + state->group.width - 1) / state->group.width,
        (NSUInteger(state->height) + state->group.height - 1) / state->group.height,
        1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:state->group];
    [encoder endEncoding];
    state->inflight = command;
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        set_ns_error(error, error_capacity, "Metal field command did not complete", command.error);
        return 0;
    }
    const double elapsed = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
    if (!(elapsed > 0.0) || !std::isfinite(elapsed)) {
        set_error(error, error_capacity, "Metal field GPU timestamps are unavailable");
        return 0;
    }
    *gpu_ms = elapsed;
    state->inflight = nil;
    return 1;
}
} // namespace

// Both Objective-C and C++ exceptions stop here. Only bounded text and a
// success/failure result cross the FFI boundary.
extern "C" void* ax_metal_field_create(
    const uint8_t* source, size_t source_length, uint32_t width, uint32_t height,
    size_t input_bytes, size_t output_bytes, size_t resource_bytes,
    AxMetalInfo* info, char* error, size_t error_capacity) noexcept {
    set_error(error, error_capacity, "");
    if (!source || source_length == 0 || width == 0 || height == 0 || !info) {
        set_error(error, error_capacity, "Invalid Metal field creation arguments");
        return nullptr;
    }
    std::memset(info, 0, sizeof(*info));
    @autoreleasepool {
        @try {
            try {
                return create_field(source, source_length, width, height, input_bytes,
                    output_bytes, resource_bytes, info, error, error_capacity);
            } catch (const std::exception& exception) {
                set_error(error, error_capacity, exception.what());
            } catch (...) {
                set_error(error, error_capacity, "C++ exception creating Metal field");
            }
        } @catch (NSException*) {
            set_error(error, error_capacity, "Objective-C exception creating Metal field");
        }
    }
    return nullptr;
}

extern "C" int ax_metal_field_run(
    void* handle, double* gpu_ms, char* error, size_t error_capacity) noexcept {
    set_error(error, error_capacity, "");
    if (!handle || !gpu_ms) {
        set_error(error, error_capacity, "Invalid Metal field execution arguments");
        return 0;
    }
    *gpu_ms = 0.0;
    @autoreleasepool {
        @try {
            try {
                return run_field(static_cast<FieldState*>(handle), gpu_ms, error, error_capacity);
            } catch (const std::exception& exception) {
                set_error(error, error_capacity, exception.what());
            } catch (...) {
                set_error(error, error_capacity, "C++ exception executing Metal field");
            }
        } @catch (NSException*) {
            set_error(error, error_capacity, "Objective-C exception executing Metal field");
        }
    }
    return 0;
}

extern "C" void ax_metal_field_destroy(void* handle) noexcept {
    if (!handle) return;
    @autoreleasepool {
        @try {
            try {
                std::unique_ptr<FieldState> state(static_cast<FieldState*>(handle));
                // Normal runs already waited. On an exceptional path, try to
                // drain accepted work before releasing the shared mappings.
                // Command buffers also retain their resources by default.
                if (state->inflight &&
                    state->inflight.status >= MTLCommandBufferStatusCommitted &&
                    state->inflight.status < MTLCommandBufferStatusCompleted) {
                    [state->inflight waitUntilCompleted];
                }
            } catch (...) {
                // Destruction cannot report an error or unwind through Rust.
            }
        } @catch (NSException*) {
            // Destruction cannot report an error or unwind through Rust.
        }
    }
}
