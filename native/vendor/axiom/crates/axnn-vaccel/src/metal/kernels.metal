// Resource helpers prepended to the generated .ax field kernel.
//
// This is an Approximate implementation, even with precise math and these
// operation-order controls. Native Metal may flush subnormal intermediates;
// finite extreme rays have not established CPU first-hit bit parity.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

struct AxMetalVoxelHit { float t; int cell_id; };

inline AxMetalVoxelHit ax_voxel_miss() {
    return AxMetalVoxelHit{INFINITY, -1};
}

inline float ax_voxel_sanitize_direction(float d) {
    return fabs(d) < 1e-12f ? copysign(1e-12f, d) : d;
}

// Rust's float-to-int cast saturates, including NaN -> 0. Avoid an out-of-range
// Metal conversion before clamping a computed entry position to its grid.
inline int ax_voxel_trunc_clamp(float value, int last) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x4f000000u) {
        if (magnitude > 0x7f800000u || (bits & 0x80000000u) != 0u) return 0;
        return last;
    }
    return clamp(int(value), 0, last);
}

// Scalar-order port of portable::voxel::brick_trace. The typed binding validates
// positive multiple-of-four dimensions and signed-I32 cell IDs before upload.
// There is no hierarchy, reciprocal estimate, native RT, or ray answer cache.
inline AxMetalVoxelHit ax_voxel_brick_trace(
    device const uchar* resource, uint3 dims,
    float3 origin, float3 direction, float limit) {
    const uint3 origin_bits = as_type<uint3>(origin);
    const uint3 direction_bits = as_type<uint3>(direction);
    const uint limit_bits = as_type<uint>(limit);
    const uint limit_magnitude = limit_bits & 0x7fffffffu;
    // Bit tests also reject negative subnormal limits on hardware that treats
    // them as zero in floating comparisons. Positive infinity and either zero
    // sign remain admitted, as in the checked CPU resource contract.
    if (any((origin_bits & 0x7f800000u) == 0x7f800000u) ||
        any((direction_bits & 0x7f800000u) == 0x7f800000u) ||
        limit_magnitude > 0x7f800000u ||
        ((limit_bits & 0x80000000u) != 0u && limit_magnitude != 0u)) {
        return ax_voxel_miss();
    }
    device const ulong* bricks = reinterpret_cast<device const ulong*>(resource);
    const float d[3] = {
        ax_voxel_sanitize_direction(direction.x),
        ax_voxel_sanitize_direction(direction.y),
        ax_voxel_sanitize_direction(direction.z)
    };
    float t0 = 0.0f;
    float t1 = limit;
    for (uint a = 0u; a < 3u; ++a) {
        const float inv = 1.0f / d[a];
        const float ta = (0.0f - origin[a]) * inv;
        const float tb = (float(dims[a]) - origin[a]) * inv;
        const float lo = ta < tb ? ta : tb;
        const float hi = ta < tb ? tb : ta;
        // Keep +0 on an equal entry boundary, as the native Rust oracle does.
        if (lo > t0) t0 = lo;
        t1 = fmin(t1, hi);
    }
    if (t0 > t1) return ax_voxel_miss();

    const uint3 bdims = dims / 4u;
    int cell[3];
    int step[3];
    float t_next[3];
    float t_delta[3];
    for (uint a = 0u; a < 3u; ++a) {
        const float p = origin[a] + t0 * d[a];
        const int c = ax_voxel_trunc_clamp(p / 4.0f, int(bdims[a]) - 1);
        cell[a] = c;
        step[a] = d[a] > 0.0f ? 1 : -1;
        t_delta[a] = 4.0f / fabs(d[a]);
        const float boundary = d[a] > 0.0f ? float(c + 1) * 4.0f : float(c) * 4.0f;
        t_next[a] = t0 + (boundary - p) / d[a];
    }
    const int nx = int(dims.x);
    const int ny = int(dims.y);
    const int bnx = int(bdims.x);
    const int bny = int(bdims.y);
    float t = t0;
    for (;;) {
        const ulong word = bricks[cell[0] + bnx * (cell[1] + bny * cell[2])];
        if (word != 0ul) {
            const int base[3] = {cell[0] * 4, cell[1] * 4, cell[2] * 4};
            int v[3];
            float vt_next[3];
            for (uint a = 0u; a < 3u; ++a) {
                const float p = origin[a] + t * d[a];
                const int l = ax_voxel_trunc_clamp(p - float(base[a]), 3);
                v[a] = l;
                const float boundary = float(base[a]) +
                    (d[a] > 0.0f ? float(l + 1) : float(l));
                vt_next[a] = t + (boundary - p) / d[a];
            }
            const float vt_delta[3] = {
                1.0f / fabs(d[0]), 1.0f / fabs(d[1]), 1.0f / fabs(d[2])
            };
            float vt = t;
            for (;;) {
                if (((word >> uint(v[0] + 4 * (v[1] + 4 * v[2]))) & 1ul) != 0ul) {
                    const int gx = base[0] + v[0];
                    const int gy = base[1] + v[1];
                    const int gz = base[2] + v[2];
                    return AxMetalVoxelHit{vt, gx + nx * (gy + ny * gz)};
                }
                const uint a = vt_next[0] <= vt_next[1] && vt_next[0] <= vt_next[2]
                    ? 0u : (vt_next[1] <= vt_next[2] ? 1u : 2u);
                vt = vt_next[a];
                if (vt > t1) return ax_voxel_miss();
                v[a] += step[a];
                if (v[a] < 0 || v[a] >= 4) break;
                vt_next[a] += vt_delta[a];
            }
        }
        const uint a = t_next[0] <= t_next[1] && t_next[0] <= t_next[2]
            ? 0u : (t_next[1] <= t_next[2] ? 1u : 2u);
        t = t_next[a];
        if (t > t1) return ax_voxel_miss();
        cell[a] += step[a];
        if (cell[a] < 0 || cell[a] >= int(bdims[a])) return ax_voxel_miss();
        t_next[a] += t_delta[a];
    }
}

// The unsigned E5M11 resource is not IEEE F16. Decode its eleventh fraction bit
// directly into F32, including its finite subnormal words (all normal in F32).
inline float ax_volume_decode_uf16(ushort word) {
    const uint raw = uint(word);
    const uint exponent = raw >> 11u;
    const uint mantissa = raw & 0x7ffu;
    if (exponent == 0u) return float(mantissa) * 0x1p-25f;
    return as_type<float>(((exponent + 112u) << 23u) | (mantissa << 12u));
}

inline float ax_volume_decode_uf8e5m3(uchar byte) {
    const uint raw = uint(byte);
    if (raw < 8u) return float(raw) * 0x1p-17f;
    return as_type<float>((raw << 20u) + (112u << 23u));
}

template <uint STORAGE>
inline float3 ax_volume_trilinear_rgb(
    device const uchar* resource, uint3 dims, float3 point) {
    const uint3 magnitude = as_type<uint3>(point) & 0x7fffffffu;
    if (any(magnitude > 0x7f800000u)) return float3(as_type<float>(0x7fc00000u));
    uint lo[3] = {0u, 0u, 0u};
    uint hi[3] = {0u, 0u, 0u};
    float fraction[3] = {0.0f, 0.0f, 0.0f};
    for (uint a = 0u; a < 3u; ++a) {
        if (dims[a] != 1u) {
            const float last = float(dims[a] - 1u);
            // NaNs were rejected above. Preserve input -0 on equality, as
            // Rust clamp does; infinities clamp to the endpoint explicitly.
            const float coordinate = point[a] < 0.0f ? 0.0f :
                (point[a] > last ? last : point[a]);
            lo[a] = uint(coordinate);
            hi[a] = min(lo[a] + 1u, dims[a] - 1u);
            fraction[a] = coordinate - float(lo[a]);
        }
    }
    float3 result = float3(0.0f);
    for (uint z = 0u; z < 2u; ++z) {
        for (uint y = 0u; y < 2u; ++y) {
            for (uint x = 0u; x < 2u; ++x) {
                const uint ix = x == 0u ? lo[0] : hi[0];
                const uint iy = y == 0u ? lo[1] : hi[1];
                const uint iz = z == 0u ? lo[2] : hi[2];
                const float wx = x == 0u ? 1.0f - fraction[0] : fraction[0];
                const float wy = y == 0u ? 1.0f - fraction[1] : fraction[1];
                const float wz = z == 0u ? 1.0f - fraction[2] : fraction[2];
                const float weight = (wx * wy) * wz;
                const ulong index = 3ul * (ulong(ix) + ulong(dims.x) *
                    (ulong(iy) + ulong(dims.y) * ulong(iz)));
                float3 value;
                if (STORAGE == 2u) {
                    value = float3(ax_volume_decode_uf8e5m3(resource[index]),
                        ax_volume_decode_uf8e5m3(resource[index + 1ul]),
                        ax_volume_decode_uf8e5m3(resource[index + 2ul]));
                } else if (STORAGE == 1u) {
                    device const ushort* words = reinterpret_cast<device const ushort*>(resource);
                    value = float3(ax_volume_decode_uf16(words[index]),
                        ax_volume_decode_uf16(words[index + 1ul]),
                        ax_volume_decode_uf16(words[index + 2ul]));
                } else {
                    device const float* values = reinterpret_cast<device const float*>(resource);
                    value = float3(values[index], values[index + 1ul], values[index + 2ul]);
                }
                // Do not fuse, reassociate, or skip zero-weight corners.
                const float3 product = value * weight;
                result = result + product;
            }
        }
    }
    return result;
}

inline float3 ax_volume_trilinear_rgb_f32(
    device const uchar* resource, uint3 dims, float3 point) {
    return ax_volume_trilinear_rgb<0u>(resource, dims, point);
}

inline float3 ax_volume_trilinear_rgb_uf16(
    device const uchar* resource, uint3 dims, float3 point) {
    return ax_volume_trilinear_rgb<1u>(resource, dims, point);
}

inline float3 ax_volume_trilinear_rgb_uf8e5m3(
    device const uchar* resource, uint3 dims, float3 point) {
    return ax_volume_trilinear_rgb<2u>(resource, dims, point);
}

inline int ax_buffer_load_u8(device const uchar* resource, uint elements, int index) {
    if (index < 0 || uint(index) >= elements) return 0;
    return int(resource[uint(index)]);
}
