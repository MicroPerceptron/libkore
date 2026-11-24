// The welcome's overlay over its black hole: the second of its two passes,
// at the display's resolution. `hole` is the first pass's colour (encoded,
// upsampled to the display where the hole was drawn at the lens map's
// resolution). The runtime prepares the overlay's coverage, as it does its
// text: the seat pill's shadow, dark fill and avatar arrive as `shape`; the
// chips' and the pill's faint edges and the Enter button share the light
// ink's cream, so they ride `ink.x`. Colours and their order are all here.
// Where nothing covers a pixel this returns `hole` exactly, so tiles without
// overlay skip this pass and pack the hole directly, with no seam.
@fragment
fn main(@builtin(position) position: vec4<f32>,
        @location(0) hole: vec3<f32>,
        @location(1) ink: vec2<f32>,
        @location(2) shape: vec3<f32>) -> @location(0) vec4<f32> {
    let c1 = hole * (1.0 - shape.x);
    let c2 = mix(c1, vec3<f32>(0.078, 0.071, 0.094), shape.y);
    let c4 = mix(c2, vec3<f32>(0.247, 0.682, 0.478), shape.z);
    let c5 = mix(c4, vec3<f32>(0.953, 0.937, 0.910), clamp(ink.x, 0.0, 1.0));
    let c9 = mix(c5, vec3<f32>(0.075, 0.110, 0.094), clamp(ink.y, 0.0, 1.0));
    return vec4<f32>(clamp(c9, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
