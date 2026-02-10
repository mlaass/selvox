// Placeholder vertex + fragment shader for Phase 1.
// Renders a full-screen triangle with a solid color.

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  // Full-screen triangle positions
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );

  var out: VertexOutput;
  out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  out.color = vec3<f32>(0.15, 0.15, 0.2);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
