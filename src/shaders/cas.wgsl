// AMD Contrast Adaptive Sharpening (CAS) — compute shader post-process

struct CASUniforms {
  viewport_size: vec2<f32>,  // 0
  sharpness:     f32,        // 8  — 0.0 = off, 1.0 = max
};

@group(0) @binding(0) var<uniform> cas: CASUniforms;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cas_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<u32>(textureDimensions(input_tex));
  if gid.x >= dims.x || gid.y >= dims.y {
    return;
  }

  let coord = vec2<i32>(gid.xy);

  // 5-tap cross pattern: center + N/S/E/W
  let c = textureLoad(input_tex, coord, 0).rgb;
  let n = textureLoad(input_tex, clamp(coord + vec2<i32>(0, -1), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
  let s = textureLoad(input_tex, clamp(coord + vec2<i32>(0,  1), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
  let w = textureLoad(input_tex, clamp(coord + vec2<i32>(-1, 0), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;
  let e = textureLoad(input_tex, clamp(coord + vec2<i32>( 1, 0), vec2<i32>(0), vec2<i32>(dims) - 1), 0).rgb;

  // Compute local min/max for contrast detection
  let local_min = min(c, min(min(n, s), min(w, e)));
  let local_max = max(c, max(max(n, s), max(w, e)));

  // Adaptive sharpening weight from local contrast
  // Higher contrast = less sharpening (already sharp), lower contrast = more sharpening
  let contrast = local_max - local_min;
  let rcp_max = 1.0 / (local_max + vec3<f32>(0.001));
  // Weight per channel: sqrt(min/max) approximates inverse contrast
  let w_raw = sqrt(saturate(local_min * rcp_max));
  // Average across channels for a single weight
  let weight = (w_raw.r + w_raw.g + w_raw.b) / 3.0;

  // Scale by sharpness control: map sharpness [0,1] to kernel weight [-0.125, -0.5]
  // More negative = stronger sharpening
  let sharp_w = mix(-0.125, -0.5, cas.sharpness) * weight;

  // Apply weighted sum: center + sharp_w * (neighbors - 4*center)
  // Rearranged: (1 - 4*sharp_w) * center + sharp_w * (n + s + w + e)
  let result = (1.0 - 4.0 * sharp_w) * c + sharp_w * (n + s + w + e);

  textureStore(output_tex, coord, vec4<f32>(max(result, vec3<f32>(0.0)), 1.0));
}
