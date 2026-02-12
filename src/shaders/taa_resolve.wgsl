// TAA resolve compute shader — temporal anti-aliasing with neighborhood clamping

struct TAAUniforms {
  inv_view_proj:   mat4x4<f32>,  // 0   — current frame unjittered inverse VP
  prev_view_proj:  mat4x4<f32>,  // 64  — previous frame unjittered VP
  viewport_size:   vec2<f32>,    // 128
  jitter:          vec2<f32>,    // 136
  blend_factor:    f32,          // 144 — 0.1 normally, 1.0 on first frame
};

@group(0) @binding(0) var<uniform> taa: TAAUniforms;
@group(0) @binding(1) var current_color: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var history_tex: texture_2d<f32>;
@group(0) @binding(4) var history_sampler: sampler;
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn taa_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<u32>(textureDimensions(current_color));
  if gid.x >= dims.x || gid.y >= dims.y {
    return;
  }

  let coord = vec2<i32>(gid.xy);
  let current = textureLoad(current_color, coord, 0);

  // Skip if pixel was not written (alpha == 0 means background/discard)
  if current.a < 0.5 {
    textureStore(output_tex, coord, current);
    return;
  }

  // Read depth and reconstruct world position
  // Subtract jitter: vertex shader shifted position by jitter, but depth was written
  // with unjittered VP, so undo jitter to get the unjittered NDC matching the depth
  let depth = textureLoad(depth_tex, coord, 0).r;
  let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0) - taa.jitter;

  let clip_pos = vec4<f32>(ndc, depth, 1.0);
  let world_h = taa.inv_view_proj * clip_pos;
  let world_pos = world_h.xyz / world_h.w;

  // Reproject to previous frame
  let prev_clip = taa.prev_view_proj * vec4<f32>(world_pos, 1.0);
  let prev_ndc = prev_clip.xy / prev_clip.w;
  let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

  // Neighborhood clamping: compute 3x3 min/max of current frame
  var min_col = current.rgb;
  var max_col = current.rgb;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let nc = coord + vec2<i32>(dx, dy);
      let sample = textureLoad(current_color, clamp(nc, vec2<i32>(0), vec2<i32>(dims) - 1), 0);
      min_col = min(min_col, sample.rgb);
      max_col = max(max_col, sample.rgb);
    }
  }

  // Sample history with bilinear filtering
  var history = textureSampleLevel(history_tex, history_sampler, prev_uv, 0.0);

  // Clamp history to neighborhood
  history = vec4<f32>(clamp(history.rgb, min_col, max_col), history.a);

  // Check if reprojected UV is in bounds
  var blend = taa.blend_factor;
  if prev_clip.w <= 0.0 {
    blend = 1.0; // behind camera — no valid history
  } else if prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 {
    blend = 1.0; // no valid history — use 100% current
  }

  let result = mix(history, current, blend);
  textureStore(output_tex, coord, vec4<f32>(result.rgb, 1.0));
}
