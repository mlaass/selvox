// Bilateral filter compute shader — edge-preserving smoothing for moiré reduction

struct BilateralUniforms {
  viewport_size: vec2<f32>,  // 0
  sigma_spatial: f32,        // 8
  sigma_color:   f32,        // 12
  sigma_depth:   f32,        // 16
};

@group(0) @binding(0) var<uniform> params: BilateralUniforms;
@group(0) @binding(1) var input_color: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bilateral_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<u32>(textureDimensions(input_color));
  if gid.x >= dims.x || gid.y >= dims.y {
    return;
  }

  let coord = vec2<i32>(gid.xy);
  let center_color = textureLoad(input_color, coord, 0);

  // Skip background pixels
  if center_color.a < 0.5 {
    textureStore(output_tex, coord, center_color);
    return;
  }

  let center_depth = textureLoad(depth_tex, coord, 0).r;

  let inv_2sigma_s2 = 1.0 / (2.0 * params.sigma_spatial * params.sigma_spatial);
  let inv_2sigma_c2 = 1.0 / (2.0 * params.sigma_color * params.sigma_color);
  let inv_2sigma_d2 = 1.0 / (2.0 * params.sigma_depth * params.sigma_depth);

  var sum_color = vec3<f32>(0.0);
  var sum_weight = 0.0;

  // 5x5 kernel (radius = 2)
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let nc = coord + vec2<i32>(dx, dy);
      let clamped = clamp(nc, vec2<i32>(0), vec2<i32>(dims) - 1);

      let sample_color = textureLoad(input_color, clamped, 0);
      let sample_depth = textureLoad(depth_tex, clamped, 0).r;

      // Spatial weight
      let dist2 = f32(dx * dx + dy * dy);
      let w_spatial = exp(-dist2 * inv_2sigma_s2);

      // Color weight
      let color_diff = sample_color.rgb - center_color.rgb;
      let color_dist2 = dot(color_diff, color_diff);
      let w_color = exp(-color_dist2 * inv_2sigma_c2);

      // Depth weight
      let depth_diff = sample_depth - center_depth;
      let w_depth = exp(-depth_diff * depth_diff * inv_2sigma_d2);

      let w = w_spatial * w_color * w_depth;
      sum_color += sample_color.rgb * w;
      sum_weight += w;
    }
  }

  let result = select(center_color.rgb, sum_color / sum_weight, sum_weight > 0.0);
  textureStore(output_tex, coord, vec4<f32>(result, 1.0));
}
