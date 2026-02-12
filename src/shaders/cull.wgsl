// GPU compute culling — frustum cull + 8-corner projection + subpixel cull

struct CullUniforms {
  view_proj:      mat4x4<f32>,   // 0
  frustum_planes: array<vec4<f32>, 6>,  // 64  (6 * 16 = 96 bytes)
  viewport_size:  vec2<f32>,     // 160
  interpolation:  f32,           // 168
  voxel_count:    u32,           // 172
  rte_offset:     vec3<f32>,     // 176
  lod_level:      u32,           // 188
  start_slot:     u32,           // 192
  chunk_index:    u32,           // 196
  subpixel_threshold: f32,      // 200
};

struct Voxel {
  pos_a:   vec4<f32>,
  color_a: u32,
  size_a:  f32,
  _pad0:   vec2<f32>,
  pos_b:   vec4<f32>,
  color_b: u32,
  size_b:  f32,
  _pad1:   vec2<f32>,
};

struct VoxelBuffer {
  voxels: array<Voxel>,
};

struct InstanceData {
  ndc_min:   vec2<f32>,  // 0
  ndc_max:   vec2<f32>,  // 8
  min_depth: f32,        // 16
  lod_level: u32,        // 20
  _pad:      vec2<f32>,  // 24
};

@group(0) @binding(0) var<uniform> cull: CullUniforms;
@group(0) @binding(1) var<storage, read> voxel_buf: VoxelBuffer;
@group(0) @binding(2) var<storage, read_write> visible_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> instance_data: array<InstanceData>;
@group(0) @binding(4) var<storage, read_write> draw_args: array<atomic<u32>>;

@compute @workgroup_size(256)
fn cull_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_id = gid.x;
  if local_id >= cull.voxel_count {
    return;
  }

  let voxel_index = cull.start_slot + local_id;
  let voxel = voxel_buf.voxels[voxel_index];
  let t = cull.interpolation;

  // Interpolate position and size, apply RTE offset
  let center = mix(voxel.pos_a.xyz, voxel.pos_b.xyz, t) + cull.rte_offset;
  let half_size = mix(voxel.size_a, voxel.size_b, t) * 0.5;
  let half = vec3<f32>(half_size, half_size, half_size);

  // Frustum-sphere culling (conservative: use bounding sphere)
  let radius = half_size * 1.732051; // sqrt(3)
  for (var p = 0u; p < 6u; p = p + 1u) {
    let plane = cull.frustum_planes[p];
    let dist = dot(plane.xyz, center) + plane.w;
    if dist < -radius {
      return; // fully outside this plane
    }
  }

  // Project 8 box corners to find screen-space AABB
  var ndc_min = vec2<f32>(1.0, 1.0);
  var ndc_max = vec2<f32>(-1.0, -1.0);
  var min_depth: f32 = 1.0;
  var front_count = 0u;
  var any_behind = false;

  for (var i = 0u; i < 8u; i = i + 1u) {
    let corner = center + half * vec3<f32>(
      select(-1.0, 1.0, (i & 1u) != 0u),
      select(-1.0, 1.0, (i & 2u) != 0u),
      select(-1.0, 1.0, (i & 4u) != 0u),
    );
    let clip = cull.view_proj * vec4<f32>(corner, 1.0);

    if clip.w > 0.0 {
      let ndc = clip.xy / clip.w;
      ndc_min = min(ndc_min, ndc);
      ndc_max = max(ndc_max, ndc);
      min_depth = min(min_depth, clip.z / clip.w);
      front_count += 1u;
    } else {
      any_behind = true;
    }
  }

  // All corners behind camera → cull
  if front_count == 0u {
    return;
  }

  // Straddles near plane → conservatively expand to full screen
  if any_behind {
    ndc_min = vec2<f32>(-1.0, -1.0);
    ndc_max = vec2<f32>(1.0, 1.0);
    min_depth = 0.0;
  }

  // Subpixel culling — smooth probabilistic falloff
  let screen_size = (ndc_max - ndc_min) * cull.viewport_size * 0.5;
  let screen_area_raw = screen_size.x * screen_size.y;
  // Quantize to 1/16 px² steps to absorb GPU FP non-determinism across frames
  let screen_area = floor(screen_area_raw * 16.0) / 16.0;
  // Smooth keep-probability: 1.0 when screen_area >= threshold, ramps to 0 below
  let keep_prob = clamp(screen_area / cull.subpixel_threshold, 0.0, 1.0);
  let hash = (local_id * 2654435761u) & 0xFFFFu;
  if f32(hash) / 65535.0 >= keep_prob {
    return;
  }

  // Pad billboard edges — only for large close-up voxels where edge clipping is visible
  let min_dim = min(screen_size.x, screen_size.y);
  let pad_scale = saturate((min_dim - 40.0) / 20.0);
  let pad = vec2<f32>(2.0 / cull.viewport_size.x, 2.0 / cull.viewport_size.y) * pad_scale;
  ndc_min = ndc_min - pad;
  ndc_max = ndc_max + pad;

  // Clamp to clip space
  ndc_min = clamp(ndc_min, vec2<f32>(-1.0), vec2<f32>(1.0));
  ndc_max = clamp(ndc_max, vec2<f32>(-1.0), vec2<f32>(1.0));

  // This voxel is visible — append to output
  // Indirect args layout per chunk: [vertex_count, instance_count, first_vertex, first_instance]
  let args_base = cull.chunk_index * 4u;
  let slot = atomicAdd(&draw_args[args_base + 1u], 1u);
  let out_index = cull.start_slot + slot;

  visible_indices[out_index] = voxel_index;

  var data: InstanceData;
  data.ndc_min = ndc_min;
  data.ndc_max = ndc_max;
  data.min_depth = min_depth;
  data.lod_level = cull.lod_level;
  data._pad = vec2<f32>(0.0, 0.0);
  instance_data[out_index] = data;
}
