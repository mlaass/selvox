// Phase 1–3 — Interpolated voxel cubes with RTE coordinates and LOD debug

struct Uniforms {
  view_proj:      mat4x4<f32>,   // 0
  inv_view_proj:  mat4x4<f32>,   // 64
  camera_pos:     vec3<f32>,     // 128
  interpolation:  f32,           // 140
  viewport_size:  vec2<f32>,     // 144
  near:           f32,           // 152
  far:            f32,           // 156
  color_t:        f32,           // 160
  debug_flags:    u32,           // 164
};

struct ChunkUniforms {
  rte_offset: vec3<f32>,
  lod_level:  u32,
};

struct Voxel {
  pos_a:   vec4<f32>,  // 0   (xyz used)
  color_a: u32,        // 16
  size_a:  f32,        // 20
  _pad0:   vec2<f32>,  // 24
  pos_b:   vec4<f32>,  // 32
  color_b: u32,        // 48
  size_b:  f32,        // 52
  _pad1:   vec2<f32>,  // 56
};

struct VoxelBuffer {
  voxels: array<Voxel>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> voxel_buf: VoxelBuffer;
@group(1) @binding(0) var<uniform> chunk: ChunkUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) ray_origin: vec3<f32>,
  @location(1) ray_dir: vec3<f32>,
  @location(2) box_center: vec3<f32>,
  @location(3) box_half: vec3<f32>,
  @location(4) color: vec3<f32>,
  @location(5) @interpolate(flat) lod_level: u32,
};

fn unpack_color(c: u32) -> vec3<f32> {
  return vec3<f32>(
    f32((c >> 0u) & 0xFFu) / 255.0,
    f32((c >> 8u) & 0xFFu) / 255.0,
    f32((c >> 16u) & 0xFFu) / 255.0,
  );
}

@vertex
fn vs_main(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
) -> VertexOutput {
  let voxel = voxel_buf.voxels[iid];
  let t = uniforms.interpolation;

  // Interpolate position, size, color — apply RTE offset
  let center = mix(voxel.pos_a.xyz, voxel.pos_b.xyz, t) + chunk.rte_offset;
  let half_size = mix(voxel.size_a, voxel.size_b, t) * 0.5;
  let color_a = unpack_color(voxel.color_a);
  let color_b = unpack_color(voxel.color_b);
  let color = mix(color_a, color_b, uniforms.color_t);

  let half = vec3<f32>(half_size, half_size, half_size);

  // Project 8 box corners to find screen-space AABB
  var ndc_min = vec2<f32>(1e9, 1e9);
  var ndc_max = vec2<f32>(-1e9, -1e9);

  for (var i = 0u; i < 8u; i = i + 1u) {
    let corner = center + half * vec3<f32>(
      select(-1.0, 1.0, (i & 1u) != 0u),
      select(-1.0, 1.0, (i & 2u) != 0u),
      select(-1.0, 1.0, (i & 4u) != 0u),
    );
    let clip = uniforms.view_proj * vec4<f32>(corner, 1.0);
    let ndc = clip.xy / clip.w;
    ndc_min = min(ndc_min, ndc);
    ndc_max = max(ndc_max, ndc);
  }

  // Pad slightly to avoid clipping at edges
  let pad = vec2<f32>(2.0 / uniforms.viewport_size.x, 2.0 / uniforms.viewport_size.y);
  ndc_min = ndc_min - pad;
  ndc_max = ndc_max + pad;

  // Clamp to clip space
  ndc_min = clamp(ndc_min, vec2<f32>(-1.0), vec2<f32>(1.0));
  ndc_max = clamp(ndc_max, vec2<f32>(-1.0), vec2<f32>(1.0));

  // 6 vertices → 2 triangles forming the quad
  // Triangle 1: 0,1,2  Triangle 2: 2,1,3
  var quad_uv: vec2<f32>;
  switch vid % 6u {
    case 0u: { quad_uv = vec2<f32>(0.0, 0.0); }
    case 1u: { quad_uv = vec2<f32>(1.0, 0.0); }
    case 2u: { quad_uv = vec2<f32>(0.0, 1.0); }
    case 3u: { quad_uv = vec2<f32>(0.0, 1.0); }
    case 4u: { quad_uv = vec2<f32>(1.0, 0.0); }
    case 5u: { quad_uv = vec2<f32>(1.0, 1.0); }
    default: { quad_uv = vec2<f32>(0.0, 0.0); }
  }

  let ndc_pos = mix(ndc_min, ndc_max, quad_uv);

  // Reconstruct world-space ray from NDC position using inv_view_proj
  let ndc4_near = vec4<f32>(ndc_pos, 0.0, 1.0);
  let ndc4_far  = vec4<f32>(ndc_pos, 1.0, 1.0);
  let world_near = uniforms.inv_view_proj * ndc4_near;
  let world_far  = uniforms.inv_view_proj * ndc4_far;
  let p_near = world_near.xyz / world_near.w;
  let p_far  = world_far.xyz / world_far.w;
  let ray_dir = normalize(p_far - p_near);

  var out: VertexOutput;
  out.position = vec4<f32>(ndc_pos, 0.0, 1.0);
  out.ray_origin = uniforms.camera_pos;
  out.ray_dir = ray_dir;
  out.box_center = center;
  out.box_half = half;
  out.color = color;
  out.lod_level = chunk.lod_level;
  return out;
}

// Majercik 2018 ray-AABB intersection
fn ray_aabb(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
  let inv_dir = 1.0 / ray_dir;
  let t1 = (box_min - ray_origin) * inv_dir;
  let t2 = (box_max - ray_origin) * inv_dir;
  let tmin_v = min(t1, t2);
  let tmax_v = max(t1, t2);
  let tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
  let tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
  return vec2<f32>(tmin, tmax);
}

// LOD debug color palette (7 levels)
fn lod_color(level: u32) -> vec3<f32> {
  switch level {
    case 0u: { return vec3<f32>(0.0, 1.0, 0.0); }   // green — finest
    case 1u: { return vec3<f32>(0.5, 1.0, 0.0); }   // yellow-green
    case 2u: { return vec3<f32>(1.0, 1.0, 0.0); }   // yellow
    case 3u: { return vec3<f32>(1.0, 0.5, 0.0); }   // orange
    case 4u: { return vec3<f32>(1.0, 0.0, 0.0); }   // red
    case 5u: { return vec3<f32>(1.0, 0.0, 0.5); }   // magenta
    case 6u: { return vec3<f32>(1.0, 0.0, 1.0); }   // purple — coarsest
    default: { return vec3<f32>(1.0, 1.0, 1.0); }
  }
}

struct FragOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
  let box_min = in.box_center - in.box_half;
  let box_max = in.box_center + in.box_half;

  let t = ray_aabb(in.ray_origin, in.ray_dir, box_min, box_max);
  let tmin = t.x;
  let tmax = t.y;

  if tmin > tmax || tmax < 0.0 {
    discard;
  }

  // Use the closer positive hit
  var t_hit = tmin;
  if t_hit < 0.0 {
    t_hit = tmax;
  }

  let hit_pos = in.ray_origin + in.ray_dir * t_hit;

  // Compute face normal from the hit position relative to box center
  let rel = (hit_pos - in.box_center) / in.box_half;
  let abs_rel = abs(rel);
  var normal: vec3<f32>;
  if abs_rel.x > abs_rel.y && abs_rel.x > abs_rel.z {
    normal = vec3<f32>(sign(rel.x), 0.0, 0.0);
  } else if abs_rel.y > abs_rel.z {
    normal = vec3<f32>(0.0, sign(rel.y), 0.0);
  } else {
    normal = vec3<f32>(0.0, 0.0, sign(rel.z));
  }

  // Simple directional lighting
  let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
  let ndotl = max(dot(normal, light_dir), 0.0);
  let ambient = 0.15;
  let diffuse = 0.85 * ndotl;
  var lit_color = in.color * (ambient + diffuse);

  // LOD debug visualization: blend with LOD palette color when bit 0 set
  if (uniforms.debug_flags & 1u) != 0u {
    let lod_col = lod_color(in.lod_level);
    lit_color = mix(lit_color, lod_col * (ambient + diffuse), 0.6);
  }

  // Project hit point to get depth
  let clip = uniforms.view_proj * vec4<f32>(hit_pos, 1.0);
  let depth = clip.z / clip.w;

  var out: FragOutput;
  out.color = vec4<f32>(lit_color, 1.0);
  out.depth = depth;
  return out;
}
