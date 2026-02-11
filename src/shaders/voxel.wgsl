// Render shader — reads precomputed instance data from compute culling pass

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

struct InstanceData {
  ndc_min:   vec2<f32>,  // 0
  ndc_max:   vec2<f32>,  // 8
  min_depth: f32,        // 16
  lod_level: u32,        // 20
  _pad:      vec2<f32>,  // 24
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> voxel_buf: VoxelBuffer;
@group(0) @binding(2) var<storage, read> visible_indices: array<u32>;
@group(0) @binding(3) var<storage, read> instance_data: array<InstanceData>;
@group(1) @binding(0) var<uniform> chunk: ChunkUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) ray_origin: vec3<f32>,
  @location(1) ray_dir: vec3<f32>,
  @location(2) box_center: vec3<f32>,
  @location(3) box_half: vec3<f32>,
  @location(4) color: vec3<f32>,
  @location(5) @interpolate(flat) lod_level: u32,
  @location(6) quad_uv: vec2<f32>,
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
  // Read precomputed instance data from compute cull pass
  let inst = instance_data[iid];
  let voxel_index = visible_indices[iid];
  let voxel = voxel_buf.voxels[voxel_index];
  let t = uniforms.interpolation;

  // Reconstruct center/half/color for fragment shader ray-AABB
  let center = mix(voxel.pos_a.xyz, voxel.pos_b.xyz, t) + chunk.rte_offset;
  let half_size = mix(voxel.size_a, voxel.size_b, t) * 0.5;
  let color_a = unpack_color(voxel.color_a);
  let color_b = unpack_color(voxel.color_b);
  let color = mix(color_a, color_b, uniforms.color_t);
  let half = vec3<f32>(half_size, half_size, half_size);

  // Read precomputed screen-space AABB
  let ndc_min = inst.ndc_min;
  let ndc_max = inst.ndc_max;
  let min_depth = inst.min_depth;

  // 6 vertices → 2 triangles forming the quad
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
  let ray_dir = p_far - p_near;

  var out: VertexOutput;
  out.position = vec4<f32>(ndc_pos, min_depth, 1.0);
  out.ray_origin = uniforms.camera_pos;
  out.ray_dir = ray_dir;
  out.box_center = center;
  out.box_half = half;
  out.color = color;
  out.lod_level = inst.lod_level;
  out.quad_uv = quad_uv;
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
  let hit_clip = uniforms.view_proj * vec4<f32>(hit_pos, 1.0);
  let frag_depth = hit_clip.z / hit_clip.w;

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

  // Billboard quad edge debug: yellow outline (bit 1)
  if (uniforms.debug_flags & 2u) != 0u {
    let uv_rate = fwidth(in.quad_uv);
    let edge_px = 1.5;
    let edge_mask = step(in.quad_uv, uv_rate * edge_px) + step(1.0 - in.quad_uv, uv_rate * edge_px);
    if max(edge_mask.x, edge_mask.y) > 0.0 {
      lit_color = vec3<f32>(1.0, 1.0, 0.0);
    }
  }

  // AABB wireframe debug: cyan edges on box faces (bit 2)
  if (uniforms.debug_flags & 4u) != 0u {
    let edge_threshold = 0.03;
    let near_edge = vec3<f32>(
      step(1.0 - edge_threshold, abs_rel.x),
      step(1.0 - edge_threshold, abs_rel.y),
      step(1.0 - edge_threshold, abs_rel.z),
    );
    let edge_count = near_edge.x + near_edge.y + near_edge.z;
    if edge_count >= 2.0 {
      lit_color = vec3<f32>(0.0, 1.0, 1.0);
    }
  }

  // Normals debug: ±XYZ → RGB (bit 3)
  if (uniforms.debug_flags & 8u) != 0u {
    lit_color = normal * 0.5 + 0.5;
  }

  // Depth debug: log-scale depth → grayscale (bit 4)
  if (uniforms.debug_flags & 16u) != 0u {
    let linear_z = uniforms.near * uniforms.far /
      (uniforms.far - frag_depth * (uniforms.far - uniforms.near));
    let norm_depth = log2(linear_z / uniforms.near) / log2(uniforms.far / uniforms.near);
    lit_color = vec3<f32>(saturate(norm_depth));
  }

  var out: FragOutput;
  out.color = vec4<f32>(lit_color, 1.0);
  out.depth = frag_depth;
  return out;
}
