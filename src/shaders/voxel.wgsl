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
  aa_mode:        u32,           // 168
  jitter_x:       f32,           // 172
  jitter_y:       f32,           // 176
  voxel_scale:    f32,           // 180
  bevel_radius:   f32,           // 184
};

struct ChunkUniforms {
  rte_offset:  vec3<f32>,
  lod_level:   u32,
  start_slot:  u32,
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
  let global_iid = chunk.start_slot + iid;
  let inst = instance_data[global_iid];
  let voxel_index = visible_indices[global_iid];
  let voxel = voxel_buf.voxels[voxel_index];
  let t = uniforms.interpolation;

  // Reconstruct center/half/color for fragment shader ray-AABB
  let center = mix(voxel.pos_a.xyz, voxel.pos_b.xyz, t) + chunk.rte_offset;
  let half_size = mix(voxel.size_a, voxel.size_b, t) * uniforms.voxel_scale * 0.5;
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
  // TAA: offset rasterization position by jitter (shifts which pixel the fragment maps to)
  // Ray reconstruction above uses unjittered ndc_pos + unjittered inv_VP → stable rays
  var pos_ndc = ndc_pos;
  if uniforms.aa_mode == 2u {
    pos_ndc = ndc_pos + vec2<f32>(uniforms.jitter_x, uniforms.jitter_y);
  }
  out.position = vec4<f32>(pos_ndc, min_depth, 1.0);
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

// Rounded box SDF — distance from point p to surface of box with half-extents b, rounded by r
fn sd_round_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
  let q = abs(p) - b + r;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Analytical normal of rounded box SDF via gradient
fn sd_round_box_normal(p: vec3<f32>, b: vec3<f32>, r: f32) -> vec3<f32> {
  let q = abs(p) - b + r;
  // Gradient of the SDF: sign(p) * normalize(max(q, 0)) when outside, sign of dominant axis when inside
  let outside = max(q, vec3<f32>(0.0));
  let len = length(outside);
  if len > 0.0001 {
    return normalize(sign(p) * outside);
  }
  // Inside: normal points along the axis with the largest q component
  let ap = abs(p);
  if q.x > q.y && q.x > q.z {
    return vec3<f32>(sign(p.x), 0.0, 0.0);
  } else if q.y > q.z {
    return vec3<f32>(0.0, sign(p.y), 0.0);
  } else {
    return vec3<f32>(0.0, 0.0, sign(p.z));
  }
}

// Sphere-trace the rounded box SDF within [t_start, t_end], returns hit t or -1.0 on miss
fn trace_round_box(ray_origin: vec3<f32>, ray_dir: vec3<f32>, center: vec3<f32>, half: vec3<f32>, bevel: f32, t_start: f32, t_end: f32) -> f32 {
  var t = t_start;
  let dir = normalize(ray_dir);
  let dir_len = length(ray_dir);
  for (var i = 0; i < 16; i++) {
    let p = ray_origin + ray_dir * t - center;
    let d = sd_round_box(p, half, bevel);
    if d < 0.0005 * (t * dir_len) {
      return t;
    }
    // Advance by SDF distance divided by ray direction length (since ray_dir may not be unit)
    t += d / dir_len;
    if t > t_end {
      return -1.0;
    }
  }
  return -1.0;
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

struct ShadeResult {
  color: vec3<f32>,
  depth: f32,
  hit: bool,
  hit_pos: vec3<f32>,
  normal: vec3<f32>,
};

fn shade_ray(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>, base_color: vec3<f32>) -> ShadeResult {
  var result: ShadeResult;
  result.hit = false;
  result.color = vec3<f32>(0.0);
  result.depth = 1.0;
  result.hit_pos = vec3<f32>(0.0);
  result.normal = vec3<f32>(0.0, 1.0, 0.0);

  let t = ray_aabb(ray_origin, ray_dir, box_min, box_max);
  let tmin = t.x;
  let tmax = t.y;

  if tmin > tmax || tmax < 0.0 {
    return result;
  }

  let center = (box_min + box_max) * 0.5;
  let half = (box_max - box_min) * 0.5;
  let bevel = uniforms.bevel_radius * half.x; // fraction of half-size

  if bevel > 0.0 {
    // SDF sphere-tracing path for rounded cubes
    let t_start = max(tmin, 0.0);
    let t_sdf = trace_round_box(ray_origin, ray_dir, center, half, bevel, t_start, tmax);
    if t_sdf < 0.0 {
      return result;
    }
    result.hit = true;
    result.hit_pos = ray_origin + ray_dir * t_sdf;
    let local_p = result.hit_pos - center;
    result.normal = sd_round_box_normal(local_p, half, bevel);
  } else {
    // Original fast AABB slab path
    var t_hit = tmin;
    if t_hit < 0.0 {
      t_hit = tmax;
    }

    result.hit = true;
    result.hit_pos = ray_origin + ray_dir * t_hit;

    // Determine face normal from slab intersection axis
    let inv_dir = 1.0 / ray_dir;
    let t1 = (box_min - ray_origin) * inv_dir;
    let t2 = (box_max - ray_origin) * inv_dir;
    let tmin_v = min(t1, t2);
    let tmax_v = max(t1, t2);

    if t_hit == tmin {
      if tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z {
        result.normal = vec3<f32>(sign(-ray_dir.x), 0.0, 0.0);
      } else if tmin_v.y >= tmin_v.z {
        result.normal = vec3<f32>(0.0, sign(-ray_dir.y), 0.0);
      } else {
        result.normal = vec3<f32>(0.0, 0.0, sign(-ray_dir.z));
      }
    } else {
      if tmax_v.x <= tmax_v.y && tmax_v.x <= tmax_v.z {
        result.normal = vec3<f32>(sign(ray_dir.x), 0.0, 0.0);
      } else if tmax_v.y <= tmax_v.z {
        result.normal = vec3<f32>(0.0, sign(ray_dir.y), 0.0);
      } else {
        result.normal = vec3<f32>(0.0, 0.0, sign(ray_dir.z));
      }
    }
  }

  // Lighting
  let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
  let ndotl = max(dot(result.normal, light_dir), 0.0);
  let ambient = 0.15;
  let diffuse = 0.85 * ndotl;
  result.color = base_color * (ambient + diffuse);

  // Depth
  let hit_clip = uniforms.view_proj * vec4<f32>(result.hit_pos, 1.0);
  result.depth = hit_clip.z / hit_clip.w;

  return result;
}

struct FragOutput {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
  let box_min = in.box_center - in.box_half;
  let box_max = in.box_center + in.box_half;

  // Compute ray direction derivatives for supersampling (must be in uniform control flow)
  let ddx_dir = dpdx(in.ray_dir);
  let ddy_dir = dpdy(in.ray_dir);

  // Center ray
  let center = shade_ray(in.ray_origin, in.ray_dir, box_min, box_max, in.color);
  if !center.hit {
    discard;
  }

  var lit_color = center.color;

  // Hit position relative to box (used by edge SS and wireframe debug)
  let abs_rel = abs((center.hit_pos - in.box_center) / in.box_half);

  // Determine if supersampling is needed
  var needs_supersample = false;

  // Distance-adaptive SS (mode 1): supersample small distant voxels
  if uniforms.aa_mode == 1u {
    let screen_px = 1.0 / fwidth(in.quad_uv);
    if min(screen_px.x, screen_px.y) < 4.0 {
      needs_supersample = true;
    }
  }

  // Edge detection for modes 0 and 1: check if hit is near a cube edge
  if uniforms.aa_mode <= 1u {
    let mx = max(abs_rel.x, max(abs_rel.y, abs_rel.z));
    let mn = min(abs_rel.x, min(abs_rel.y, abs_rel.z));
    let mid = abs_rel.x + abs_rel.y + abs_rel.z - mx - mn;
    if mid > 0.92 {
      needs_supersample = true;
    }
  }

  if needs_supersample {
    // RGSS 4x sub-pixel rays
    let d0 = in.ray_dir + ddx_dir * -0.125 + ddy_dir * -0.375;
    let d1 = in.ray_dir + ddx_dir *  0.375 + ddy_dir * -0.125;
    let d2 = in.ray_dir + ddx_dir * -0.375 + ddy_dir *  0.125;
    let d3 = in.ray_dir + ddx_dir *  0.125 + ddy_dir *  0.375;

    let s0 = shade_ray(in.ray_origin, d0, box_min, box_max, in.color);
    let s1 = shade_ray(in.ray_origin, d1, box_min, box_max, in.color);
    let s2 = shade_ray(in.ray_origin, d2, box_min, box_max, in.color);
    let s3 = shade_ray(in.ray_origin, d3, box_min, box_max, in.color);

    var sum = vec3<f32>(0.0);
    var count = 0.0;
    if s0.hit { sum += s0.color; count += 1.0; }
    if s1.hit { sum += s1.color; count += 1.0; }
    if s2.hit { sum += s2.color; count += 1.0; }
    if s3.hit { sum += s3.color; count += 1.0; }
    if count > 0.0 {
      lit_color = sum / count;
    }
  }

  // LOD debug visualization: blend with LOD palette color when bit 0 set
  if (uniforms.debug_flags & 1u) != 0u {
    let lod_col = lod_color(in.lod_level);
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(center.normal, light_dir), 0.0);
    lit_color = mix(lit_color, lod_col * (0.15 + 0.85 * ndotl), 0.6);
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
    lit_color = center.normal * 0.5 + 0.5;
  }

  // Depth debug: log-scale depth → grayscale (bit 4)
  if (uniforms.debug_flags & 16u) != 0u {
    let linear_z = uniforms.near * uniforms.far /
      (uniforms.far - center.depth * (uniforms.far - uniforms.near));
    let norm_depth = log2(linear_z / uniforms.near) / log2(uniforms.far / uniforms.near);
    lit_color = vec3<f32>(saturate(norm_depth));
  }

  var out: FragOutput;
  out.color = vec4<f32>(lit_color, 1.0);
  out.depth = center.depth;
  return out;
}
