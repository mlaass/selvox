/** Minimal column-major mat4 utilities (zero dependencies). */

export type Mat4 = Float32Array;

export function mat4Create(): Mat4 {
  const out = new Float32Array(16);
  out[0] = 1; out[5] = 1; out[10] = 1; out[15] = 1;
  return out;
}

export function mat4Multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[row]      * b[col * 4]     +
        a[4 + row]  * b[col * 4 + 1] +
        a[8 + row]  * b[col * 4 + 2] +
        a[12 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

export function mat4Invert(out: Mat4, m: Mat4): Mat4 {
  const [
    m00, m01, m02, m03,
    m10, m11, m12, m13,
    m20, m21, m22, m23,
    m30, m31, m32, m33,
  ] = m;

  const b00 = m00 * m11 - m01 * m10;
  const b01 = m00 * m12 - m02 * m10;
  const b02 = m00 * m13 - m03 * m10;
  const b03 = m01 * m12 - m02 * m11;
  const b04 = m01 * m13 - m03 * m11;
  const b05 = m02 * m13 - m03 * m12;
  const b06 = m20 * m31 - m21 * m30;
  const b07 = m20 * m32 - m22 * m30;
  const b08 = m20 * m33 - m23 * m30;
  const b09 = m21 * m32 - m22 * m31;
  const b10 = m21 * m33 - m23 * m31;
  const b11 = m22 * m33 - m23 * m32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-10) {
    out.fill(0);
    return out;
  }
  det = 1.0 / det;

  out[0]  = ( m11 * b11 - m12 * b10 + m13 * b09) * det;
  out[1]  = (-m01 * b11 + m02 * b10 - m03 * b09) * det;
  out[2]  = ( m31 * b05 - m32 * b04 + m33 * b03) * det;
  out[3]  = (-m21 * b05 + m22 * b04 - m23 * b03) * det;
  out[4]  = (-m10 * b11 + m12 * b08 - m13 * b07) * det;
  out[5]  = ( m00 * b11 - m02 * b08 + m03 * b07) * det;
  out[6]  = (-m30 * b05 + m32 * b02 - m33 * b01) * det;
  out[7]  = ( m20 * b05 - m22 * b02 + m23 * b01) * det;
  out[8]  = ( m10 * b10 - m11 * b08 + m13 * b06) * det;
  out[9]  = (-m00 * b10 + m01 * b08 - m03 * b06) * det;
  out[10] = ( m30 * b04 - m31 * b02 + m33 * b00) * det;
  out[11] = (-m20 * b04 + m21 * b02 - m23 * b00) * det;
  out[12] = (-m10 * b09 + m11 * b07 - m12 * b06) * det;
  out[13] = ( m00 * b09 - m01 * b07 + m02 * b06) * det;
  out[14] = (-m30 * b03 + m31 * b01 - m32 * b00) * det;
  out[15] = ( m20 * b03 - m21 * b01 + m22 * b00) * det;

  return out;
}

export function mat4Perspective(out: Mat4, fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovY * 0.5);
  const rangeInv = 1.0 / (near - far);

  out.fill(0);
  out[0]  = f / aspect;
  out[5]  = f;
  out[10] = far * rangeInv;
  out[11] = -1;
  out[14] = near * far * rangeInv;

  return out;
}

export function mat4LookAt(out: Mat4, eye: Float32Array, center: Float32Array, up: Float32Array): Mat4 {
  let fx = center[0] - eye[0];
  let fy = center[1] - eye[1];
  let fz = center[2] - eye[2];

  let len = Math.sqrt(fx * fx + fy * fy + fz * fz);
  fx /= len; fy /= len; fz /= len;

  let sx = fy * up[2] - fz * up[1];
  let sy = fz * up[0] - fx * up[2];
  let sz = fx * up[1] - fy * up[0];
  len = Math.sqrt(sx * sx + sy * sy + sz * sz);
  sx /= len; sy /= len; sz /= len;

  const ux = sy * fz - sz * fy;
  const uy = sz * fx - sx * fz;
  const uz = sx * fy - sy * fx;

  out[0]  = sx;  out[1]  = ux;  out[2]  = -fx; out[3]  = 0;
  out[4]  = sy;  out[5]  = uy;  out[6]  = -fy; out[7]  = 0;
  out[8]  = sz;  out[9]  = uz;  out[10] = -fz; out[11] = 0;
  out[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
  out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  out[14] = -(-fx * eye[0] + -fy * eye[1] + -fz * eye[2]);
  out[15] = 1;

  return out;
}
