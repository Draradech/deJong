struct uniforms {
  a:      f32,
  b:      f32,
  c:      f32,
  d:      f32,
  frame:  f32,
  txsz:   f32,
  lpcnt:  f32,
  bright: f32,
  budget: f32
};

struct timestamp {
  start:      u32,
  start_high: u32,
  end:        u32,
  end_high:   u32
};

struct frameinfo {
  p1s: u32,
  p1e: u32,
  p3s: u32,
  p3e: u32,
  p5s: u32,
  p5e: u32,
  rps: u32,
  rpe: u32,
  p1p: u32,
  p3p: u32,
  p5p: u32,
  p:   u32,
  pas: u32
}

struct indirect {
  x: u32,
  y: u32,
  z: u32
}

@group(0) @binding(0) var<uniform>               uni:    uniforms;
@group(0) @binding(1) var<storage>               time:   timestamp;
@group(0) @binding(2) var<storage, read_write>   info:   frameinfo;
@group(0) @binding(3) var<storage, read_write>   exi:    indirect;
@group(0) @binding(4) var<storage, read_write>   texa:   array<array<atomic<u32>, 3>>;
@group(0) @binding(5) var<storage, read_write>   tex:    array<array<u32, 3>>;

const wgsz = 16u;

fn pcg3d(vin: vec3u) -> vec3u
{
  var v = vin * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}

fn pcg3df(vin: vec3u) -> vec3f
{
  return vec3f(pcg3d(vin)) / f32(0xffffffffu);
}

@compute @workgroup_size(1)
fn pass2()
{
  info.p1p = 16u * wgsz * wgsz * u32(uni.lpcnt);
  info.p1s = time.start;
  info.p1e = time.end;
  let t1 = f32(i32(info.p1e) - i32(info.p1s)) / 1000000.;
  let ratio_p1 = t1 / uni.budget;
  let ratio_p3 = .5 - ratio_p1;
  var p3 = f32(info.p1p) / ratio_p1 * ratio_p3;
  p3 = max(p3, 0.);
  var iv3 = u32(p3 / f32(wgsz) / f32(wgsz) / uni.lpcnt);
  iv3 = max(iv3, 0u);
  info.p3p = iv3 * wgsz * wgsz * u32(uni.lpcnt);
  exi.x = iv3;
  exi.y = 1u;
  exi.z = 1u;
  info.pas = 3u;
}

@compute @workgroup_size(1)
fn pass4()
{
  info.p3s = time.start;
  info.p3e = time.end;
  let t13 = f32(i32(info.p3e) - i32(info.p1s)) / 1000000.;
  let ratio_p13 = t13 / uni.budget;
  let ratio_p5 = 1. - ratio_p13;
  var p5 = f32(info.p1p + info.p3p) / ratio_p13 * ratio_p5;
  p5 = max(p5, 0.);
  var iv5 = u32(p5 / f32(wgsz) / f32(wgsz) / uni.lpcnt);
  iv5 = max(iv5, 0u);
  info.p5p = iv5 * wgsz * wgsz * u32(uni.lpcnt);
  info.p = info.p1p + info.p3p + info.p5p;
  exi.x = iv5;
  exi.y = 1u;
  exi.z = 1u;
  info.pas = 5u;
}

@compute @workgroup_size(1)
fn pass6()
{
  info.p5s = time.start;
  info.p5e = time.end;
}

@compute @workgroup_size(1)
fn pass7()
{
  info.rps = time.start;
  info.rpe = time.end;
  info.pas = 1u;
}

@compute @workgroup_size(wgsz, wgsz)
fn deJong(@builtin(global_invocation_id) id: vec3u)
{
  // random starting point per thread
  let rnd = pcg3df(vec3u(id.xy, u32(uni.frame) + info.pas));
  var x1 = 2. * sin(6.28 * rnd.x);
  var y1 = 2. * sin(6.28 * rnd.y);

  // prerun to converge onto attractor from random starting point
  for(var i = 0u; i < 16; i++)
  {
    let x2 = sin(uni.a * y1) - cos(uni.b * x1);
    let y2 = sin(uni.c * x1) - cos(uni.d * y1);
    x1 = x2;
    y1 = y2;
  }

  // uni.lpcnt iterations per thread
  for(var i = 0u; i < u32(uni.lpcnt); i++)
  {
    let x2 = sin(uni.a * y1) - cos(uni.b * x1);
    let y2 = sin(uni.c * x1) - cos(uni.d * y1);
    let x = u32(x2 * 0.25 * uni.txsz * 0.96 + uni.txsz * 0.5);
    let y = u32(y2 * 0.25 * uni.txsz * 0.96 + uni.txsz * 0.5);
    let idx = (y * u32(uni.txsz) + x);
    let dx = x2 - x1;
    let dy = y2 - y1;
    atomicAdd(&texa[idx][0], u32(256. * abs(dx)));
    atomicAdd(&texa[idx][1], u32(256. * abs(dy)));
    atomicAdd(&texa[idx][2], 256u);
    x1 = x2;
    y1 = y2;
  }
}

@vertex
fn vs(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f
{
  let v1 = 4. * f32(vertexIndex % 2);
  let v2 = 4. * f32(vertexIndex / 2);
  return vec4f(vec2f(-1. + v1, -1. + v2), 0., 1.);
}

@fragment
fn fs(@builtin(position) pos : vec4f) -> @location(0) vec4f
{
  let px = vec2u(pos.xy);
  let idx = (px.x + (u32(uni.txsz) - px.y - 1) * u32(uni.txsz));
  let col = vec3f(f32(tex[idx][0]), f32(tex[idx][1]), f32(tex[idx][2]));
  return vec4f(col * uni.txsz  * uni.txsz * uni.bright / f32(info.p), 1);
}
