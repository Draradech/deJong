struct Uni {
  a: f32, b: f32, c: f32, d: f32,
  frame: f32,
  txsz: f32,
  npoints: f32
};

@group(0) @binding(0) var<uniform> uni: Uni;
@group(0) @binding(1) var<storage, read_write> a_red: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> a_green: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> a_blue: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> u_red: array<u32>;
@group(0) @binding(5) var<storage, read_write> u_green: array<u32>;
@group(0) @binding(6) var<storage, read_write> u_blue: array<u32>;

fn pcg3d(vin: vec3u) -> vec3u
{
  var v = vin * 1664525u + 1013904223u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  return v;
}

fn pcg3df(vin: vec3u) -> vec3f
{
  return vec3f(pcg3d(vin)) / f32(0xffffffffu);
}

@compute @workgroup_size(8, 8)
fn deJongAttractor(@builtin(global_invocation_id) id: vec3u)
{
  // random starting point per thread
  let rnd = pcg3df(vec3u(id.xy, u32(uni.frame)));
  var x1 = 2.0 * sin(6.28 * rnd.x);
  var y1 = 2.0 * sin(6.28 * rnd.y);

  // prerun to converge onto attractor from random starting point
  for(var i = 0u; i < 32u; i++)
  {
    let x2 = sin(uni.a * y1) - cos(uni.b * x1);
    let y2 = sin(uni.c * x1) - cos(uni.d * y1);
    x1 = x2;
    y1 = y2;
  }

  // 1024 iterations per thread
  for(var i = 0u; i < 1024u; i++)
  {
    let x2 = sin(uni.a * y1) - cos(uni.b * x1);
    let y2 = sin(uni.c * x1) - cos(uni.d * y1);
    let x = u32(x2 * 0.25 * uni.txsz * 0.96 + uni.txsz * 0.5);
    let y = u32(y2 * 0.25 * uni.txsz * 0.96 + uni.txsz * 0.5);
    let idx = y * u32(uni.txsz) + x;
    let dx = x2 - x1;
    let dy = y2 - y1;
    atomicAdd(&a_red[idx], u32(256. * abs(dx)));
    atomicAdd(&a_green[idx], u32(256. * abs(dy)));
    atomicAdd(&a_blue[idx], 256u);
    x1 = x2;
    y1 = y2;
  }
}

@vertex fn vs(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f
{
  let pos = array(
    vec2f(-1.0,  1.0), // top left
    vec2f( 1.0,  1.0), // top right
    vec2f(-1.0, -1.0), // bottom left
    vec2f( 1.0, -1.0) // bottom right
  );
  return vec4f(pos[vertexIndex], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) pos : vec4f) -> @location(0) vec4f
{
  let px = vec2u(pos.xy);
  let idx = px.x + (u32(uni.txsz) - px.y - 1) * u32(uni.txsz);
  let col = vec3f(f32(u_red[idx]), f32(u_green[idx]), f32(u_blue[idx]));
  return vec4f(col * uni.txsz  * uni.txsz / (2048 * uni.npoints), 1);
}
