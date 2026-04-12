struct uniform_t {
  a: f32,
  b: f32,
  c: f32,
  d: f32,
  frame: f32,
  texture_size: f32,
  brightness: f32,
  gamma: f32,
  budget: f32,
  timestamp_res: f32,
  screen_width: f32,
  screen_height: f32,
  debug_overlay: f32,
};

struct timestamp_t {
  start: u32,
  start_high: u32,
  end: u32,
  end_high: u32,
};

struct frame_info_t {
  pass_1_start: u32,
  pass_1_end: u32,
  pass_2_start: u32,
  pass_2_end: u32,
  pass_3_start: u32,
  pass_3_end: u32,
  render_start: u32,
  render_end: u32,
  pass_1_points: u32,
  pass_2_points: u32,
  pass_3_points: u32,
  total_points: u32,
  current_pass: u32,
};

@group(0) @binding(0) var<uniform> uni: uniform_t;
@group(0) @binding(1) var<storage> timestamp: timestamp_t;
@group(0) @binding(2) var<storage, read_write> frame_info: frame_info_t;
@group(0) @binding(3) var<storage, read_write> dispatch: vec3u;
@group(0) @binding(4) var<storage, read_write> counts: array<array<atomic<u32>, 3>>;
@group(0) @binding(5) var<storage> counts_ro: array<array<u32, 3>>;
@group(0) @binding(6) var<storage> frame_info_ro: frame_info_t;
@group(0) @binding(7) var<storage> font: array<u32>;
@group(0) @binding(8) var<storage> text: array<u32>;

const workgroup_size = 16u;
const loop_count = 512u;
const font_first = 32u;
const font_width = 8u;
const font_height = 8u;

fn pcg3d(vin: vec3u) -> vec3u {
  var v = vin * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn pcg3df(vin: vec3u) -> vec3f {
  return vec3f(pcg3d(vin)) / f32(0xffffffffu);
}

@compute @workgroup_size(1)
fn pass_1_timing() {
  frame_info.pass_1_points = 16u * workgroup_size * workgroup_size * loop_count;
  frame_info.pass_1_start = timestamp.start;
  frame_info.pass_1_end = timestamp.end;
  var pass_1_ms = f32(i32(frame_info.pass_1_end) - i32(frame_info.pass_1_start)) * uni.timestamp_res / 1000000.0;
  pass_1_ms = max(pass_1_ms, 0.01);
  let pass_1_ratio = pass_1_ms / uni.budget;
  let pass_2_ratio = 0.5 - pass_1_ratio;
  var pass_2_points = f32(frame_info.pass_1_points) / pass_1_ratio * pass_2_ratio;
  pass_2_points = max(pass_2_points, 0.0);
  var pass_2_invocations = u32(pass_2_points / f32(workgroup_size) / f32(workgroup_size) / f32(loop_count));
  pass_2_invocations = max(pass_2_invocations, 1u);
  pass_2_invocations = min(pass_2_invocations, 0xffffu);
  frame_info.pass_2_points = pass_2_invocations * workgroup_size * workgroup_size * loop_count;
  dispatch.x = pass_2_invocations;
  dispatch.y = 1u;
  dispatch.z = 1u;
  frame_info.current_pass = 2u;
}

@compute @workgroup_size(1)
fn pass_2_timing() {
  frame_info.pass_2_start = timestamp.start;
  frame_info.pass_2_end = timestamp.end;
  var pass_12_ms = f32(i32(frame_info.pass_2_end) - i32(frame_info.pass_1_start)) * uni.timestamp_res / 1000000.0;
  pass_12_ms = max(pass_12_ms, 0.01);
  let pass_12_ratio = pass_12_ms / uni.budget;
  let pass_3_ratio = 1.0 - pass_12_ratio;
  var pass_3_points = f32(frame_info.pass_1_points + frame_info.pass_2_points) / pass_12_ratio * pass_3_ratio;
  pass_3_points = max(pass_3_points, 0.0);
  var pass_3_invocations = u32(pass_3_points / f32(workgroup_size) / f32(workgroup_size) / f32(loop_count));
  pass_3_invocations = max(pass_3_invocations, 1u);
  pass_3_invocations = min(pass_3_invocations, 0xffffu);
  frame_info.pass_3_points = pass_3_invocations * workgroup_size * workgroup_size * loop_count;
  frame_info.total_points = frame_info.pass_1_points + frame_info.pass_2_points + frame_info.pass_3_points;
  dispatch.x = pass_3_invocations;
  dispatch.y = 1u;
  dispatch.z = 1u;
  frame_info.current_pass = 3u;
}

@compute @workgroup_size(1)
fn pass_3_timing() {
  frame_info.pass_3_start = timestamp.start;
  frame_info.pass_3_end = timestamp.end;
}

@compute @workgroup_size(1)
fn render_timing() {
  frame_info.render_start = timestamp.start;
  frame_info.render_end = timestamp.end;
  frame_info.current_pass = 1u;
}

@compute @workgroup_size(workgroup_size, workgroup_size)
fn dejong(@builtin(global_invocation_id) id: vec3u) {
  let random = pcg3df(vec3u(id.xy, u32(uni.frame) + frame_info.current_pass));
  var p1 = 2.0 * sin(6.28 * random.xy);

  for (var i = 0u; i < 16u; i++) {
    let p2 = vec2f(
      sin(uni.a * p1.y) - cos(uni.b * p1.x),
      sin(uni.c * p1.x) - cos(uni.d * p1.y),
    );
    p1 = p2;
  }

  for (var i = 0u; i < loop_count; i++) {
    let p2 = vec2f(
      sin(uni.a * p1.y) - cos(uni.b * p1.x),
      sin(uni.c * p1.x) - cos(uni.d * p1.y),
    );
    let texel = vec2u(p2 * 0.25 * uni.texture_size * 0.96 + uni.texture_size * 0.5);
    let index = texel.y * u32(uni.texture_size) + texel.x;
    let delta = p2 - p1;
    atomicAdd(&counts[index][0], u32(256.0 * abs(delta.x)));
    atomicAdd(&counts[index][1], u32(256.0 * abs(delta.y)));
    atomicAdd(&counts[index][2], 256u);
    p1 = p2;
  }
}

@vertex
fn dejong_vs(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {
  let v1 = 4.0 * f32(vertex % 2u);
  let v2 = 4.0 * f32(vertex / 2u);
  return vec4f(vec2f(-1.0 + v1, -1.0 + v2), 0.0, 1.0);
}

@fragment
fn dejong_fs(@builtin(position) pos : vec4f) -> @location(0) vec4f
{
  var col = sample_dejong(pos.xy);
  col = mix(col, vec3f(5e-3), overlay_alpha(pos.xy));
  col = mix(col, vec3f(1.0), overlay_text(pos.xy));
  col = mix(col, vec3f(1.0), overlay_points(pos.xy));
  return vec4f(col, 1.0);
}

fn sample_dejong(pos: vec2f) -> vec3f {
  let display_size = uni.screen_height;
  var px = pos - vec2f((uni.screen_width - display_size) * 0.5, 0.0);
  let sample = px * uni.texture_size / display_size - 0.5;
  let base = vec2i(floor(sample));
  let frac = fract(sample);
  let max_texel = vec2i(i32(uni.texture_size) - 1);
  let p00 = vec2u(clamp(base, vec2i(0), max_texel));
  let p10 = vec2u(clamp(base + vec2i(1, 0), vec2i(0), max_texel));
  let p01 = vec2u(clamp(base + vec2i(0, 1), vec2i(0), max_texel));
  let p11 = vec2u(clamp(base + vec2i(1, 1), vec2i(0), max_texel));
  let c0 = mix(dejong_color(p00), dejong_color(p10), frac.x);
  let c1 = mix(dejong_color(p01), dejong_color(p11), frac.x);
  return mix(c0, c1, frac.y);
}

fn dejong_color(texel: vec2u) -> vec3f {
  let texture_size = u32(uni.texture_size);
  let idx = texel.x + (texture_size - texel.y - 1u) * texture_size;
  let cnt = vec3f(f32(counts_ro[idx][0]), f32(counts_ro[idx][1]), f32(counts_ro[idx][2]));
  var col = cnt * uni.texture_size * uni.texture_size * uni.brightness / f32(frame_info_ro.total_points);
  col = pow(col, vec3f(uni.gamma));
  return clamp(col, vec3f(0.0), vec3f(1.0));
}

fn overlay_pos() -> vec2f {
  return vec2f(uni.screen_width - 320.0 - 16.0, 16.0);
}

fn overlay_alpha(pos: vec2f) -> f32 {
  let rel = pos - overlay_pos();
  if uni.debug_overlay < 0.5 || any(rel < vec2f(0.0)) || any(rel >= vec2f(320.0, 180.0)) {
    return 0.0;
  }
  return 0.95;
}

///////////////////
// TEXT RENDERER //
///////////////////

fn glyph_pixel(ch: u32, px: vec2u) -> bool {
  let row = font[(ch - font_first) * font_height + px.y];
  return (row & (1u << px.x)) != 0u;
}

fn text_char(rel: vec2f, origin: vec2f, scale: f32, ch: u32) -> f32 {
  let local = rel - origin;
  let glyph_size = vec2f(f32(font_width), f32(font_height)) * scale;
  if any(local < vec2f(0.0)) || any(local >= glyph_size) {
    return 0.0;
  }
  return select(0.0, 1.0, glyph_pixel(ch, vec2u(local / scale)));
}

fn overlay_text(pos: vec2f) -> f32 {
  let rel = pos - overlay_pos() - vec2f(16.0);
  let scale = 2.0;
  let glyph_width = f32(font_width) * scale;
  let glyph_height = f32(font_height) * scale;
  let text = array<u32, 13>(68u, 69u, 66u, 85u, 71u, 32u, 79u, 86u, 69u, 82u, 76u, 65u, 89u);
  if uni.debug_overlay < 0.5 || any(rel < vec2f(0.0)) || rel.y >= glyph_height {
    return 0.0;
  }
  let ch = u32(rel.x / glyph_width);
  if ch >= 13u {
    return 0.0;
  }
  return text_char(rel, vec2f(f32(ch) * glyph_width, 0.0), scale, text[ch]);
}

fn overlay_points(pos: vec2f) -> f32 {
  let rel = pos - overlay_pos() - vec2f(16.0, 40.0);
  let scale = 2.0;
  let glyph_width = f32(font_width) * scale;
  let glyph_height = f32(font_height) * scale;
  if uni.debug_overlay < 0.5 || any(rel < vec2f(0.0)) || rel.y >= glyph_height {
    return 0.0;
  }
  let value = frame_info_ro.total_points / 100000u;
  var text = array<u32, 7>(
    48u + (value / 10000u) % 10u,
    48u + (value / 1000u) % 10u,
    48u + (value / 100u) % 10u,
    48u + (value / 10u) % 10u,
    46u,
    48u + value % 10u,
    77u,
  );
  if text[0] == 48u {
    text[0] = 32u;
  }
  if text[0] == 32u && text[1] == 48u {
    text[1] = 32u;
  }
  if text[1] == 32u && text[2] == 48u {
    text[2] = 32u;
  }
  let ch = u32(rel.x / glyph_width);
  if ch >= 7u {
    return 0.0;
  }
  return text_char(rel, vec2f(f32(ch) * glyph_width, 0.0), scale, text[ch]);
}
