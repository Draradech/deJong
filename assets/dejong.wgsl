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
  prev_pass_1_start: u32,
  graph_col: u32,
};

@group(0) @binding(0) var<uniform> uni: uniform_t;
@group(0) @binding(1) var<storage> timestamp: timestamp_t;
@group(0) @binding(2) var<storage, read_write> frame_info: frame_info_t;
@group(0) @binding(3) var<storage, read_write> dispatch: vec3u;
@group(0) @binding(4) var<storage, read_write> counts: array<array<atomic<u32>, 3>>;
@group(0) @binding(5) var<storage> counts_ro: array<array<u32, 3>>;
@group(0) @binding(6) var<storage, read_write> filter_values: array<f32>;
@group(0) @binding(7) var<storage> font: array<u32>;
@group(0) @binding(8) var<storage, read_write> text: array<u32>;
@group(0) @binding(9) var<storage, read_write> graph: array<u32>;

const workgroup_size = 16u;
const loop_count = 100u;

fn ticks_ms(end: u32, start: u32) -> f32 {
  return f32(end - start) * uni.timestamp_res / 1e6;
}

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
  frame_info.prev_pass_1_start = frame_info.pass_1_start;
  frame_info.pass_1_points = 64u * workgroup_size * workgroup_size * loop_count;
  frame_info.pass_1_start = timestamp.start;
  frame_info.pass_1_end = timestamp.end;
  var pass_1_ms = ticks_ms(frame_info.pass_1_end, frame_info.pass_1_start);
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

@compute @workgroup_size(graph_size.y)
fn pass_2_timing(@builtin(global_invocation_id) id: vec3u) {
  graph[id.x * graph_size.x + frame_info.graph_col] = 0u;
  if id.x != 0u {
    return;
  }
  frame_info.pass_2_start = timestamp.start;
  frame_info.pass_2_end = timestamp.end;
  var pass_12_ms = ticks_ms(frame_info.pass_2_end, frame_info.pass_1_start);
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
fn pass_3_timing(@builtin(global_invocation_id) id: vec3u) {
  frame_info.pass_3_start = timestamp.start;
  frame_info.pass_3_end = timestamp.end;
  draw_graph_points();
  update_overlay_values();
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
  if uni.debug_overlay > 0.5 {
    col = mix(col, vec3f(0.01), overlay_alpha(vec2u(pos.xy)));
    let graph_col = overlay_graph(vec2u(pos.xy));
    col = mix(col, graph_col.rgb, graph_col.a);
    col = mix(col, vec3f(1.0), overlay_text(vec2u(pos.xy)));
  }
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
  var col = cnt * uni.texture_size * uni.texture_size * uni.brightness / f32(frame_info.total_points);
  col = pow(col, vec3f(uni.gamma));
  return clamp(col, vec3f(0.0), vec3f(1.0));
}

const font_first = 32u;
const font_size = 8u;
const overlay_margin = vec2u(16);
const overlay_padding = vec2u(8);
const text_cols = 26u;
const text_rows = 7u;
const text_scale = 2u;
const glyph_size = vec2u(font_size) * text_scale;
const stride = glyph_size + vec2u(0, glyph_size.y / 2);
const graph_offset = vec2u(0, stride.y * text_rows + overlay_padding.y) + overlay_padding;
const graph_size = vec2u(stride.x * text_cols, 160u);
const overlay_size = vec2u(graph_size.x, stride.y * text_rows + overlay_padding.y + graph_size.y) + overlay_padding * 2;

fn overlay_pos() -> vec2u {
  return vec2u(u32(uni.screen_width) - overlay_size.x - overlay_margin.x, overlay_margin.y);
}

fn overlay_alpha(pos: vec2u) -> f32 {
  let rel = pos - overlay_pos();
  if any(rel < vec2u(0)) || any(rel >= overlay_size) {
    return 0.0;
  }
  return 0.95;
}

fn overlay_text(pos: vec2u) -> f32 {

  let rel = pos - overlay_pos() - overlay_padding;
  let cell = rel / stride;
  if cell.x >= text_cols || cell.y >= text_rows {
    return 0.0;
  }

  let local = rel - cell * stride;
  if any(local >= glyph_size) {
    return 0.0;
  }

  let ch = text[cell.y * text_cols + cell.x];
  let px = local / text_scale;
  let row = font[(ch - font_first) * font_size + px.y];

  return select(0.0, 1.0, (row & (1u << px.x)) != 0u);
}

fn overlay_graph(pos: vec2u) -> vec4f {
  let rel = pos - overlay_pos() - graph_offset;
  if any(rel >= graph_size) {
    return vec4f(0.0);
  }
  let x = (rel.x + frame_info.graph_col) % graph_size.x;
  switch (graph[rel.y * graph_size.x + x]) {
    case 1u: {return vec4f(0.0, 1.0, 0.0, 1.0);}
    case 2u: {return vec4f(1.0, 0.0, 0.0, 1.0);}
    case 3u: {return vec4f(0.0, 0.1, 0.0, 1.0);}
    case 4u: {return vec4f(0.15, 0.1, 0.0, 1.0);}
    case 5u: {return vec4f(0.15, 0.0, 0.0, 1.0);}
    case 6u: {return vec4f(0.0, 1.0, 1.0, 1.0);}
    default: {return vec4f(vec3f(0.0), 0.5);}
  }
}

fn pow10(n: u32) -> u32 {
  var x = 1u;
  for (var i = 0u; i < n; i++) {
    x *= 10u;
  }
  return x;
}

fn write_uint(offset: u32, value: u32, digits: u32) {
  var v = value;
  for (var i = 0u; i < digits; i++) {
    text[offset + digits - i - 1u] = 48u + v % 10u;
    v /= 10u;
  }
}

fn write_number(offset: u32, value: f32, digits_base: u32, digits_fract: u32) {
  let scale = pow10(digits_fract);
  let scaled = u32(max(value, 0.0) * f32(scale) + 0.5);
  write_uint(offset, scaled / scale, digits_base);
  for (var i = 0u; i + 1u < digits_base; i++) {
    if text[offset + i] != 48u {
      break;
    }
    text[offset + i] = 32u;
  }
  if digits_fract > 0u {
    text[offset + digits_base] = 46u;
    write_uint(offset + digits_base + 1u, scaled % scale, digits_fract);
  }
}

fn update_overlay_values() {
  for (var i = 0u; i < 13u; i++) {
    var value = 0.0;
    var fmt = vec3u(0u);

    switch (i) {
      case 0u: {
        value = ticks_ms(frame_info.pass_1_start, frame_info.prev_pass_1_start);
        fmt = vec3u(3u, 2u, 7u);
      }
      case 1u: {
        value = 1000.0 / max(ticks_ms(frame_info.pass_1_start, frame_info.prev_pass_1_start), 0.01);
        fmt = vec3u(3u, 1u, 17u);
      }
      case 2u: {
        value = ticks_ms(frame_info.render_end, frame_info.prev_pass_1_start);
        fmt = vec3u(3u, 2u, 26u + 7u);
      }
      case 3u: {
        value = f32(frame_info.total_points) / 1e6;
        fmt = vec3u(3u, 1u, 26u + 17u);
      }
      case 4u: {
        value = ticks_ms(frame_info.pass_1_end, frame_info.pass_1_start);
        fmt = vec3u(3u, 2u, 52u + 7u);
      }
      case 5u: {
        value = f32(frame_info.pass_1_points) / 1e6;
        fmt = vec3u(3u, 1u, 52u + 17u);
      }
      case 6u: {
        value = ticks_ms(frame_info.pass_2_end, frame_info.pass_2_start);
        fmt = vec3u(3u, 2u, 78u + 7u);
      }
      case 7u: {
        value = f32(frame_info.pass_2_points) / 1e6;
        fmt = vec3u(3u, 1u, 78u + 17u);
      }
      case 8u: {
        value = ticks_ms(frame_info.pass_3_end, frame_info.pass_3_start);
        fmt = vec3u(3u, 2u, 104u + 7u);
      }
      case 9u: {
        value = f32(frame_info.pass_3_points) / 1e6;
        fmt = vec3u(3u, 1u, 104u + 17u);
      }
      case 10u: {
        value = ticks_ms(frame_info.render_end, frame_info.render_start);
        fmt = vec3u(3u, 2u, 130u + 7u);
      }
      case 11u: {
        value = uni.texture_size;
        fmt = vec3u(4u, 0u, 156u + 9u);
      }
      default: {
        value = uni.texture_size * uni.texture_size * 12.0 / 1024.0 / 1024.0;
        fmt = vec3u(3u, 1u, 156u + 17u);
      }
    }

    if i < 11 {
      filter_values[i] = value + (filter_values[i] - value) * 0.98;
      value = filter_values[i];
    }
    write_number(fmt.z, value, fmt.x, fmt.y);
  }
}

fn draw_graph_points() {
  draw_graph_log_point(6u, f32(frame_info.total_points), 1e5, 1e9);
  draw_graph_point(2u, ticks_ms(frame_info.render_end, frame_info.prev_pass_1_start) / 20.0);
  draw_graph_point(3u, ticks_ms(frame_info.pass_1_end, frame_info.pass_1_start) / 20.0);
  draw_graph_point(4u, ticks_ms(frame_info.pass_2_end, frame_info.pass_2_start) / 20.0);
  draw_graph_point(5u, ticks_ms(frame_info.pass_3_end, frame_info.pass_3_start) / 20.0);
  draw_graph_point(1u, ticks_ms(frame_info.pass_1_start, frame_info.prev_pass_1_start) / 20.0);
  frame_info.graph_col = (frame_info.graph_col + 1u) % graph_size.x;
}

fn draw_graph_log_point(graph_id: u32, value: f32, minv: f32, maxv: f32) {
  let logv = (log2(clamp(value, minv, maxv)) - log2(minv)) / (log2(maxv) - log2(minv));
  draw_graph_point(graph_id, logv);
}
  
fn draw_graph_point(graph_id: u32, yf: f32) {
  let y = min(u32(yf * f32(graph_size.y)), graph_size.y - 2u);
  graph[(graph_size.y - y - 1u) * graph_size.x + frame_info.graph_col] = graph_id;
  graph[(graph_size.y - y - 2u) * graph_size.x + frame_info.graph_col] = graph_id;
}
