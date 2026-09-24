use std::mem::size_of;
use std::sync::Arc;

use bytemuck::{bytes_of, cast_slice};
use wgpu::BufferUsages;

use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::font::{FONT_FIRST, FONT_HEIGHT, FONT_LAST, FONT_PIXELS, FONT_WIDTH};
use crate::params::{Params, UniformData};
use crate::renderer::{BufferId, Renderer};

const U32_SIZE: u64 = size_of::<u32>() as u64;
const UNIFORM_DATA_SIZE: u64 = size_of::<UniformData>() as u64;
const FONT_DATA_SIZE: u64 = size_of::<[u32; FONT_PIXELS.len()]>() as u64;
const TEXT_SCALE: usize = 1;
const TEXT_ROW_GAP: usize = 4;
const PERF_TEXT_COLS: usize = 26;
const PERF_TEXT_ROWS: usize = 7;
const PERF_TEXT_GRID_SIZE: u64 = size_of::<[u32; PERF_TEXT_COLS * PERF_TEXT_ROWS]>() as u64;
const PERF_GRAPH_WIDTH: usize = PERF_TEXT_COLS * FONT_WIDTH as usize * TEXT_SCALE;
const PERF_GRAPH_HEIGHT: usize = 160;
const PERF_GRAPH_SIZE: u64 = size_of::<[u32; PERF_GRAPH_WIDTH * PERF_GRAPH_HEIGHT]>() as u64;
const CTRL_TEXT_COLS: usize = 26;
const CTRL_TEXT_ROWS: usize = 12;
const CTRL_TEXT_GRID_SIZE: u64 = size_of::<[u32; CTRL_TEXT_COLS * CTRL_TEXT_ROWS]>() as u64;

pub struct Dejong {
    renderer: Renderer,
    pub(crate) params: Params,
    uniform_id: BufferId,
    data_id: BufferId,
    ctrl_text_id: BufferId,
    screen_size: PhysicalSize<u32>,
    frame_index: u32,
}

impl Dejong {
    fn shader_prelude() -> String {
        format!(
            "\
const font_first = {}u;
const font_width = {}u;
const font_height = {}u;
const text_scale = {}u;
const text_row_gap = {}u;
const perf_text_cols = {}u;
const perf_text_rows = {}u;
const perf_graph_width = {}u;
const perf_graph_height = {}u;
const ctrl_text_cols = {}u;
const ctrl_text_rows = {}u;
",
            FONT_FIRST,
            FONT_WIDTH,
            FONT_HEIGHT,
            TEXT_SCALE,
            TEXT_ROW_GAP,
            PERF_TEXT_COLS,
            PERF_TEXT_ROWS,
            PERF_GRAPH_WIDTH,
            PERF_GRAPH_HEIGHT,
            CTRL_TEXT_COLS,
            CTRL_TEXT_ROWS,
        )
    }

    fn perf_text_grid() -> [u32; PERF_TEXT_COLS * PERF_TEXT_ROWS] {
        Self::pack_text_grid::<PERF_TEXT_COLS, PERF_TEXT_ROWS, { PERF_TEXT_COLS * PERF_TEXT_ROWS }>(&[
            "Frame  000.00 ms 000.0 fps",
            "Total  000.00 ms 000.0 M",
            "Pass 1 000.00 ms 000.0 M",
            "Pass 2 000.00 ms 000.0 M",
            "Pass 3 000.00 ms 000.0 M",
            "Render 000.00 ms",
            "Texture  0000 px 000.0 MB",
        ])
    }

    fn pack_char(byte: u8) -> u32 {
        match byte {
            b if (FONT_FIRST as u8..=FONT_LAST as u8).contains(&b) => b as u32,
            _ => b'?' as u32,
        }
    }

    fn pack_text_grid<const COLS: usize, const ROWS: usize, const SIZE: usize>(
        lines: &[impl AsRef<str>],
    ) -> [u32; SIZE] {
        let mut grid = [b' ' as u32; SIZE];
        for (row, line) in lines.iter().enumerate() {
            for (col, byte) in line.as_ref().bytes().enumerate() {
                grid[row * COLS + col] = Self::pack_char(byte);
            }
        }
        grid
    }

    fn data_texture_size(screen_height: u32, scale: f32) -> u32 {
        (screen_height as f32 * scale * 0.01) as u32
    }

    fn data_buffer_size(screen_height: u32, scale: f32) -> u64 {
        let texture_size = Self::data_texture_size(screen_height, scale);
        texture_size as u64 * texture_size as u64 * 3 * U32_SIZE
    }

    pub async fn new(params: Params, window: Arc<Window>) -> Self {
        let mut renderer = Renderer::new(window.clone()).await;

        let screen_size = window.inner_size();
        let data_size = Self::data_buffer_size(screen_size.height, params.scale);

        let shader_source = Self::shader_prelude() + include_str!("dejong.wgsl");
        let shader = renderer.create_shader(&shader_source);
        let tsquery = renderer.create_tsquery();
        let uniform = renderer.create_buffer(UNIFORM_DATA_SIZE, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let frameinfo = renderer.create_buffer(15 * U32_SIZE, BufferUsages::STORAGE);
        let filter = renderer.create_buffer(13 * U32_SIZE, BufferUsages::STORAGE);
        let timestamp = renderer.create_buffer(4 * U32_SIZE, BufferUsages::STORAGE | BufferUsages::QUERY_RESOLVE);
        let indirect = renderer.create_buffer(3 * U32_SIZE, BufferUsages::STORAGE | BufferUsages::INDIRECT);
        let data = renderer.create_buffer(data_size, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let font = renderer.create_buffer(FONT_DATA_SIZE, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let perf_text = renderer.create_buffer(PERF_TEXT_GRID_SIZE, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let graph = renderer.create_buffer(PERF_GRAPH_SIZE, BufferUsages::STORAGE);
        let ctrl_text = renderer.create_buffer(CTRL_TEXT_GRID_SIZE, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        renderer.update_buffer(font, cast_slice(&FONT_PIXELS));
        renderer.update_buffer(perf_text, cast_slice(&Self::perf_text_grid()));

        renderer.add_clear_pass(data);
        let bind = [(0, uniform), (2, frameinfo), (4, data)];
        renderer.add_compute_pass(shader, "dejong", 64, &bind, Some(tsquery));
        renderer.add_resolve_query(tsquery, timestamp);
        let bind = [(0, uniform), (1, timestamp), (2, frameinfo), (3, indirect)];
        renderer.add_compute_pass(shader, "pass_1_timing", 1, &bind, None);
        let bind = [(0, uniform), (2, frameinfo), (4, data)];
        renderer.add_compute_pass_indirect(shader, "dejong", indirect, &bind, Some(tsquery));
        renderer.add_resolve_query(tsquery, timestamp);
        let bind = [(0, uniform), (1, timestamp), (2, frameinfo), (3, indirect), (9, graph)];
        renderer.add_compute_pass(shader, "pass_2_timing", 1, &bind, None);
        let bind = [(0, uniform), (2, frameinfo), (4, data)];
        renderer.add_compute_pass_indirect(shader, "dejong", indirect, &bind, Some(tsquery));
        renderer.add_resolve_query(tsquery, timestamp);
        let bind = [(0, uniform), (1, timestamp), (2, frameinfo), (6, filter), (8, perf_text), (9, graph)];
        renderer.add_compute_pass(shader, "pass_3_timing", 1, &bind, None);
        let bind = [(0, uniform), (2, frameinfo), (5, data), (7, font), (8, perf_text), (9, graph), (10, ctrl_text)];
        renderer.add_render_pass(shader, "dejong_vs", "dejong_fs", 3, &bind, Some(tsquery));
        renderer.add_resolve_query(tsquery, timestamp);
        let bind = [(1, timestamp), (2, frameinfo)];
        renderer.add_compute_pass(shader, "render_timing", 1, &bind, None);

        Self {
            renderer,
            params,
            uniform_id: uniform,
            data_id: data,
            ctrl_text_id: ctrl_text,
            screen_size,
            frame_index: 0,
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.renderer.resize(size);
        self.screen_size = size;
        let data_size = Self::data_buffer_size(self.screen_size.height, self.params.scale);
        self.renderer.replace_buffer(self.data_id, data_size, BufferUsages::STORAGE | BufferUsages::COPY_DST);
    }

    pub fn redraw(&mut self) {
        self.frame_index += 1;
        self.params.advance_t();
        let texture_size = Self::data_texture_size(self.screen_size.height, self.params.scale);
        let uniform_data = self.params.uniforms(
            self.frame_index,
            texture_size,
            self.renderer.timestamp_res(),
            self.screen_size.into(),
        );
        let ctrl_text_grid = Self::pack_text_grid::<CTRL_TEXT_COLS, CTRL_TEXT_ROWS, { CTRL_TEXT_COLS * CTRL_TEXT_ROWS }>(
            &self.params.ctrl_overlay_lines(),
        );
        self.renderer.update_buffer(self.ctrl_text_id, cast_slice(&ctrl_text_grid));
        self.renderer.update_buffer(self.uniform_id, bytes_of(&uniform_data));
        self.renderer.render();
    }
}
