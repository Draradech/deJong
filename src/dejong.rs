use std::mem::size_of;
use std::sync::Arc;

use bytemuck::bytes_of;
use wgpu::BufferUsages;

use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::params::{Params, UniformData};
use crate::renderer::{BufferId, Renderer};

const U32_SIZE: u64 = size_of::<u32>() as u64;
const UNIFORM_DATA_SIZE: u64 = size_of::<UniformData>() as u64;

pub struct Dejong {
    renderer: Renderer,
    pub(crate) params: Params,
    uniform_id: BufferId,
    data_id: BufferId,
    screen_size: PhysicalSize<u32>,
    frame_index: u32,
}

impl Dejong {
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

        let shader = renderer.create_shader("assets/dejong.wgsl");
        let tsquery = renderer.create_tsquery(2);
        let uniform = renderer.create_buffer(UNIFORM_DATA_SIZE, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let frameinfo = renderer.create_buffer(13 * U32_SIZE, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let timestamp = renderer.create_buffer(16 * U32_SIZE, BufferUsages::STORAGE | BufferUsages::QUERY_RESOLVE);
        let indirect = renderer.create_buffer(12 * U32_SIZE, BufferUsages::STORAGE | BufferUsages::INDIRECT);
        let data = renderer.create_buffer(data_size, BufferUsages::STORAGE | BufferUsages::COPY_DST);

        renderer.add_clear_pass(data);
        let bind = [(0, uniform), (2, frameinfo), (4, data)];
        renderer.add_compute_pass(shader, "dejong", 16, &bind, Some(tsquery));
        // renderer.add_resolve_query(tsquery, timestamp);
        // let bind = [(0, uniform), (1, timestamp), (2, frameinfo), (3, indirect)];
        // renderer.add_compute_pass(shader, "pass_1_timing", 1, &bind, None);
        // let bind = [(0, uniform), (2, frameinfo), (4, data)];
        // renderer.add_compute_pass_indirect(shader, "dejong", indirect, &bind, Some(tsquery));
        // renderer.add_resolve_query(tsquery, timestamp);
        // let bind = [(0, uniform), (1, timestamp), (2, frameinfo), (3, indirect)];
        // renderer.add_compute_pass(shader, "pass_2_timing", 1, &bind, None);
        // let bind = [(0, uniform), (2, frameinfo), (4, data)];
        // renderer.add_compute_pass_indirect(shader, "dejong", indirect, &bind, Some(tsquery));
        // renderer.add_resolve_query(tsquery, timestamp);
        // let bind = [(1, timestamp), (2, frameinfo)];
        // renderer.add_compute_pass(shader, "pass_3_timing", 1, &bind, None);
        let bind = [(0, uniform), (5, data), (6, frameinfo)];
        renderer.add_render_pass(shader, "dejong_vs", "dejong_fs", 3, &bind, Some(tsquery));
        // renderer.add_resolve_query(tsquery, timestamp);
        // let bind = [(1, timestamp), (2, frameinfo)];
        // renderer.add_compute_pass(shader, "render_timing", 1, &bind, None);
        // renderer.add_buffer_download(frameinfo, 4, readback);

        let _ = (shader, tsquery, frameinfo, timestamp, indirect);

        Self { renderer, params, uniform_id: uniform, data_id: data, screen_size, frame_index: 0 }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.renderer.resize(size);
        self.screen_size = size;
        let data_size = Self::data_buffer_size(self.screen_size.height, self.params.scale);
        self.renderer.replace_buffer(self.data_id, data_size, BufferUsages::STORAGE | BufferUsages::COPY_DST);
    }

    pub fn toggle_debug_overlay(&mut self) {}

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
        self.renderer.update_buffer(self.uniform_id, bytes_of(&uniform_data));
        self.renderer.render();
    }
}
