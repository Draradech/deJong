use std::sync::Arc;

use wgpu::BufferUsages;

use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::params::Params;
use crate::renderer::{BufferId, Renderer};

pub struct Dejong {
    renderer: Renderer,
    pub(crate) params: Params,
    data_buffer_id: BufferId
}

impl Dejong {
    fn resize_data_buffer(&mut self, size: u32) {
        self.renderer.replace_buffer(self.data_buffer_id, size as u64, BufferUsages::STORAGE | BufferUsages::COPY_DST);
    }

    pub async fn new(params: Params, window: Arc<Window>) -> Self {
        let mut renderer = Renderer::new(window).await;

        let shader = renderer.create_shader("assets/dejong.wgsl");
        let tsquery = renderer.create_tsquery(8);
        let uniform = renderer.create_buffer(std::mem::size_of::<crate::params::UniformData>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let frameinfo = renderer.create_buffer(13, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let timestamp = renderer.create_buffer(4, BufferUsages::STORAGE | BufferUsages::QUERY_RESOLVE);
        let indirect = renderer.create_buffer(3, BufferUsages::STORAGE | BufferUsages::INDIRECT);
        let data = renderer.create_buffer(0, BufferUsages::STORAGE | BufferUsages::COPY_DST);

        // renderer.add_clear(data);
        // let bind = [(0, uniform), (2, frameinfo), (4, data)];
        // renderer.add_compute_pass(shader, "deJong", 16, &bind, Some(tsquery));
        // renderer.add_resolve_query(tsquery, timestamp);
        // let bind = [(0, uniform), (1, timestamp), (2, frameinfo), (3, indirect)];
        // renderer.add_compute_pass(shader, "pass1t", 1, &bind, None);
        // let bind = [(0, uniform), (2, frameinfo), (4, data)];
        // renderer.add_compute_pass_indirect(shader, "deJong", indirect, &bind, Some(tsquery));
        // renderer.add_resolve_query(tsquery, timestamp);
        // let bind = [(0, uniform), (1, timestamp), (2, frameinfo), (3, indirect)];
        // renderer.add_compute_pass(shader, "pass2t", 1, &bind, None);
        // let bind = [(0, uniform), (2, frameinfo), (4, data)];
        // renderer.add_compute_pass_indirect(shader, "deJong", indirect, &bind, Some(tsquery));
        // renderer.add_resolve_query(tsquery, timestamp);
        // let bind = [(1, timestamp), (2, frameinfo)];
        // renderer.add_compute_pass(shader, "pass3t", 1, &bind, None);
        // let bind = [(0, uniform), (2, frameinfo), (5, data)];
        // renderer.add_render_pass(shader, "vs", "fs", 3, &bind, Some(tsquery));
        // renderer.add_resolve_query(tsquery, timestamp);
        // let bind = [(1, timestamp), (2, frameinfo)];
        // renderer.add_compute_pass(shader, "passrt", 1, &bind, None);
        // renderer.add_buffer_download(frameinfo, 4, readback);

        let _ = (shader, tsquery, uniform, frameinfo, timestamp, indirect);

        Self { renderer, params, data_buffer_id: data }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {}

    pub fn toggle_debug_overlay(&mut self) {}

    pub fn redraw(&mut self) {
        //let uniform_data = self.params.uniforms(frame_index, texture_size, timestamp_period_ns, viewport_size);
        //self.renderer.update_buffer("uniform", uniform_data);
        self.renderer.render();
    }
}
