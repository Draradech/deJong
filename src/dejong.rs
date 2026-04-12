use std::sync::Arc;

use wgpu::BufferUsages;

use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::params::Params;
use crate::renderer::Renderer;

pub struct Dejong {
    renderer: Renderer,
    pub(crate) params: Params,
}

impl Dejong {
    fn create_data_buffer(size: u32) {
        // renderer.createBuffer("data", size, BufferUsages::STORAGE | BufferUsages::COPY_DST);
    }

    pub async fn new(params: Params, window: Arc<Window>) -> Self {
        let renderer = pollster::block_on(Renderer::new(window));

        // renderer.create_shader("dejong", "assets/dejong.wgsl");
        // renderer.create_tsquery("tsquery");
        // renderer.create_buffer("uniform", Params::UniformData::length(), BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        // renderer.create_buffer("frameinfo", 13, BufferUsages::STORAGE | BufferUsages::COPY_SCR);
        // renderer.create_buffer("timestamp", 4,  BufferUsages::STORAGE | BufferUsages::QUERY_RESOLVE);
        // renderer.create_buffer("indirect", 3, BufferUsages::STORAGE | BufferUsages::INDIRECT);

        // renderer.add_clear("data");
        // let bind = [("uniform", 0), ("frameinfo", 2), ("data", 4)];
        // renderer.add_compute_pass("dejong", "deJong", 16, &bind, Some("tsquery"));
        // renderer.add_resolve_query("tsquery", "timestamp");
        // let bind = [("uniform", 0), ("timestamp", 1), ("frameinfo", 2), ("indirect", 3)];
        // renderer.add_compute_pass("dejong", "pass1t", 1, &bind, None);
        // let bind = [("uniform", 0), ("frameinfo", 2), ("data", 4)];
        // renderer.add_compute_pass_indirect("dejong", "deJong", "indirect", bind, Some("tsquery"));
        // renderer.add_resolve_query("tsquery", "timestamp");
        // let bind = [("uniform", 0), ("timestamp", 1), ("frameinfo", 2), ("indirect", 3)];
        // renderer.add_compute_pass("dejong", "pass2t", 1, bind, None);
        // let bind = [("uniform", 0), ("frameinfo", 2), ("data", 4)];
        // renderer.add_compute_pass_indirect("dejong", "deJong", "indirect", bind, Some("tsquery"));
        // renderer.add_resolve_query("tsquery", "timestamp");
        // let bind = [("timestamp", 1), ("frameinfo", 2)];
        // renderer.add_compute_pass("dejong", "pass3t", 1, bind, None);
        // let bind = [("uniform", 0), ("frameinfo", 2), ("data", 5)];
        // renderer.add_render_pass("dejong", "vs", "fs", 3, bind, Some("tsquery"));
        // renderer.add_resolve_query("tsquery", "timestamp");
        // let bind = [("timestamp", 1), ("frameinfo", 2)];
        // renderer.add_compute_pass("dejong", "passrt", 1, bind, None);
        // renderer.add_buffer_download("frameinfo", 4, readback);

        Self { renderer, params }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {}

    pub fn toggle_debug_overlay(&mut self) {}

    pub fn redraw(&mut self) {
        //let uniform_data = self.params.uniforms(frame_index, texture_size, timestamp_period_ns, viewport_size);
        //self.renderer.update_buffer("uniform", uniform_data);
        self.renderer.render();
    }
}
