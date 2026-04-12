use std::{fs, path::Path, sync::Arc};

use winit::dpi::PhysicalSize;
use winit::window::Window;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryId(usize);

pub type BufferBinding = (u32, BufferId);

enum Pass {
    Clear(ClearPass),
    Compute(ComputePass),
    Copy(CopyPass),
    Render(RenderPass),
    Resolve(ResolvePass),
}

struct ClearPass {}
struct ComputePass {}
struct CopyPass {}
struct RenderPass {
    pipeline: wgpu::RenderPipeline,
    bindings: Vec<BufferBinding>,
    bind_group: wgpu::BindGroup,
    vertices: u32,
    query: Option<QueryId>,
}
struct ResolvePass {}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    shaders: Vec<wgpu::ShaderModule>,
    buffers: Vec<wgpu::Buffer>,
    textures: Vec<wgpu::Texture>,
    samplers: Vec<wgpu::Sampler>,
    queries: Vec<wgpu::QuerySet>,
    passes: Vec<Pass>,
}

fn create_bind_group(
    device: &wgpu::Device,
    buffers: &[wgpu::Buffer],
    layout: &wgpu::BindGroupLayout,
    bindings: &[BufferBinding],
) -> wgpu::BindGroup {
    let entries = bindings
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffers[buffer.0].as_entire_binding(),
        })
        .collect::<Vec<_>>();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.expect("request_adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                ..Default::default()
            })
            .await
            .expect("request device");
        let size = window.inner_size();
        let config =
            surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).expect("get surface config");
        assert!(config.format.is_srgb(), "non-sRGB surface format: {:?}", config.format);
        surface.configure(&device, &config);
        let info = adapter.get_info();
        println!("adapter: {} ({:?})", info.name, info.backend);
        Self {
            surface,
            device,
            queue,
            config,
            shaders: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            queries: Vec::new(),
            passes: Vec::new(),
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    pub fn timestamp_res(&mut self) -> f32 {
        self.queue.get_timestamp_period()
    }

    pub fn create_shader(&mut self, path: impl AsRef<Path>) -> ShaderId {
        let id = ShaderId(self.shaders.len());
        let label = format!("shader_{}", id.0);
        let path = path.as_ref();
        let source =
            fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read shader {}: {}", path.display(), err));
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.shaders.push(shader);
        id
    }

    pub fn create_buffer(&mut self, size: u64, usage: wgpu::BufferUsages) -> BufferId {
        let id = BufferId(self.buffers.len());
        let label = format!("buffer_{}", id.0);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&label),
            size,
            usage,
            mapped_at_creation: false,
        });
        self.buffers.push(buffer);
        id
    }

    pub fn replace_buffer(&mut self, id: BufferId, size: u64, usage: wgpu::BufferUsages) {
        let label = format!("buffer_{}", id.0);
        self.buffers[id.0] = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&label),
            size,
            usage,
            mapped_at_creation: false,
        });
        for pass in &mut self.passes {
            if let Pass::Render(pass) = pass
                && pass.bindings.iter().any(|(_, buffer)| *buffer == id)
            {
                pass.bind_group = create_bind_group(
                    &self.device,
                    &self.buffers,
                    &pass.pipeline.get_bind_group_layout(0),
                    &pass.bindings,
                );
            }
        }
    }

    pub fn create_tsquery(&mut self, count: u32) -> QueryId {
        let id = QueryId(self.queries.len());
        let label = format!("query_{}", id.0);
        let query = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some(&label),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        self.queries.push(query);
        id
    }

    pub fn add_render_pass(
        &mut self,
        shader: ShaderId,
        vsentry: &'static str,
        fsentry: &'static str,
        vertices: u32,
        bindings: &[BufferBinding],
        query: Option<QueryId>,
    ) {
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &self.shaders[shader.0],
                entry_point: Some(vsentry),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shaders[shader.0],
                entry_point: Some(fsentry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let bind_group = create_bind_group(&self.device, &self.buffers, &pipeline.get_bind_group_layout(0), bindings);
        self.passes.push(Pass::Render(RenderPass {
            pipeline,
            bindings: bindings.to_vec(),
            bind_group,
            vertices,
            query,
        }));
    }

    fn encode_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surface_texture: &wgpu::SurfaceTexture,
        pass: &RenderPass,
    ) {
        let view = surface_texture.texture.create_view(&Default::default());
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
        })];
        let timestamp_writes = pass.query.map(|query| wgpu::RenderPassTimestampWrites {
            query_set: &self.queries[query.0],
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&pass.pipeline);
        render_pass.set_bind_group(0, &pass.bind_group, &[]);
        render_pass.draw(0..pass.vertices, 0..1);
    }

    pub fn render(&mut self) {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => panic!("surface lost"),
            wgpu::CurrentSurfaceTexture::Validation => panic!("surface validation error"),
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for pass in &self.passes {
            match pass {
                Pass::Clear(_) => {}
                Pass::Copy(_) => {}
                Pass::Resolve(_) => {}
                Pass::Compute(_) => {}
                Pass::Render(pass) => self.encode_render_pass(&mut encoder, &surface_texture, pass),
            }
        }

        let cmd_buffer = encoder.finish();

        self.queue.submit([cmd_buffer]);

        surface_texture.present();
    }
}
