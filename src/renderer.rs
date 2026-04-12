use std::{fs, path::Path, sync::Arc};

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
struct RenderPass {}
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

    pub fn create_shader(&mut self, path: impl AsRef<Path>) -> ShaderId {
        let id = ShaderId(self.shaders.len());
        let label = format!("shader_{}", id.0);
        let path = path.as_ref();
        let source = fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read shader {}: {}", path.display(), err));
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
        _vertices: u32,
        _bindings: &[BufferBinding],
        _query: Option<QueryId>,
    ) {
        let _pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
    }

    pub fn render(&mut self) {
        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for pass in &self.passes {
            match pass {
                Pass::Clear(_) => {}
                Pass::Copy(_) => {}
                Pass::Resolve(_) => {}
                Pass::Compute(_) => {}
                Pass::Render(_) => {}
            }
        }
        let cmd_buffer = encoder.finish();
        self.queue.submit([cmd_buffer]);
    }
}
