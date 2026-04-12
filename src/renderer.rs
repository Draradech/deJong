use std::{collections::HashMap, sync::Arc};

use winit::window::Window;

type ResourceName = &'static str;
type Literal = &'static str;
type Bindings = HashMap<ResourceName, u32>;

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
    shaders: HashMap<ResourceName, wgpu::ShaderModule>,
    buffers: HashMap<ResourceName, wgpu::Buffer>,
    textures: HashMap<ResourceName, wgpu::Texture>,
    samplers: HashMap<ResourceName, wgpu::Sampler>,
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
            shaders: HashMap::new(),
            buffers: HashMap::new(),
            textures: HashMap::new(),
            samplers: HashMap::new(),
            passes: Vec::new(),
        }
    }

    pub fn add_render_pass(
        &mut self,
        shader: ResourceName,
        vsentry: Literal,
        fsentry: Literal,
        vertices: u32,
        bindings: Bindings,
        query: Option<ResourceName>,
    ) {
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &self.shaders[shader],
                entry_point: Some(vsentry),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shaders[shader],
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
                Pass::Clear(p) => {}
                Pass::Copy(p) => {}
                Pass::Resolve(p) => {}
                Pass::Compute(p) => {}
                Pass::Render(p) => {}
            }
        }
        let cmd_buffer = encoder.finish();
        self.queue.submit([cmd_buffer]);
    }
}
