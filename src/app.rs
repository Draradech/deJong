use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use crate::dejong::Dejong;
use crate::params::Params;

pub enum App {
    Early(EarlyApp),
    Live(LiveApp),
}

pub struct EarlyApp {
    startup_params: Params,
}

pub struct LiveApp {
    window: Arc<Window>,
    dejong: Dejong,
    fullscreen: bool,
}

impl App {
    pub fn new(startup_params: Params) -> Self {
        Self::Early(EarlyApp { startup_params })
    }
}

impl LiveApp {
    fn toggle_fullscreen(&mut self) {
        if self.fullscreen {
            self.window.set_fullscreen(None);
        } else {
            self.window.set_fullscreen(Some(Fullscreen::Borderless(self.window.current_monitor())));
        }
        self.fullscreen = !self.fullscreen;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let App::Early(early) = self else {
            return;
        };
        let attributes = WindowAttributes::default()
            .with_title("deJong (Rust/WGPU)")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(event_loop.create_window(attributes).expect("failed to create window"));
        let dejong = pollster::block_on(Dejong::new(early.startup_params.clone(), window.clone()));
        println!("controls: D debug, F fullscreen, Esc quit, Up/Down speed, Left/Right budget, P pause");
        println!("startup: {}", dejong.params.describe());
        window.request_redraw();
        *self = App::Live(LiveApp { window, dejong, fullscreen: false });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let App::Live(app) = self else {
            return;
        };
        if window_id != app.window.id() {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => app.dejong.resize(size),
            WindowEvent::RedrawRequested => {
                app.dejong.redraw();
                app.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::KeyD) => app.dejong.toggle_debug_overlay(),
                    PhysicalKey::Code(KeyCode::KeyF) => app.toggle_fullscreen(),
                    PhysicalKey::Code(KeyCode::KeyP) => app.dejong.params.toggle_pause(),
                    PhysicalKey::Code(KeyCode::ArrowUp) => app.dejong.params.adjust_speed_step(1),
                    PhysicalKey::Code(KeyCode::ArrowDown) => app.dejong.params.adjust_speed_step(-1),
                    PhysicalKey::Code(KeyCode::ArrowLeft) => app.dejong.params.adjust_budget(-0.5),
                    PhysicalKey::Code(KeyCode::ArrowRight) => app.dejong.params.adjust_budget(0.5),
                    _ => {}
                }
                println!("params: {}", app.dejong.params.describe());
            }
            _ => {}
        }
    }
}
