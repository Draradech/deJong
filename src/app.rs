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
    Live(Box<LiveApp>),
}

pub struct EarlyApp {
    startup_params: Params,
}

pub struct LiveApp {
    window: Arc<Window>,
    dejong: Dejong,
}

impl App {
    pub fn new(startup_params: Params) -> Self {
        Self::Early(EarlyApp { startup_params })
    }
}

impl LiveApp {
    fn toggle_fullscreen(&mut self) {
        if self.dejong.params.fullscreen {
            self.window.set_fullscreen(None);
        } else {
            self.window.set_fullscreen(Some(Fullscreen::Borderless(self.window.current_monitor())));
        }
        self.dejong.params.fullscreen = !self.dejong.params.fullscreen;
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
        println!(
            "controls: C control box, D debug, F fullscreen, Up/Down select, Left/Right adjust, Space toggle, Esc quit"
        );
        window.request_redraw();
        *self = App::Live(Box::new(LiveApp { window, dejong }));
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
                    PhysicalKey::Code(KeyCode::KeyC) if !event.repeat => {
                        app.dejong.params.control_visible = !app.dejong.params.control_visible;
                    }
                    PhysicalKey::Code(KeyCode::KeyD) if !event.repeat => {
                        app.dejong.params.toggle_debug_overlay();
                    }
                    PhysicalKey::Code(KeyCode::KeyF) if !event.repeat => {
                        app.toggle_fullscreen();
                    }
                    PhysicalKey::Code(KeyCode::ArrowUp) if app.dejong.params.control_visible => {
                        app.dejong.params.move_selection(-1);
                    }
                    PhysicalKey::Code(KeyCode::ArrowDown) if app.dejong.params.control_visible => {
                        app.dejong.params.move_selection(1);
                    }
                    PhysicalKey::Code(KeyCode::ArrowLeft | KeyCode::ArrowRight)
                        if app.dejong.params.control_visible =>
                    {
                        let direction =
                            if event.physical_key == PhysicalKey::Code(KeyCode::ArrowRight) { 1 } else { -1 };
                        if app.dejong.params.adjust_selected(direction) {
                            app.dejong.resize_data_buffer();
                        }
                    }
                    PhysicalKey::Code(KeyCode::Space) if app.dejong.params.control_visible && !event.repeat => {
                        app.dejong.params.toggle_selected();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}
