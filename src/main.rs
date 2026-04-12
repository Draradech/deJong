mod app;
mod dejong;
mod params;
mod renderer;

use winit::event_loop::{ControlFlow, EventLoop};

use app::App;
use params::Params;

fn main() {
    env_logger::init();

    let params = Params::from_cli();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(params);
    event_loop.run_app(&mut app).expect("event loop run error");
}
