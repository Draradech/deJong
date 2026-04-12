mod app;
mod dejong;
mod params;
mod renderer;

use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

use app::App;
use params::Params;

fn main() {
    env_logger::init();

    let params = Cli::parse().into_params();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(params);
    event_loop.run_app(&mut app).expect("event loop run error");
}

#[derive(Parser, Debug)]
#[command(name = "dejong_rust")]
struct Cli {
    #[arg(long)]
    t: Option<f64>,
    #[arg(long)]
    scale: Option<f32>,
    #[arg(long)]
    speed: Option<f32>,
    #[arg(long)]
    brightness: Option<f32>,
    #[arg(long)]
    budget: Option<f32>,
    #[arg(long)]
    gamma: Option<f32>,
}

impl Cli {
    fn into_params(self) -> Params {
        let mut params = Params::default();

        if let Some(t) = self.t {
            params.t = t;
            params.paused = true;
        }
        if let Some(scale) = self.scale {
            params.scale = scale;
        }
        if let Some(speed) = self.speed {
            params.speed = speed;
        }
        if let Some(brightness) = self.brightness {
            params.brightness = brightness;
        }
        if let Some(budget) = self.budget {
            params.budget = budget;
        }
        if let Some(gamma) = self.gamma {
            params.gamma = gamma;
        }

        params
    }
}
