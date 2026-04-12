use bytemuck::{Pod, Zeroable};
use rand::RngExt;

const MIN_SPEED_PERCENT: f32 = 1.0;
const MAX_SPEED_PERCENT: f32 = 10000.0;
const SPEED_STEPS_PER_DECADE: f32 = 5.0;
const MIN_BUDGET_MS: f32 = 0.5;
const MAX_BUDGET_MS: f32 = 100.0;
const T_INIT_MARGIN: f64 = 50.0;
const T_PERIOD: f64 = std::f64::consts::PI * 200.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct UniformData {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub frame: f32,
    pub texture_size: f32,
    pub brightness: f32,
    pub gamma: f32,
    pub budget: f32,
    pub timestamp_period_ns: f32,
    pub viewport_width: f32,
    pub viewport_height: f32,
}

#[derive(Debug, Clone)]
pub struct Params {
    pub t: f64,
    pub speed: f32,
    pub brightness: f32,
    pub budget: f32,
    pub gamma: f32,
    pub scale: f32,
    pub paused: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            t: rand::rng().random_range(T_INIT_MARGIN..(T_PERIOD - T_INIT_MARGIN)),
            speed: 100.0,
            brightness: 100.0,
            budget: 15.0,
            gamma: 1.8,
            scale: 100.0,
            paused: false,
        }
    }
}

impl Params {
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn adjust_speed_step(&mut self, step_delta: i32) {
        let current_step =
            (self.speed.clamp(MIN_SPEED_PERCENT, MAX_SPEED_PERCENT).log10() * SPEED_STEPS_PER_DECADE).round() as i32;
        let min_step = (MIN_SPEED_PERCENT.log10() * SPEED_STEPS_PER_DECADE).round() as i32;
        let max_step = (MAX_SPEED_PERCENT.log10() * SPEED_STEPS_PER_DECADE).round() as i32;
        let next_step = (current_step + step_delta).clamp(min_step, max_step);
        self.speed = 10.0_f32.powf(next_step as f32 / SPEED_STEPS_PER_DECADE);
    }

    pub fn adjust_budget(&mut self, delta_ms: f32) {
        self.budget = (self.budget + delta_ms).clamp(MIN_BUDGET_MS, MAX_BUDGET_MS);
    }

    pub fn advance_t(&mut self) {
        if !self.paused {
            self.t += 1e-6_f64 * self.speed as f64;
        }
    }

    pub fn uniforms(
        &self,
        frame_index: u32,
        texture_size: u32,
        timestamp_period_ns: f32,
        viewport_size: (u32, u32),
    ) -> UniformData {
        let t = self.t;
        UniformData {
            a: (4.0_f64 * (t * 1.03_f64).sin()) as f32,
            b: (4.0_f64 * (t * 1.07_f64).sin()) as f32,
            c: (4.0_f64 * (t * 1.09_f64).sin()) as f32,
            d: (4.0_f64 * (t * 1.13_f64).sin()) as f32,
            frame: frame_index as f32,
            texture_size: texture_size as f32,
            brightness: self.brightness * 4e-6,
            budget: self.budget,
            timestamp_period_ns,
            viewport_width: viewport_size.0 as f32,
            viewport_height: viewport_size.1 as f32,
            gamma: self.gamma,
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "scale={:.0}% speed={:.1}% bright={:.0}% gamma={:.2} budget={:.2}ms paused={}",
            self.scale, self.speed, self.brightness, self.gamma, self.budget, self.paused
        )
    }
}
