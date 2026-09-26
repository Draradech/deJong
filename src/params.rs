use bytemuck::{Pod, Zeroable};
use clap::Parser;
use rand::RngExt;

const MIN_SPEED_PERCENT: f32 = 1.0;
const MAX_SPEED_PERCENT: f32 = 10000.0;
const LOG_STEPS_PER_DECADE: f32 = 5.0;
const MIN_STEP: f32 = 0.001;
const MAX_STEP: f32 = 1000.0;
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
    pub timestamp_res: f32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub debug_overlay: f32,
    pub control_overlay: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Speed,
    Step,
    T,
    A,
    B,
    C,
    D,
    Scale,
    Brightness,
    Gamma,
    Budget,
}

impl Control {
    pub const ALL: [Self; 11] = [
        Self::Speed,
        Self::Step,
        Self::T,
        Self::A,
        Self::B,
        Self::C,
        Self::D,
        Self::Scale,
        Self::Brightness,
        Self::Gamma,
        Self::Budget,
    ];

    fn coefficient_index(self) -> Option<usize> {
        match self {
            Self::A => Some(0),
            Self::B => Some(1),
            Self::C => Some(2),
            Self::D => Some(3),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    t: f64,
    speed: f32,
    step: f32,
    brightness: f32,
    budget: f32,
    gamma: f32,
    pub scale: f32,
    paused: bool,
    coefficients: [f64; 4],
    coefficient_auto: [bool; 4],
    debug_overlay: bool,
    pub fullscreen: bool,
    pub control_visible: bool,
    selected_control: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            t: rand::rng().random_range(T_INIT_MARGIN..(T_PERIOD - T_INIT_MARGIN)),
            speed: 100.0,
            step: 1.0,
            brightness: 100.0,
            budget: 15.0,
            gamma: 1.5,
            scale: 100.0,
            paused: false,
            coefficients: [0.0; 4],
            coefficient_auto: [true; 4],
            debug_overlay: false,
            fullscreen: false,
            control_visible: true,
            selected_control: 0,
        }
    }
}

impl Params {
    pub fn from_cli() -> Self {
        Cli::parse().into_params()
    }

    fn coefficient_from_t(&self, index: usize) -> f64 {
        4.0 * (self.t * [1.03, 1.07, 1.09, 1.13][index]).sin()
    }

    fn coefficient(&self, index: usize) -> f64 {
        if self.coefficient_auto[index] { self.coefficient_from_t(index) } else { self.coefficients[index] }
    }

    pub fn toggle_debug_overlay(&mut self) {
        self.debug_overlay = !self.debug_overlay;
    }

    pub fn move_selection(&mut self, direction: i32) {
        if direction < 0 {
            self.selected_control = self.selected_control.saturating_sub(1);
        } else {
            self.selected_control = (self.selected_control + 1).min(Control::ALL.len() - 1);
        }
    }

    pub fn toggle_selected(&mut self) {
        match Control::ALL[self.selected_control] {
            Control::T => self.paused = !self.paused,
            control => {
                if let Some(index) = control.coefficient_index() {
                    if self.coefficient_auto[index] {
                        self.coefficients[index] = self.coefficient_from_t(index);
                    }
                    self.coefficient_auto[index] = !self.coefficient_auto[index];
                }
            }
        }
    }

    pub fn adjust_selected(&mut self, direction: i32) -> bool {
        let previous_scale = self.scale;
        let delta = direction as f32 * self.step;
        match Control::ALL[self.selected_control] {
            Control::Speed => {
                self.speed = Self::adjust_logarithmic(self.speed, direction, MIN_SPEED_PERCENT, MAX_SPEED_PERCENT);
            }
            Control::Step => {
                self.step = Self::adjust_logarithmic(self.step, direction, MIN_STEP, MAX_STEP);
            }
            Control::T => {
                self.paused = true;
                self.t += delta as f64;
            }
            Control::Scale => self.scale = (self.scale + delta).clamp(1.0, 200.0),
            Control::Brightness => self.brightness = (self.brightness + delta).clamp(0.0, 10000.0),
            Control::Gamma => self.gamma = (self.gamma + delta).clamp(0.01, 10.0),
            Control::Budget => self.budget = (self.budget + delta).clamp(MIN_BUDGET_MS, MAX_BUDGET_MS),
            control => {
                if let Some(index) = control.coefficient_index() {
                    if self.coefficient_auto[index] {
                        self.coefficients[index] = self.coefficient_from_t(index);
                        self.coefficient_auto[index] = false;
                    }
                    self.coefficients[index] += delta as f64;
                }
            }
        }
        self.scale != previous_scale
    }

    fn adjust_logarithmic(value: f32, direction: i32, min: f32, max: f32) -> f32 {
        let current = (value.clamp(min, max).log10() * LOG_STEPS_PER_DECADE).round() as i32;
        let min_step = (min.log10() * LOG_STEPS_PER_DECADE).round() as i32;
        let max_step = (max.log10() * LOG_STEPS_PER_DECADE).round() as i32;
        10.0_f32.powf((current + direction).clamp(min_step, max_step) as f32 / LOG_STEPS_PER_DECADE)
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
        timestamp_res: f32,
        screen_size: (u32, u32),
    ) -> UniformData {
        UniformData {
            a: self.coefficient(0) as f32,
            b: self.coefficient(1) as f32,
            c: self.coefficient(2) as f32,
            d: self.coefficient(3) as f32,
            frame: frame_index as f32,
            texture_size: texture_size as f32,
            brightness: self.brightness * 4e-6,
            budget: self.budget,
            timestamp_res,
            screen_width: screen_size.0 as f32,
            screen_height: screen_size.1 as f32,
            debug_overlay: if self.debug_overlay { 1.0 } else { 0.0 },
            control_overlay: if self.control_visible { 1.0 } else { 0.0 },
            gamma: self.gamma,
        }
    }

    pub fn ctrl_overlay_lines(&self) -> Vec<String> {
        Control::ALL
            .iter()
            .enumerate()
            .map(|(index, control)| {
                let marker = if index == self.selected_control { '>' } else { ' ' };
                let numeric =
                    |name: &str, value: f64, state: &str| format!("{} {:<7} {:>9.3} {:<6}", marker, name, value, state);
                match control {
                    Control::Speed => format!("{} {:<7} {:>9.3}%", marker, "speed", self.speed),
                    Control::Step => numeric("step", self.step as f64, ""),
                    Control::T => numeric("t", self.t, if self.paused { "manual" } else { "auto" }),
                    Control::A | Control::B | Control::C | Control::D => {
                        let coefficient_index = control.coefficient_index().unwrap();
                        numeric(
                            ["a", "b", "c", "d"][coefficient_index],
                            self.coefficient(coefficient_index),
                            if self.coefficient_auto[coefficient_index] { "auto" } else { "manual" },
                        )
                    }
                    Control::Scale => format!("{} {:<7} {:>9.3}%", marker, "scale", self.scale),
                    Control::Brightness => format!("{} {:<10} {:>9.3}%", marker, "brightness", self.brightness),
                    Control::Gamma => numeric("gamma", self.gamma as f64, ""),
                    Control::Budget => format!("{} {:<7} {:>9.3}ms", marker, "budget", self.budget),
                }
            })
            .collect()
    }
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
