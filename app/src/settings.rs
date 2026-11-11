//! 用户设置, 保存在 ~/.config/npu-studio/settings.json

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    Npu,
    GpuRusticl,
    GpuRocm,
    Cpu,
}

impl Backend {
    pub fn from_index(i: i32) -> Self {
        match i {
            1 => Backend::GpuRusticl,
            2 => Backend::GpuRocm,
            3 => Backend::Cpu,
            _ => Backend::Npu,
        }
    }
    pub fn index(self) -> i32 {
        match self {
            Backend::Npu => 0,
            Backend::GpuRusticl => 1,
            Backend::GpuRocm => 2,
            Backend::Cpu => 3,
        }
    }
}

/// 抠像模型
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Matting {
    /// MediaPipe 人像分割: 很轻, 边缘一般
    Mediapipe,
    /// RVM 跑在核显上 (ncnn + Vulkan): 头发丝级边缘, CPU 占用低
    RvmGpu,
    /// RVM 跑在 CPU 上 (ONNX Runtime): 同样的效果, 但要 2 个多核
    RvmCpu,
}

impl Matting {
    pub fn from_index(i: i32) -> Self {
        match i {
            1 => Matting::RvmGpu,
            2 => Matting::RvmCpu,
            _ => Matting::Mediapipe,
        }
    }
    pub fn index(self) -> i32 {
        match self {
            Matting::Mediapipe => 0,
            Matting::RvmGpu => 1,
            Matting::RvmCpu => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BgMode {
    Blur,
    Image,
    Color,
}

impl BgMode {
    pub fn from_index(i: i32) -> Self {
        match i {
            1 => BgMode::Image,
            2 => BgMode::Color,
            _ => BgMode::Blur,
        }
    }
    pub fn index(self) -> i32 {
        match self {
            BgMode::Blur => 0,
            BgMode::Image => 1,
            BgMode::Color => 2,
        }
    }
    pub fn engine_arg(self) -> &'static str {
        match self {
            BgMode::Blur => "blur",
            BgMode::Image => "image",
            BgMode::Color => "color",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub backend: Backend,
    pub matting: Matting,
    pub bg_mode: BgMode,
    pub blur: u32,
    pub smoothing: f32,
    pub bg_image: Option<String>,
    pub camera: String,
    pub high_fps: bool,
    /// 功能开关 (id → 是否开启), 供后续 NPU 功能使用
    pub features: BTreeMap<String, bool>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            backend: Backend::Npu,
            matting: Matting::Mediapipe,
            bg_mode: BgMode::Blur,
            blur: 61,
            smoothing: 0.5,
            bg_image: None,
            camera: "/dev/video0".into(),
            high_fps: false,
            features: BTreeMap::new(),
        }
    }
}

fn path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config"));
    base.join("npu-studio").join("settings.json")
}

impl Settings {
    pub fn load() -> Self {
        std::fs::read_to_string(path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let p = path();
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let tmp = p.with_extension("json.tmp");
            if std::fs::write(&tmp, s).is_ok() {
                let _ = std::fs::rename(tmp, p);
            }
        }
    }
}
