//! 系统探测: 项目目录、Python 引擎、可用的加速器、摄像头、v4l2loopback

use std::path::{Path, PathBuf};
use std::process::Command;

pub const VCAM_LABEL: &str = "NPU Studio Camera";
const MODULES_LOAD: &str = "/etc/modules-load.d/v4l2loopback.conf";
const MODPROBE_OPTS: &str = "/etc/modprobe.d/v4l2loopback.conf";

/// npu-studio-effects 项目根目录 (引擎所在). 可用 NPU_STUDIO_ROOT 覆盖.
pub fn project_root() -> PathBuf {
    if let Some(p) = std::env::var_os("NPU_STUDIO_ROOT") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// 引擎用的 Python: 有 IRON 环境 (.venv-iron) 就用它, 否则退回 .venv (无 NPU)
pub fn python() -> PathBuf {
    let root = project_root();
    let iron = root.join(".venv-iron/bin/python3");
    if iron.exists() {
        iron
    } else {
        root.join(".venv/bin/python3")
    }
}

pub fn iron_site_packages() -> Option<PathBuf> {
    let lib = project_root().join(".venv-iron/lib");
    std::fs::read_dir(lib).ok()?.flatten().map(|e| e.path().join("site-packages"))
        .find(|p| p.join("mlir_aie").exists())
}

pub fn npu_available() -> bool {
    Path::new("/dev/accel/accel0").exists() && iron_site_packages().is_some()
}

fn opencl_icds() -> Vec<String> {
    std::fs::read_dir("/etc/OpenCL/vendors")
        .map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().to_lowercase()).collect())
        .unwrap_or_default()
}

pub fn rusticl_available() -> bool {
    opencl_icds().iter().any(|n| n.contains("rusticl"))
}

pub fn rocm_available() -> bool {
    opencl_icds().iter().any(|n| n.contains("amdocl"))
}

/// RVM 核显版需要本机生成的 worker 和 ncnn 模型 (studio/native/ 里的两步准备)
pub fn rvm_gpu_available() -> bool {
    let root = project_root();
    root.join("third_party/rvm_worker").exists()
        && root.join("models/rvm_mobilenetv3_fp32.ncnn.param").exists()
        && root.join("models/rvm_mobilenetv3_fp32.ncnn.bin").exists()
}

pub fn rvm_cpu_available() -> bool {
    project_root().join("models/rvm_mobilenetv3_fp32.onnx").exists()
}

fn sysfs_name(dev: &Path) -> String {
    std::fs::read_to_string(dev.join("name")).unwrap_or_default().trim().to_string()
}

/// 真实摄像头: (设备路径, 显示名). 只取每个设备的主节点 (index == 0), 排除虚拟设备.
pub fn cameras() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/sys/class/video4linux") else { return out };
    for e in dir.flatten() {
        let p = e.path();
        let real = std::fs::canonicalize(&p).unwrap_or_default();
        if real.to_string_lossy().contains("/virtual/") {
            continue;
        }
        if std::fs::read_to_string(p.join("index")).map(|s| s.trim() != "0").unwrap_or(false) {
            continue;
        }
        let node = e.file_name().to_string_lossy().to_string();
        out.push((format!("/dev/{node}"), sysfs_name(&p)));
    }
    out.sort_by_key(|(dev, _)| dev.trim_start_matches("/dev/video").parse::<u32>().unwrap_or(u32::MAX));
    out
}

/// 已加载的 v4l2loopback 设备 (第一个)
pub fn loopback_device() -> Option<String> {
    let dir = std::fs::read_dir("/sys/devices/virtual/video4linux").ok()?;
    let mut nodes: Vec<_> = dir.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
    nodes.sort();
    nodes.first().map(|n| format!("/dev/{n}"))
}

pub fn autoload_configured() -> bool {
    Path::new(MODULES_LOAD).exists()
}

fn pkexec(args: &[&str]) -> Result<(), String> {
    let out = Command::new("pkexec").args(args).output().map_err(|e| format!("无法运行 pkexec: {e}"))?;
    match out.status.code() {
        Some(0) => Ok(()),
        Some(126) | Some(127) => Err("已取消授权".into()),
        _ => Err(String::from_utf8_lossy(&out.stderr).trim().to_string()),
    }
}

/// 加载 v4l2loopback (会弹出系统授权窗口)
pub fn load_loopback() -> Result<(), String> {
    pkexec(&[
        "modprobe",
        "v4l2loopback",
        "devices=1",
        "exclusive_caps=1",
        &format!("card_label={VCAM_LABEL}"),
    ])
}

/// 写入开机自动加载配置, 并立即加载一次
pub fn setup_autoload() -> Result<(), String> {
    let script = format!(
        "printf 'v4l2loopback\\n' > {MODULES_LOAD} && \
         printf 'options v4l2loopback devices=1 exclusive_caps=1 card_label=\"{VCAM_LABEL}\"\\n' > {MODPROBE_OPTS} && \
         (lsmod | grep -q '^v4l2loopback' || modprobe v4l2loopback)"
    );
    pkexec(&["sh", "-c", &script])
}

pub fn preview_path() -> PathBuf {
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/dev/shm/npu-studio-{uid}.rgba"))
}
