//! Python 引擎 (python -m studio --status-json) 的进程管理

use crate::settings::{Backend, Settings};
use crate::system;
use serde::Deserialize;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Stats {
    #[serde(default)]
    pub fps: f32,
    #[serde(default)]
    pub seg_ms: f32,
    #[serde(default)]
    pub fx_ms: f32,
    #[serde(default)]
    pub prep_ms: f32,
    #[serde(default)]
    pub accel_ms: f32,
    #[serde(default)]
    pub cpu_pct: f32,
    #[serde(default)]
    pub person: f32,
}

#[derive(Debug, Clone)]
pub enum Event {
    Starting,
    Ready { backend: String, device: Option<String> },
    Stats(Stats),
    Error(String),
    /// 进程退出; 带上 stderr 最后几行 (非正常退出时用于展示)
    Exited { ok: bool, stderr_tail: String },
}

pub struct Engine {
    pub id: u64,
    pid: i32,
    exited: Arc<AtomicBool>,
}

fn engine_args(s: &Settings) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "-m".into(),
        "studio".into(),
        "--status-json".into(),
        "--preview-shm".into(),
        system::preview_path().to_string_lossy().into_owned(),
        "--camera".into(),
        s.camera.clone(),
        "--mode".into(),
        s.bg_mode.engine_arg().into(),
        "--blur".into(),
        s.blur.to_string(),
        "--smooth".into(),
        format!("{:.2}", s.smoothing),
    ];
    match s.backend {
        Backend::Npu => a.push("--npu".into()),
        Backend::GpuRusticl => a.extend(["--gpu".into(), "rusticl".into()]),
        Backend::GpuRocm => a.extend(["--gpu".into(), "rocm".into()]),
        Backend::Cpu => {}
    }
    if let (crate::settings::BgMode::Image, Some(img)) = (s.bg_mode, &s.bg_image) {
        a.extend(["--bg".into(), img.clone()]);
    }
    match s.matting {
        crate::settings::Matting::Mediapipe => {}
        crate::settings::Matting::RvmGpu => a.extend(["--matting".into(), "rvm".into(), "--rvm-gpu".into()]),
        crate::settings::Matting::RvmCpu => a.extend(["--matting".into(), "rvm".into()]),
    }
    if s.high_fps {
        a.push("--high-fps".into());
    }
    a
}

/// 与 npu/env.sh 等价的环境变量
fn apply_env(cmd: &mut Command) {
    let root = system::project_root();
    cmd.env("XILINX_XRT", "/usr").env("NPU2", "0");
    if let Some(site) = system::iron_site_packages() {
        let mlir = site.join("mlir_aie");
        let path = format!(
            "{}:{}:{}",
            mlir.join("bin").display(),
            root.join(".venv-iron/bin").display(),
            std::env::var("PATH").unwrap_or_default()
        );
        cmd.env("MLIR_AIE_INSTALL_DIR", &mlir)
            .env("PEANO_INSTALL_DIR", site.join("llvm-aie"))
            .env("PATH", path)
            .env("PYTHONPATH", mlir.join("python"))
            .env("LD_LIBRARY_PATH", mlir.join("lib"));
    }
}

impl Engine {
    pub fn spawn(
        id: u64,
        settings: &Settings,
        on_event: impl Fn(u64, Event) + Send + Sync + 'static,
    ) -> std::io::Result<Engine> {
        let mut cmd = Command::new(system::python());
        cmd.args(engine_args(settings))
            .current_dir(system::project_root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_env(&mut cmd);
        let mut child: Child = cmd.spawn()?;
        let pid = child.id() as i32;
        let exited = Arc::new(AtomicBool::new(false));
        let on_event = Arc::new(on_event);

        // stderr: 只保留最后 30 行
        let tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
        if let Some(err) = child.stderr.take() {
            let tail = tail.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    let mut t = tail.lock().unwrap();
                    t.push_back(line);
                    if t.len() > 30 {
                        t.pop_front();
                    }
                }
            });
        }

        // stdout: JSON 事件; 读完后等待进程退出
        let stdout = child.stdout.take();
        let ex = exited.clone();
        let ev = on_event.clone();
        std::thread::spawn(move || {
            if let Some(out) = stdout {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    if let Some(e) = parse(&line) {
                        ev(id, e);
                    }
                }
            }
            let ok = child.wait().map(|s| s.success() || s.code() == Some(130)).unwrap_or(false);
            ex.store(true, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(50)); // 让 stderr 线程读完
            let t = tail.lock().unwrap();
            let skip = t.len().saturating_sub(6);
            let stderr_tail = t.iter().skip(skip).cloned().collect::<Vec<_>>().join("\n");
            ev(id, Event::Exited { ok, stderr_tail });
        });

        Ok(Engine { id, pid, exited })
    }

    /// SIGTERM 让引擎正常收尾 (释放摄像头、还原设置); 5 秒没退出就 SIGKILL
    pub fn stop(&self) {
        if self.exited.load(Ordering::SeqCst) {
            return;
        }
        unsafe { libc::kill(self.pid, libc::SIGTERM) };
        let (pid, exited) = (self.pid, self.exited.clone());
        std::thread::spawn(move || {
            for _ in 0..50 {
                if exited.load(Ordering::SeqCst) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            unsafe { libc::kill(pid, libc::SIGKILL) };
        });
    }

    pub fn is_running(&self) -> bool {
        !self.exited.load(Ordering::SeqCst)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop();
    }
}

fn parse(line: &str) -> Option<Event> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    match v.get("event")?.as_str()? {
        "starting" => Some(Event::Starting),
        "ready" => {
            let backend = v.get("backend").and_then(|b| b.as_str()).unwrap_or("");
            let matting = v.get("matting").and_then(|m| m.as_str()).unwrap_or("");
            Some(Event::Ready {
                backend: if matting.is_empty() { backend.to_string() } else { format!("抠像 {matting} · 合成 {backend}") },
                device: v.get("device").and_then(|d| d.as_str()).map(str::to_string),
            })
        }
        "stats" => serde_json::from_value(v).ok().map(Event::Stats),
        "error" => Some(Event::Error(
            v.get("message").and_then(|m| m.as_str()).unwrap_or("未知错误").to_string(),
        )),
        _ => None,
    }
}
