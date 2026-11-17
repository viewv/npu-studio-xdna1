//! 实时语音转写后台线程：parecord/arecord 16kHz PCM → NPU (C++ 四档) / CPU (Python) → Slint UI 流式上屏

use crate::AppWindow;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AsrBackend {
    NpuCpp = 0,
    Cpu = 1,
}

impl AsrBackend {
    pub fn from_index(i: i32) -> Self {
        if i == 1 { Self::Cpu } else { Self::NpuCpp }
    }
}

pub struct AsrEngine {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl AsrEngine {
    pub fn start(ui: slint::Weak<AppWindow>, backend: AsrBackend) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = Arc::clone(&stop);
        let handle = thread::spawn(move || run_loop(ui, backend, stop_bg));
        Self {
            stop,
            handle: Some(handle),
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for AsrEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

fn cpp_bin_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".npu/asr/asr_npu_cpp")
}

fn repo_root() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        for p in exe.ancestors() {
            if p.join("npu/asr/shell_asr.py").exists() {
                return p.to_path_buf();
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn set_ui_status(ui: &slint::Weak<AppWindow>, status: &str) {
    let weak = ui.clone();
    let s = status.to_string();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_asr_status(s.into());
        }
    });
}

fn set_ui_partial(ui: &slint::Weak<AppWindow>, partial: &str, status: &str, bucket: &str, ms: &str) {
    let weak = ui.clone();
    let p = partial.to_string();
    let s = status.to_string();
    let b = bucket.to_string();
    let m = ms.to_string();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_asr_partial(p.into());
            w.set_asr_status(s.into());
            if !b.is_empty() {
                w.set_asr_bucket(b.into());
            }
            if !m.is_empty() {
                w.set_asr_ms(m.into());
            }
        }
    });
}

fn commit_ui_text(ui: &slint::Weak<AppWindow>, chunk: &str, status: &str, bucket: &str, ms: &str) {
    let weak = ui.clone();
    let c = chunk.to_string();
    let s = status.to_string();
    let b = bucket.to_string();
    let m = ms.to_string();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            let mut cur = w.get_asr_text().to_string();
            if !c.is_empty() {
                if !cur.is_empty() {
                    cur.push('\n');
                }
                cur.push_str(&c);
                w.set_asr_text(cur.into());
            }
            w.set_asr_partial("".into());
            w.set_asr_status(s.into());
            if !b.is_empty() {
                w.set_asr_bucket(b.into());
            }
            if !m.is_empty() {
                w.set_asr_ms(m.into());
            }
        }
    });
}

fn spawn_recorder() -> Option<Child> {
    if let Ok(c) = Command::new("parecord")
        .args(["--raw", "--rate=16000", "--channels=1", "--format=s16le"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        return Some(c);
    }
    Command::new("arecord")
        .args(["-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()
}

fn spawn_backend(backend: AsrBackend) -> Result<Child, String> {
    match backend {
        AsrBackend::NpuCpp => {
            let bin = cpp_bin_path();
            if !bin.exists() {
                return Err("未找到 ~/.npu/asr/asr_npu_cpp".into());
            }
            Command::new(&bin)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("启动失败: {e}"))
        }
        AsrBackend::Cpu => {
            let root = repo_root();
            let py = root.join(".venv-iron/bin/python");
            Command::new(&py)
                .current_dir(&root)
                .args(["-m", "npu.asr.shell_asr", "--cpu-daemon"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("CPU 后端启动失败: {e}"))
        }
    }
}

fn infer_pcm(
    stdin: &mut std::process::ChildStdin,
    stdout: &mut BufReader<std::process::ChildStdout>,
    samples: &[i16],
) -> Option<(String, String, String)> {
    let hdr = format!("PCM {}\n", samples.len());
    stdin.write_all(hdr.as_bytes()).ok()?;
    let bytes = unsafe {
        std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 2)
    };
    stdin.write_all(bytes).ok()?;
    stdin.flush().ok()?;

    let mut resp = String::new();
    stdout.read_line(&mut resp).ok()?;
    let resp = resp.trim();
    if let Some((meta, txt)) = resp.split_once(" | ") {
        let parts: Vec<&str> = meta.split_whitespace().collect();
        let bucket = parts.get(1).copied().unwrap_or("").to_string();
        let ms = parts.get(2).copied().unwrap_or("").trim_end_matches("ms").to_string();
        Some((txt.trim().to_string(), bucket, ms))
    } else {
        None
    }
}

fn chunk_rms(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum / samples.len() as f64).sqrt()
}

fn run_loop(ui: slint::Weak<AppWindow>, backend: AsrBackend, stop: Arc<AtomicBool>) {
    set_ui_status(&ui, if backend == AsrBackend::NpuCpp { "正在加载 NPU 模型..." } else { "正在加载 CPU 模型..." });

    let mut proc = match spawn_backend(backend) {
        Ok(c) => c,
        Err(msg) => {
            set_ui_status(&ui, &msg);
            return;
        }
    };

    let mut proc_in = proc.stdin.take().unwrap();
    let mut proc_out = BufReader::new(proc.stdout.take().unwrap());

    loop {
        if stop.load(Ordering::Relaxed) {
            let _ = proc_in.write_all(b"quit\n");
            let _ = proc.wait();
            return;
        }
        let mut line = String::new();
        if proc_out.read_line(&mut line).unwrap_or(0) == 0 {
            set_ui_status(&ui, "语音后端意外退出");
            return;
        }
        if line.trim() == "READY" {
            break;
        }
    }

    let mut rec = match spawn_recorder() {
        Some(r) => r,
        None => {
            set_ui_status(&ui, "未找到 parecord / arecord");
            let _ = proc_in.write_all(b"quit\n");
            let _ = proc.wait();
            return;
        }
    };
    let mut rec_out = rec.stdout.take().unwrap();
    let ready_lbl = if backend == AsrBackend::NpuCpp { "监听中 (NPU 四档)" } else { "监听中 (CPU)" };
    set_ui_status(&ui, ready_lbl);

    const CHUNK: usize = 1600; // 100ms @ 16kHz
    const MAX_UTT: usize = 16000 * 68 / 10; // 6.8s
    const SPEECH_RMS: f64 = 240.0;

    let mut raw = vec![0u8; CHUNK * 2];
    let mut samples = vec![0i16; CHUNK];
    let mut preroll: Vec<i16> = Vec::with_capacity(CHUNK * 3);
    let mut utt: Vec<i16> = Vec::with_capacity(MAX_UTT);
    let mut in_speech = false;
    let mut sil_chunks = 0usize;
    let mut since_infer = 0usize;
    let infer_every = if backend == AsrBackend::NpuCpp { 4 } else { 8 };

    while !stop.load(Ordering::Relaxed) {
        if rec_out.read_exact(&mut raw).is_err() {
            break;
        }
        for (i, ch) in raw.chunks_exact(2).enumerate() {
            samples[i] = i16::from_le_bytes([ch[0], ch[1]]);
        }
        let rms = chunk_rms(&samples);

        if !in_speech {
            if rms > SPEECH_RMS {
                in_speech = true;
                sil_chunks = 0;
                since_infer = 0;
                utt.clear();
                utt.extend_from_slice(&preroll);
                utt.extend_from_slice(&samples);
            } else {
                if preroll.len() >= CHUNK * 2 {
                    preroll.drain(..CHUNK);
                }
                preroll.extend_from_slice(&samples);
            }
            continue;
        }

        utt.extend_from_slice(&samples);
        since_infer += 1;
        if rms > SPEECH_RMS {
            sil_chunks = 0;
        } else {
            sil_chunks += 1;
        }

        let commit = sil_chunks >= 7 || utt.len() >= MAX_UTT;
        if commit {
            if utt.len() >= 3200 {
                if let Some((txt, bucket, ms)) = infer_pcm(&mut proc_in, &mut proc_out, &utt) {
                    commit_ui_text(&ui, &txt, ready_lbl, &bucket, &ms);
                }
            } else {
                set_ui_partial(&ui, "", ready_lbl, "", "");
            }
            utt.clear();
            preroll.clear();
            in_speech = false;
            sil_chunks = 0;
            since_infer = 0;
        } else if since_infer >= infer_every && utt.len() >= 6400 {
            since_infer = 0;
            if let Some((txt, bucket, ms)) = infer_pcm(&mut proc_in, &mut proc_out, &utt) {
                set_ui_partial(&ui, &txt, "识别中...", &bucket, &ms);
            }
        }
    }

    let _ = rec.kill();
    let _ = rec.wait();
    let _ = proc_in.write_all(b"quit\n");
    let _ = proc_in.flush();
    let _ = proc.wait();
    set_ui_status(&ui, "已停止");
}
