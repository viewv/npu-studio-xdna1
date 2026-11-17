//! 屏幕框选 OCR 后台模块：region snip (spectacle / grim+slurp) → 常驻 npu.ocr.worker → Slint UI

use crate::AppWindow;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OcrBackend {
    Npu = 0,
    Cpu = 1,
}

impl OcrBackend {
    pub fn from_index(i: i32) -> Self {
        if i == 1 { Self::Cpu } else { Self::Npu }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Npu => "npu",
            Self::Cpu => "cpu",
        }
    }
}

pub struct OcrEngine {
    ui: slint::Weak<AppWindow>,
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    ready: Arc<Mutex<bool>>,
    busy: Arc<Mutex<bool>>,
}

impl OcrEngine {
    pub fn new(ui: slint::Weak<AppWindow>) -> Self {
        Self {
            ui,
            child: Arc::new(Mutex::new(None)),
            stdin: Arc::new(Mutex::new(None)),
            ready: Arc::new(Mutex::new(false)),
            busy: Arc::new(Mutex::new(false)),
        }
    }

    pub fn ensure_started(&self) {
        {
            let guard = self.child.lock().unwrap();
            if guard.is_some() {
                return;
            }
        }
        let root = repo_root();
        set_status(&self.ui, "Loading…", true);
        let cmd_str = format!(
            "cd {:?} && source npu/env.sh && exec .venv-iron/bin/python -u -m npu.ocr.worker",
            root
        );
        let mut child = match Command::new("bash")
            .args(["-lc", &cmd_str])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                set_status(&self.ui, &format!("Error: {e}"), false);
                return;
            }
        };

        let child_stdin = child.stdin.take();
        let stdout = child.stdout.take();
        *self.stdin.lock().unwrap() = child_stdin;
        *self.child.lock().unwrap() = Some(child);

        if let Some(out) = stdout {
            let ui = self.ui.clone();
            let ready = Arc::clone(&self.ready);
            let busy = Arc::clone(&self.busy);
            thread::spawn(move || {
                let reader = BufReader::new(out);
                for line in reader.lines().map_while(Result::ok) {
                    let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
                        continue;
                    };
                    match v.get("type").and_then(|t| t.as_str()) {
                        Some("status") => {
                            set_status(&ui, "Loading…", true);
                        }
                        Some("ready") => {
                            *ready.lock().unwrap() = true;
                            set_status(&ui, "Ready", false);
                        }
                        Some("result") => {
                            *busy.lock().unwrap() = false;
                            let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string();
                            let boxes = v.get("boxes").and_then(|x| x.as_i64()).unwrap_or(0);
                            let det_ms = v.get("det_ms").and_then(|x| x.as_f64()).unwrap_or(0.0);
                            let rec_ms = v.get("rec_ms").and_then(|x| x.as_f64()).unwrap_or(0.0);
                            let weak = ui.clone();
                            let _ = slint::invoke_from_event_loop(move || {
                                if let Some(w) = weak.upgrade() {
                                    w.set_ocr_text(text.into());
                                    w.set_ocr_boxes(format!("{boxes}").into());
                                    w.set_ocr_det_ms(format!("{det_ms:.1}").into());
                                    w.set_ocr_rec_ms(format!("{rec_ms:.1}").into());
                                    w.set_ocr_busy(false);
                                    w.set_ocr_status("Ready".into());
                                }
                            });
                        }
                        Some("error") => {
                            *busy.lock().unwrap() = false;
                            let msg = v.get("message").and_then(|x| x.as_str()).unwrap_or("Error").to_string();
                            set_status(&ui, &format!("Error: {msg}"), false);
                        }
                        _ => {}
                    }
                }
            });
        }
    }

    pub fn snip_and_recognize(&self, backend: OcrBackend) {
        self.ensure_started();
        {
            let mut b = self.busy.lock().unwrap();
            if *b {
                return;
            }
            *b = true;
        }
        set_status(&self.ui, "Snipping…", true);

        let ui = self.ui.clone();
        let stdin = Arc::clone(&self.stdin);
        let ready = Arc::clone(&self.ready);
        let busy = Arc::clone(&self.busy);

        thread::spawn(move || {
            let tmp = std::env::temp_dir().join("npu_studio_ocr_snip.png");
            let _ = std::fs::remove_file(&tmp);

            let captured = capture_region(&tmp);
            if !captured || !tmp.exists() {
                *busy.lock().unwrap() = false;
                let is_ready = *ready.lock().unwrap();
                set_status(&ui, if is_ready { "Ready" } else { "Loading…" }, !is_ready);
                return;
            }

            // 更新缩略图预览
            let preview_path = tmp.clone();
            let weak_img = ui.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(w) = weak_img.upgrade() {
                    if let Ok(img) = slint::Image::load_from_path(&preview_path) {
                        w.set_ocr_preview(img);
                        w.set_ocr_has_preview(true);
                    }
                }
            });

            // 等待 worker 就绪 (最多 30s)
            set_status(&ui, "Recognizing…", true);
            for _ in 0..300 {
                if *ready.lock().unwrap() {
                    break;
                }
                thread::sleep(std::time::Duration::from_millis(100));
            }

            let req = serde_json::json!({
                "cmd": "ocr",
                "path": tmp.to_string_lossy(),
                "backend": backend.as_str(),
            });
            let mut sent = false;
            if let Some(ref mut w) = *stdin.lock().unwrap() {
                if writeln!(w, "{}", req).is_ok() && w.flush().is_ok() {
                    sent = true;
                }
            }
            if !sent {
                *busy.lock().unwrap() = false;
                set_status(&ui, "Worker offline", false);
            }
        });
    }
}

impl Drop for OcrEngine {
    fn drop(&mut self) {
        drop(self.stdin.lock().unwrap().take());
        if let Some(mut c) = self.child.lock().unwrap().take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn capture_region(out: &PathBuf) -> bool {
    let out_s = out.to_string_lossy().to_string();
    // 1. KDE Plasma Wayland/X11: spectacle -r -b -n -o <file>
    if let Ok(st) = Command::new("spectacle")
        .args(["-r", "-b", "-n", "-o", &out_s])
        .status()
    {
        if st.success() && out.exists() {
            return true;
        }
    }
    // 2. wlroots Wayland: grim -g "$(slurp)" <file>
    if let Ok(st) = Command::new("bash")
        .args(["-c", &format!("grim -g \"$(slurp)\" {:?}", out_s)])
        .status()
    {
        if st.success() && out.exists() {
            return true;
        }
    }
    // 3. GNOME: gnome-screenshot -a -f <file>
    if let Ok(st) = Command::new("gnome-screenshot")
        .args(["-a", "-f", &out_s])
        .status()
    {
        if st.success() && out.exists() {
            return true;
        }
    }
    false
}

pub fn copy_to_clipboard(text: &str) {
    if text.is_empty() {
        return;
    }
    if let Ok(mut c) = Command::new("wl-copy").stdin(Stdio::piped()).spawn() {
        if let Some(mut s) = c.stdin.take() {
            let _ = s.write_all(text.as_bytes());
        }
        let _ = c.wait();
        return;
    }
    if let Ok(mut c) = Command::new("xclip")
        .args(["-selection", "clipboard"])
        .stdin(Stdio::piped())
        .spawn()
    {
        if let Some(mut s) = c.stdin.take() {
            let _ = s.write_all(text.as_bytes());
        }
        let _ = c.wait();
    }
}

fn repo_root() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        for p in exe.ancestors() {
            if p.join("npu/ocr/worker.py").exists() {
                return p.to_path_buf();
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn set_status(ui: &slint::Weak<AppWindow>, status: &str, busy: bool) {
    let weak = ui.clone();
    let s = status.to_string();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(w) = weak.upgrade() {
            w.set_ocr_status(s.into());
            w.set_ocr_busy(busy);
        }
    });
}
