//! NPU Studio — 背景虚化虚拟摄像头的图形前端
//!
//! 界面 (Slint) ── 启动/停止、参数 ──▶ Python 引擎 (python -m studio --status-json)
//!          ◀── JSON 状态行 / 共享内存预览 ──

mod asr;
mod engine;
mod ocr;
mod preview;
mod settings;
mod system;
mod tray;

use engine::{Engine, Event};
use settings::{Backend, BgMode, Matting, Settings};
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::cell::RefCell;
use std::time::{Duration, Instant};

mod shot;

slint::include_modules!();

/// 功能注册表: 以后新增的 NPU 功能在这里登记一行即可出现在「更多功能」卡片里
struct FeatureDef {
    id: &'static str,
    title: &'static str,
    title_en: &'static str,
    description: &'static str,
    description_en: &'static str,
    tag: &'static str,
    tag_en: &'static str,
    available: bool,
}

const FEATURES: &[FeatureDef] = &[
    FeatureDef {
        id: "noise-suppression",
        title: "通话降噪",
        title_en: "Noise Suppression",
        description: "在 NPU 上实时去除键盘声、风扇声等背景噪音",
        description_en: "Real-time keyboard & fan noise removal on NPU",
        tag: "即将推出",
        tag_en: "Coming Soon",
        available: false,
    },
    FeatureDef {
        id: "auto-framing",
        title: "自动构图",
        title_en: "Auto Framing",
        description: "跟踪人脸位置, 自动裁剪让你保持在画面中央",
        description_en: "Track face position and auto-crop to keep you centered",
        tag: "即将推出",
        tag_en: "Coming Soon",
        available: false,
    },
    FeatureDef {
        id: "presence",
        title: "离开时关闭画面",
        title_en: "Auto Blackout on Away",
        description: "检测到人离开座位后自动黑屏, 回来后恢复",
        description_en: "Blank camera automatically when away and resume on return",
        tag: "即将推出",
        tag_en: "Coming Soon",
        available: false,
    },
];

const RUN_STOPPED: i32 = 0;
const RUN_STARTING: i32 = 1;
const RUN_RUNNING: i32 = 2;
const RUN_STOPPING: i32 = 3;
const RUN_ERROR: i32 = 4;

struct App {
    ui: slint::Weak<AppWindow>,
    settings: Settings,
    cameras: Vec<(String, String)>,
    engine: Option<Engine>,
    asr_engine: Option<asr::AsrEngine>,
    ocr_engine: ocr::OcrEngine,
    next_id: u64,
    restart_pending: bool,
    restart_timer: slint::Timer,
    preview: preview::PreviewReader,
    tray: Option<ksni::blocking::Handle<tray::StudioTray>>,
    last_error: String,
    started_at: Option<Instant>,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
}

fn with_app<R>(f: impl FnOnce(&mut App, &AppWindow) -> R) -> Option<R> {
    APP.with(|a| {
        let mut a = a.borrow_mut();
        let app = a.as_mut()?;
        let ui = app.ui.upgrade()?;
        Some(f(app, &ui))
    })
}

// ------------------------------------------------------------------ 设置 <-> 界面

fn settings_to_ui(app: &App, ui: &AppWindow) {
    let s = &app.settings;
    ui.set_backend(s.backend.index());
    ui.set_matting(s.matting.index());
    ui.set_bg_mode(s.bg_mode.index());
    ui.set_blur_strength(s.blur as f32);
    ui.set_smoothing(s.smoothing);
    ui.set_high_fps(s.high_fps);
    ui.set_bg_image_name(
        s.bg_image
            .as_deref()
            .and_then(|p| std::path::Path::new(p).file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default()
            .into(),
    );
    let names: Vec<SharedString> = app
        .cameras
        .iter()
        .map(|(dev, name)| SharedString::from(format!("{name}  ({dev})")))
        .collect();
    ui.set_cameras(ModelRc::new(VecModel::from(names)));
    let idx = app.cameras.iter().position(|(d, _)| *d == s.camera).unwrap_or(0);
    ui.set_camera_index(idx as i32);
    let is_en = s.lang == "en";
    ui.set_is_en(is_en);
    if ui.get_run_state() == RUN_STOPPED {
        ui.set_status_text(if is_en { "Stopped".into() } else { "未运行".into() });
    }
    if !ui.get_asr_active() {
        ui.set_asr_status(if is_en { "Idle".into() } else { "未开启".into() });
    }
    let feats: Vec<FeatureData> = FEATURES
        .iter()
        .map(|f| FeatureData {
            id: f.id.into(),
            title: (if is_en { f.title_en } else { f.title }).into(),
            description: (if is_en { f.description_en } else { f.description }).into(),
            tag: (if is_en { f.tag_en } else { f.tag }).into(),
            available: f.available,
            enabled: f.available && *s.features.get(f.id).unwrap_or(&false),
        })
        .collect();
    ui.set_features(ModelRc::new(VecModel::from(feats)));
}

fn ui_to_settings(app: &mut App, ui: &AppWindow) {
    let s = &mut app.settings;
    s.backend = Backend::from_index(ui.get_backend());
    s.matting = Matting::from_index(ui.get_matting());
    s.bg_mode = BgMode::from_index(ui.get_bg_mode());
    s.blur = (ui.get_blur_strength().round() as u32) | 1;
    s.smoothing = (ui.get_smoothing() * 100.0).round() / 100.0;
    s.high_fps = ui.get_high_fps();
    if let Some((dev, _)) = app.cameras.get(ui.get_camera_index().max(0) as usize) {
        s.camera = dev.clone();
    }
    s.save();
}

fn backend_ok(s: &Settings) -> Result<(), String> {
    match s.backend {
        Backend::Npu if !system::npu_available() => return Err("没有找到可用的 NPU 环境 (/dev/accel/accel0 或 .venv-iron)".into()),
        Backend::GpuRusticl if !system::rusticl_available() => return Err("没有安装 Mesa OpenCL (opencl-mesa)".into()),
        Backend::GpuRocm if !system::rocm_available() => return Err("没有安装 ROCm OpenCL (rocm-opencl-runtime)".into()),
        _ => {}
    }
    match s.matting {
        Matting::RvmGpu if !system::rvm_gpu_available() => Err(
            "RVM 核显版缺少本机生成的文件: 先运行 studio/native/convert_rvm.py 和 build_rvm_worker.sh".into(),
        ),
        Matting::RvmCpu if !system::rvm_cpu_available() => Err("缺少 models/rvm_mobilenetv3_fp32.onnx".into()),
        _ => Ok(()),
    }
}

// ------------------------------------------------------------------ 引擎控制

fn set_state(app: &App, ui: &AppWindow, state: i32, text: &str) {
    ui.set_run_state(state);
    ui.set_status_text(text.into());
    if let Some(t) = &app.tray {
        let running = state == RUN_RUNNING || state == RUN_STARTING;
        t.update(|tr| tr.running = running);
    }
}

fn start(app: &mut App, ui: &AppWindow) {
    if let Err(e) = backend_ok(&app.settings) {
        app.last_error = e.clone();
        ui.set_error_text(e.into());
        return;
    }
    if system::loopback_device().is_none() {
        let msg = "虚拟摄像头还没有加载: 点下方「加载」(或「开机自动加载」) 后再开启";
        ui.set_error_text(msg.into());
        return;
    }
    if app.settings.bg_mode == BgMode::Image && app.settings.bg_image.is_none() {
        ui.set_error_text("「替换图片」模式需要先选择一张背景图片".into());
        return;
    }
    let _ = std::fs::remove_file(system::preview_path());
    app.preview.reset();
    app.last_error.clear();
    ui.set_error_text("".into());
    ui.set_has_preview(false);

    app.next_id += 1;
    let id = app.next_id;
    let weak = app.ui.clone();
    let on_event = move |id: u64, ev: Event| {
        let weak = weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if weak.upgrade().is_some() {
                handle_event(id, ev);
            }
        });
    };
    match Engine::spawn(id, &app.settings, on_event) {
        Ok(e) => {
            app.engine = Some(e);
            app.started_at = Some(Instant::now());
            let hint = if app.settings.backend == Backend::Npu { "正在启动 (加载 NPU 程序)…" } else { "正在启动…" };
            set_state(app, ui, RUN_STARTING, hint);
        }
        Err(e) => {
            ui.set_error_text(format!("无法启动引擎 {}: {e}", system::python().display()).into());
            set_state(app, ui, RUN_ERROR, "启动失败");
        }
    }
}

fn stop(app: &mut App, ui: &AppWindow) {
    if let Some(e) = &app.engine {
        e.stop();
        set_state(app, ui, RUN_STOPPING, "正在停止…");
    }
}

fn handle_event(id: u64, ev: Event) {
    with_app(|app, ui| {
        if app.engine.as_ref().map(|e| e.id) != Some(id) {
            return; // 旧引擎的残留事件
        }
        match ev {
            Event::Starting => {}
            Event::Ready { backend, device } => {
                let secs = app.started_at.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
                ui.set_backend_name(backend.into());
                if let Some(d) = device {
                    ui.set_vcam_device(d.into());
                }
                set_state(app, ui, RUN_RUNNING, &format!("运行中 · 启动用时 {secs:.1} 秒"));
            }
            Event::Stats(s) => {
                ui.set_fps(s.fps);
                ui.set_seg_ms(s.seg_ms);
                ui.set_fx_ms(s.fx_ms);
                ui.set_prep_ms(s.prep_ms);
                ui.set_accel_ms(s.accel_ms);
                ui.set_cpu_pct(s.cpu_pct);
                ui.set_person_pct(s.person);
                if ui.get_run_state() == RUN_RUNNING {
                    ui.set_status_text(format!("运行中 · {:.1} fps", s.fps).into());
                }
            }
            Event::Error(msg) => {
                app.last_error = msg.clone();
                ui.set_error_text(msg.into());
            }
            Event::Exited { ok, stderr_tail } => {
                app.engine = None;
                ui.set_has_preview(false);
                let _ = std::fs::remove_file(system::preview_path());
                if ok && app.last_error.is_empty() {
                    set_state(app, ui, RUN_STOPPED, "已停止");
                } else {
                    if app.last_error.is_empty() {
                        let msg = if stderr_tail.is_empty() { "引擎意外退出".to_string() } else { stderr_tail };
                        ui.set_error_text(msg.into());
                    }
                    set_state(app, ui, RUN_ERROR, "出错了");
                }
                if app.restart_pending {
                    app.restart_pending = false;
                    start(app, ui);
                }
            }
        }
    });
}

/// 运行中修改设置: 防抖 600ms 后自动重启引擎
fn schedule_restart(app: &mut App) {
    if app.engine.is_none() {
        return;
    }
    app.restart_timer.start(slint::TimerMode::SingleShot, Duration::from_millis(600), || {
        with_app(|app, ui| {
            if app.engine.is_some() {
                app.restart_pending = true;
                stop(app, ui);
            }
        });
    });
}

fn refresh_vcam(ui: &AppWindow) {
    ui.set_vcam_device(system::loopback_device().unwrap_or_default().into());
    ui.set_vcam_autoload(system::autoload_configured());
}

fn quit() {
    with_app(|app, _ui| {
        app.restart_pending = false;
        if let Some(mut a) = app.asr_engine.take() {
            a.stop();
        }
        if let Some(e) = &app.engine {
            e.stop();
            // 等引擎释放摄像头并还原设置 (最多 4 秒)
            let t = Instant::now();
            while e.is_running() && t.elapsed() < Duration::from_secs(4) {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        let _ = std::fs::remove_file(system::preview_path());
        if let Some(t) = app.tray.take() {
            t.shutdown();
        }
    });
    let _ = slint::quit_event_loop();
}

fn run_blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static, done: impl FnOnce(T) + Send + 'static) {
    std::thread::spawn(move || {
        let r = work();
        let _ = slint::invoke_from_event_loop(move || done(r));
    });
}

// ------------------------------------------------------------------ main

fn main() -> Result<(), slint::PlatformError> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--screenshot") {
        let path = args.get(i + 1).map(String::as_str).unwrap_or("npu-studio.ppm");
        if let Err(e) = shot::run(path, args.iter().any(|a| a == "running")) {
            eprintln!("截图失败: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }

    let ui = AppWindow::new()?;
    let settings = Settings::load();
    let cameras = system::cameras();

    ui.set_npu_available(system::npu_available());
    ui.set_rusticl_available(system::rusticl_available());
    ui.set_rocm_available(system::rocm_available());
    ui.set_rvm_gpu_available(system::rvm_gpu_available());
    ui.set_rvm_cpu_available(system::rvm_cpu_available());
    refresh_vcam(&ui);

    let weak = ui.as_weak();
    let tray = tray::spawn(tray::StudioTray {
        running: false,
        on_action: Box::new(move |act| {
            let weak = weak.clone();
            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = weak.upgrade() else { return };
                match act {
                    tray::TrayAction::Show => {
                        let _ = ui.show();
                    }
                    tray::TrayAction::StartStop => ui.invoke_start_stop(),
                    tray::TrayAction::Quit => quit(),
                }
            });
        }),
    });
    let has_tray = tray.is_some();

    let app = App {
        ui: ui.as_weak(),
        settings,
        cameras,
        engine: None,
        asr_engine: None,
        ocr_engine: ocr::OcrEngine::new(ui.as_weak()),
        next_id: 0,
        restart_pending: false,
        restart_timer: slint::Timer::default(),
        preview: preview::PreviewReader::new(system::preview_path()),
        tray,
        last_error: String::new(),
        started_at: None,
    };
    settings_to_ui(&app, &ui);
    APP.with(|a| *a.borrow_mut() = Some(app));

    ui.on_start_stop(|| {
        with_app(|app, ui| {
            app.restart_pending = false;
            if app.engine.is_some() {
                stop(app, ui);
            } else {
                start(app, ui);
            }
        });
    });

    ui.on_settings_changed(|| {
        with_app(|app, ui| {
            ui_to_settings(app, ui);
            schedule_restart(app);
        });
    });

    ui.on_feature_toggled(|i, on| {
        with_app(|app, ui| {
            if let Some(f) = FEATURES.get(i as usize) {
                app.settings.features.insert(f.id.to_string(), on);
                app.settings.save();
                let model = ui.get_features();
                if let Some(mut d) = model.row_data(i as usize) {
                    d.enabled = on;
                    model.set_row_data(i as usize, d);
                }
            }
        });
    });

    ui.on_asr_start_stop(|| {
        with_app(|app, ui| {
            if let Some(mut a) = app.asr_engine.take() {
                a.stop();
                ui.set_asr_active(false);
                ui.set_asr_partial("".into());
                ui.set_asr_status("已停止".into());
            } else {
                let backend = asr::AsrBackend::from_index(ui.get_asr_backend());
                app.asr_engine = Some(asr::AsrEngine::start(ui.as_weak(), backend));
                ui.set_asr_active(true);
            }
        });
    });

    ui.on_asr_clear(|| {
        with_app(|_, ui| {
            ui.set_asr_text("".into());
            ui.set_asr_partial("".into());
            ui.set_asr_bucket("–".into());
            ui.set_asr_ms("–".into());
        });
    });

    ui.on_ocr_init(|| {
        with_app(|app, _| {
            app.ocr_engine.ensure_started();
        });
    });

    ui.on_ocr_snip(|| {
        with_app(|app, ui| {
            let backend = ocr::OcrBackend::from_index(ui.get_ocr_backend());
            app.ocr_engine.snip_and_recognize(backend);
        });
    });

    ui.on_ocr_clear(|| {
        with_app(|_, ui| {
            ui.set_ocr_text("".into());
            ui.set_ocr_det_ms("–".into());
            ui.set_ocr_rec_ms("–".into());
            ui.set_ocr_boxes("–".into());
            ui.set_ocr_has_preview(false);
        });
    });

    ui.on_ocr_copy(|| {
        with_app(|_, ui| {
            let txt = ui.get_ocr_text();
            ocr::copy_to_clipboard(&txt);
        });
    });

    ui.on_toggle_lang(|| {
        with_app(|app, ui| {
            app.settings.lang = if app.settings.lang == "en" { "zh".into() } else { "en".into() };
            app.settings.save();
            settings_to_ui(app, ui);
        });
    });

    ui.on_pick_image(|| {
        run_blocking(
            || {
                rfd::FileDialog::new()
                    .set_title("选择背景图片")
                    .add_filter("图片", &["jpg", "jpeg", "png", "webp", "bmp"])
                    .pick_file()
            },
            |picked| {
                if let Some(p) = picked {
                    with_app(|app, ui| {
                        app.settings.bg_image = Some(p.to_string_lossy().into_owned());
                        app.settings.save();
                        settings_to_ui(app, ui);
                        ui.set_error_text("".into());
                        schedule_restart(app);
                    });
                }
            },
        );
    });

    let vcam_action = |work: fn() -> Result<(), String>| {
        with_app(|_, ui| ui.set_vcam_busy(true));
        run_blocking(work, |r| {
            with_app(|_, ui| {
                ui.set_vcam_busy(false);
                refresh_vcam(ui);
                match r {
                    Ok(()) => ui.set_error_text("".into()),
                    Err(e) if e == "已取消授权" => {}
                    Err(e) => ui.set_error_text(format!("加载虚拟摄像头失败: {e}").into()),
                }
            });
        });
    };
    ui.on_load_vcam(move || vcam_action(system::load_loopback));
    ui.on_unload_vcam(move || vcam_action(system::unload_loopback));
    ui.on_setup_autoload(move || vcam_action(system::setup_autoload));

    // 预览刷新 (约 25 fps 轮询, 引擎写入 15 fps)
    let preview_timer = slint::Timer::default();
    preview_timer.start(slint::TimerMode::Repeated, Duration::from_millis(40), || {
        with_app(|app, ui| {
            if ui.get_run_state() != RUN_RUNNING {
                return;
            }
            if let Some(buf) = app.preview.poll() {
                ui.set_preview(slint::Image::from_rgba8(buf));
                ui.set_has_preview(true);
            }
        });
    });

    // 虚拟摄像头状态 (外部 modprobe 也能反映出来)
    let vcam_timer = slint::Timer::default();
    vcam_timer.start(slint::TimerMode::Repeated, Duration::from_secs(2), || {
        with_app(|_, ui| {
            if !ui.get_vcam_busy() {
                refresh_vcam(ui);
            }
        });
    });

    // 关闭窗口: 运行中且有托盘 → 缩到托盘继续运行; 否则退出
    ui.window().on_close_requested(move || {
        let running = with_app(|app, _| app.engine.is_some()).unwrap_or(false);
        if running && has_tray {
            slint::CloseRequestResponse::HideWindow
        } else {
            quit();
            slint::CloseRequestResponse::HideWindow
        }
    });

    ui.show()?;
    slint::run_event_loop_until_quit()?;
    Ok(())
}
