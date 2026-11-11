//! 离屏截图 (开发用): npu-studio --screenshot out.ppm [running]
//! 用 Slint 软件渲染器画出界面, 不在桌面上弹窗口。

use crate::{AppWindow, FeatureData};
use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
use slint::platform::{Platform, WindowAdapter};
use slint::{ComponentHandle, ModelRc, Rgb8Pixel, Rgba8Pixel, SharedPixelBuffer, SharedString, VecModel};
use std::io::Write;
use std::rc::Rc;

struct ShotPlatform {
    window: Rc<MinimalSoftwareWindow>,
}

impl Platform for ShotPlatform {
    fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, slint::PlatformError> {
        Ok(self.window.clone())
    }
}

pub fn run(path: &str, running: bool) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (1060u32, 760u32);
    let window = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(ShotPlatform { window: window.clone() }))
        .map_err(|e| format!("{e:?}"))?;
    let ui = AppWindow::new()?;
    ui.set_cameras(ModelRc::new(VecModel::from(vec![SharedString::from("USB Camera  (/dev/video0)")])));
    let feats = vec![
        FeatureData { id: "a".into(), title: "通话降噪".into(), description: "在 NPU 上实时去除键盘声、风扇声等背景噪音".into(), tag: "即将推出".into(), available: false, enabled: false },
        FeatureData { id: "b".into(), title: "自动构图".into(), description: "跟踪人脸位置, 自动裁剪让你保持在画面中央".into(), tag: "即将推出".into(), available: false, enabled: false },
    ];
    ui.set_features(ModelRc::new(VecModel::from(feats)));
    ui.set_vcam_device("/dev/video2".into());
    if running {
        ui.set_run_state(2);
        ui.set_status_text("运行中 · 30.1 fps".into());
        ui.set_backend_name("NPU ×4列".into());
        ui.set_fps(30.1);
        ui.set_seg_ms(2.9);
        ui.set_fx_ms(6.1);
        ui.set_accel_ms(1.9);
        ui.set_cpu_pct(35.0);
        ui.set_person_pct(16.0);
        // 合成一张渐变图代替摄像头画面
        let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(640, 360);
        for (i, p) in buf.make_mut_slice().iter_mut().enumerate() {
            let (x, y) = ((i % 640) as u32, (i / 640) as u32);
            *p = Rgba8Pixel { r: (x / 3) as u8, g: (y / 2) as u8, b: 160, a: 255 };
        }
        ui.set_preview(slint::Image::from_rgba8(buf));
        ui.set_has_preview(true);
    } else {
        ui.set_vcam_device("".into());
        ui.set_error_text("虚拟摄像头还没有加载: 点下方「加载」(或「开机自动加载」) 后再开启".into());
    }
    ui.show()?;
    window.set_size(slint::PhysicalSize::new(w, h));
    let mut pixels = vec![Rgb8Pixel::default(); (w * h) as usize];
    window.draw_if_needed(|r| {
        r.render(&mut pixels, w as usize);
    });
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    for p in &pixels {
        f.write_all(&[p.r, p.g, p.b])?;
    }
    Ok(())
}
