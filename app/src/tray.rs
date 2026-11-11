//! 系统托盘 (KDE/freedesktop StatusNotifierItem)

use ksni::blocking::TrayMethods;
use ksni::menu::StandardItem;

pub enum TrayAction {
    Show,
    StartStop,
    Quit,
}

pub struct StudioTray {
    pub running: bool,
    pub on_action: Box<dyn Fn(TrayAction) + Send>,
}

impl ksni::Tray for StudioTray {
    fn id(&self) -> String {
        "npu-studio".into()
    }
    fn title(&self) -> String {
        if self.running { "NPU Studio · 运行中" } else { "NPU Studio" }.into()
    }
    fn icon_name(&self) -> String {
        if self.running { "camera-web" } else { "camera-off" }.into()
    }
    fn activate(&mut self, _x: i32, _y: i32) {
        (self.on_action)(TrayAction::Show);
    }
    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        vec![
            StandardItem {
                label: "显示主窗口".into(),
                activate: Box::new(|t: &mut Self| (t.on_action)(TrayAction::Show)),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: if self.running { "停止背景虚化" } else { "开启背景虚化" }.into(),
                activate: Box::new(|t: &mut Self| (t.on_action)(TrayAction::StartStop)),
                ..Default::default()
            }
            .into(),
            ksni::MenuItem::Separator,
            StandardItem {
                label: "退出".into(),
                activate: Box::new(|t: &mut Self| (t.on_action)(TrayAction::Quit)),
                ..Default::default()
            }
            .into(),
        ]
    }
}

pub fn spawn(tray: StudioTray) -> Option<ksni::blocking::Handle<StudioTray>> {
    match tray.spawn() {
        Ok(h) => Some(h),
        Err(e) => {
            eprintln!("[npu-studio] 托盘不可用: {e}");
            None
        }
    }
}
