#!/usr/bin/env bash
# 编译 NPU Studio 并安装到当前用户的应用菜单 (不需要 root)
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release

BIN="$PWD/target/release/npu-studio"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$ICON_DIR" "$APP_DIR"
cp assets/npu-studio.svg "$ICON_DIR/npu-studio.svg"

cat > "$APP_DIR/npu-studio.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=NPU Studio
GenericName=背景虚化虚拟摄像头
Comment=用 Ryzen AI NPU 给视频会议加背景虚化
Exec=$BIN
Icon=npu-studio
Terminal=false
Categories=AudioVideo;Video;
Keywords=camera;webcam;blur;npu;背景虚化;摄像头;
StartupWMClass=npu-studio
EOF

update-desktop-database "$APP_DIR" 2>/dev/null || true
echo "已安装: $APP_DIR/npu-studio.desktop"
echo "在应用菜单里搜索「NPU Studio」即可打开"
