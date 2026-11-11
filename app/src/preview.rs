//! 读取引擎写在 /dev/shm 的预览帧 (seqlock 协议, 见 studio/preview.py)

use memmap2::Mmap;
use slint::{Rgba8Pixel, SharedPixelBuffer};
use std::fs::File;
use std::path::PathBuf;

const MAGIC: u32 = 0x5355_504E;
const HEADER: usize = 16;

pub struct PreviewReader {
    path: PathBuf,
    map: Option<Mmap>,
    last_seq: u32,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

impl PreviewReader {
    pub fn new(path: PathBuf) -> Self {
        PreviewReader { path, map: None, last_seq: 0 }
    }

    /// 引擎重启后文件会被重建, 需要重新映射
    pub fn reset(&mut self) {
        self.map = None;
        self.last_seq = 0;
    }

    pub fn poll(&mut self) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
        if self.map.is_none() {
            let f = File::open(&self.path).ok()?;
            let m = unsafe { Mmap::map(&f) }.ok()?;
            if m.len() < HEADER || u32_at(&m, 0) != MAGIC {
                return None;
            }
            self.map = Some(m);
        }
        let m = self.map.as_ref()?;
        let (w, h) = (u32_at(m, 8), u32_at(m, 12));
        let len = (w * h * 4) as usize;
        if m.len() < HEADER + len {
            self.map = None;
            return None;
        }
        for _ in 0..3 {
            let s1 = u32_at(m, 4);
            if s1 % 2 == 1 || s1 == self.last_seq {
                return None; // 正在写, 或没有新帧
            }
            let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w, h);
            buf.make_mut_bytes().copy_from_slice(&m[HEADER..HEADER + len]);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            if u32_at(m, 4) == s1 {
                self.last_seq = s1;
                return Some(buf);
            }
        }
        None
    }
}
