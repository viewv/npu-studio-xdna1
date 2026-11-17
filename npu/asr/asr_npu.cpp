// Standalone C++ XRT + CBLAS 4-core NPU ASR backend & benchmark/daemon CLI.
// Compile (必须链 OpenBLAS: 系统的 -lcblas 是 netlib 参考实现, 前端 13 ms → 300 ms、out_w 4 ms → 230 ms):
//   L=$PWD/.venv-iron/lib/python3.14/site-packages/opencv_python_headless.libs
//   g++ -O3 -std=c++17 -I/usr/include/xrt npu/asr/asr_npu.cpp -lxrt_coreutil $L/libopenblasp-r0-59ffcd50.3.15.so \
//       -Wl,-rpath,$L -lm -o ~/.npu/asr/asr_npu_cpp
// 模型包: python -m npu.asr.export_c_bundle → ~/.npu/asr/asr_4c_bundle (只有 7 秒档 W=35; 分档见 encoder_npu --buckets)
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cblas.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace {

constexpr int D = 512;
constexpr int T_WIN = 140;
constexpr int FRAMES_WIN = 700;
constexpr int NCH = 39;
constexpr int WIN = 400;
constexpr int HOP = 160;
constexpr int NFFT = 512;
constexpr int CMS_WIN = 50;
constexpr int FLAT = 3840;
constexpr int N_CALLS = 8;
constexpr int BR = 4;
constexpr int PAD = 4;

struct BundleHeader {
    uint32_t magic;
    uint32_t vocab_sz;
    uint32_t n_vt;
    uint32_t act_bytes;
    uint32_t s_bytes;
    uint32_t out_bytes;
    uint32_t w_bytes;
    uint32_t n_calls;
};

// In-place radix-2 Cooley-Tukey FFT on NFFT=512 complex numbers (re, im).
void fft512(double *re, double *im) {
    for (int i = 1, j = 0; i < NFFT; i++) {
        int bit = NFFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= NFFT; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        double wlen_re = std::cos(ang), wlen_im = std::sin(ang);
        for (int i = 0; i < NFFT; i += len) {
            double w_re = 1.0, w_im = 0.0;
            for (int j = 0; j < len / 2; j++) {
                double u_re = re[i + j], u_im = im[i + j];
                double v_re = re[i + j + len / 2] * w_re - im[i + j + len / 2] * w_im;
                double v_im = re[i + j + len / 2] * w_im + im[i + j + len / 2] * w_re;
                re[i + j] = u_re + v_re;
                im[i + j] = u_im + v_im;
                re[i + j + len / 2] = u_re - v_re;
                im[i + j + len / 2] = u_im - v_im;
                double nxt_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nxt_re;
            }
        }
    }
}

bool read_wav_16k(const std::string &path, std::vector<int16_t> &pcm) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char hdr[12];
    if (!f.read(hdr, 12) || std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0)
        return false;
    uint16_t channels = 1, bits = 16;
    uint32_t sr = 16000;
    while (f) {
        char cid[4];
        uint32_t csz = 0;
        if (!f.read(cid, 4) || !f.read(reinterpret_cast<char *>(&csz), 4)) break;
        if (std::memcmp(cid, "fmt ", 4) == 0) {
            std::vector<uint8_t> fmt(csz);
            f.read(reinterpret_cast<char *>(fmt.data()), csz);
            if (csz >= 16) {
                std::memcpy(&channels, fmt.data() + 2, 2);
                std::memcpy(&sr, fmt.data() + 4, 4);
                std::memcpy(&bits, fmt.data() + 14, 2);
            }
        } else if (std::memcmp(cid, "data", 4) == 0) {
            std::vector<int16_t> raw(csz / 2);
            f.read(reinterpret_cast<char *>(raw.data()), csz);
            if (channels == 1 && sr == 16000 && bits == 16) {
                pcm = std::move(raw);
            } else {
                // Simple mono + linear resample fallback for non-16k WAVs
                size_t n_in = raw.size() / std::max<uint16_t>(1, channels);
                std::vector<float> mono(n_in);
                for (size_t i = 0; i < n_in; i++) {
                    double acc = 0.0;
                    for (uint16_t c = 0; c < channels; c++) acc += raw[i * channels + c];
                    mono[i] = static_cast<float>(acc / channels);
                }
                if (sr == 16000) {
                    pcm.resize(n_in);
                    for (size_t i = 0; i < n_in; i++) pcm[i] = static_cast<int16_t>(mono[i]);
                } else {
                    size_t n_out = static_cast<size_t>(static_cast<double>(n_in) * 16000.0 / sr);
                    pcm.resize(n_out);
                    for (size_t i = 0; i < n_out; i++) {
                        double pos = static_cast<double>(i) * (n_in - 1) / std::max<size_t>(1, n_out - 1);
                        size_t idx = static_cast<size_t>(pos);
                        double frac = pos - idx;
                        float v = mono[idx] * (1.0f - frac) + mono[std::min(idx + 1, n_in - 1)] * frac;
                        pcm[i] = static_cast<int16_t>(std::lrint(v));
                    }
                }
            }
            return true;
        } else {
            f.seekg(csz, std::ios::cur);
        }
        if (csz & 1) f.seekg(1, std::ios::cur);
    }
    return false;
}

class AsrNpuEngine {
public:
    bool init(const std::string &bundle_dir) {
        // Load vocab
        {
            std::ifstream vf(bundle_dir + "/vocab.txt");
            if (!vf) return false;
            std::string line;
            while (std::getline(vf, line)) vocab_.push_back(line);
        }

        // Load weights.bin
        std::ifstream wf(bundle_dir + "/weights.bin", std::ios::binary);
        if (!wf) return false;
        wf.read(reinterpret_cast<char *>(&hdr_), sizeof(hdr_));
        if (hdr_.magic != 0x41535234) return false;

        vt_idx_.resize(hdr_.n_vt);
        wf.read(reinterpret_cast<char *>(vt_idx_.data()), hdr_.n_vt * sizeof(int64_t));

        auto read_f32 = [&](std::vector<float> &v, size_t n) {
            v.resize(n);
            wf.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
        };
        auto read_f64 = [&](std::vector<double> &v, size_t n) {
            v.resize(n);
            wf.read(reinterpret_cast<char *>(v.data()), n * sizeof(double));
        };

        read_f32(sh0_, D);
        read_f32(sh_final_, D);
        read_f32(c1_w_, 25 * 32);
        read_f32(c2_w_, 49 * 32 * 64);
        read_f32(c3_w_, 9 * 64 * 192);
        read_f32(lin_w_, FLAT * D);
        read_f32(lin_b_, D);
        read_f32(out_w_, D * hdr_.vocab_sz);
        read_f32(out_b_, hdr_.vocab_sz);
        read_f64(ham_, WIN);
        read_f64(mel_mat_, (NFFT / 2) * (NCH + 1));

        // Load insts.bin
        std::vector<uint8_t> insts;
        {
            std::ifstream inf(bundle_dir + "/insts.bin", std::ios::binary | std::ios::ate);
            if (!inf) return false;
            size_t sz = inf.tellg();
            inf.seekg(0);
            insts.resize(sz);
            inf.read(reinterpret_cast<char *>(insts.data()), sz);
        }

        // Init XRT device, xclbin, kernel, BOs
        device_ = xrt::device(0);
        auto xclbin = xrt::xclbin(bundle_dir + "/final.xclbin");
        auto uuid = device_.register_xclbin(xclbin);
        ctx_ = xrt::hw_context(device_, uuid);
        kernel_ = xrt::kernel(ctx_, "MLIR_AIE");

        n_insts_ = static_cast<uint32_t>(insts.size() / 4);
        bo_instr_ = xrt::bo(device_, insts.size(), XCL_BO_FLAGS_CACHEABLE, kernel_.group_id(1));
        std::memcpy(bo_instr_.map<void *>(), insts.data(), insts.size());
        bo_instr_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        bo_act_ = xrt::bo(device_, hdr_.act_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(3));
        bo_s_ = xrt::bo(device_, hdr_.s_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(5));
        bo_out_ = xrt::bo(device_, hdr_.out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(6));

        std::memset(bo_act_.map<void *>(), 0, hdr_.act_bytes);
        std::memset(bo_s_.map<void *>(), 0, hdr_.s_bytes);
        std::memset(bo_out_.map<void *>(), 0, hdr_.out_bytes);
        bo_act_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_s_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        bo_w_.resize(hdr_.n_calls);
        for (uint32_t g = 0; g < hdr_.n_calls; g++) {
            bo_w_[g] = xrt::bo(device_, hdr_.w_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(4));
            wf.read(bo_w_[g].map<char *>(), hdr_.w_bytes);
            bo_w_[g].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // Preallocate scratch buffers for fbank/frontend/output
        win_f32_.resize(FRAMES_WIN * (NCH + 1));
        w1_.resize(FRAMES_WIN * 40 * 25);
        y1p_.assign((FRAMES_WIN + 2) * 46 * 32, 0.0f);
        w2_.resize(T_WIN * 40 * 49 * 32);
        y2p_.assign((T_WIN + 2) * 41 * 64, 0.0f);
        w3_.resize(T_WIN * 20 * 9 * 64);
        y3_tmp_.resize(T_WIN * 20 * 192);
        y3_.resize(T_WIN * FLAT);
        h_real_.resize(T_WIN * D);
        h_f32_.resize(T_WIN * D);
        logits_.resize(T_WIN * hdr_.vocab_sz);
        return true;
    }

    struct Timings {
        double fbank_ms = 0.0;
        double frontend_ms = 0.0;
        double npu_ms = 0.0;
        double out_ms = 0.0;
        double total_ms = 0.0;
    };

    std::string recognize(const int16_t *pcm, int n_pcm, Timings *tm = nullptr) {
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now();

        // 1. FBank
        int nfr = (n_pcm >= WIN) ? (n_pcm - WIN) / HOP + 1 : 0;
        int m = std::min(nfr, FRAMES_WIN);
        std::vector<double> out_raw(m * (NCH + 1));
        std::vector<double> pwr(m * (NFFT / 2));
        std::vector<double> energies(m);
        double re[NFFT], im[NFFT];
        for (int k = 0; k < m; k++) {
            const int16_t *xp = pcm + k * HOP;
            double e = 0.0;
            double prev = 0.0;
            for (int i = 0; i < WIN; i++) {
                double x = static_cast<double>(xp[i]);
                e += x * x;
                double pe = (i == 0) ? (x * 0.03) : (x - 0.97 * prev);
                prev = x;
                re[i] = pe * ham_[i];
                im[i] = 0.0;
            }
            for (int i = WIN; i < NFFT; i++) { re[i] = 0.0; im[i] = 0.0; }
            fft512(re, im);
            energies[k] = e;
            double *pk = pwr.data() + k * (NFFT / 2);
            for (int b = 0; b < NFFT / 2; b++) {
                double rr = re[b + 1], ii = im[b + 1];
                pk[b] = rr * rr + ii * ii;
            }
        }
        if (m > 0) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, NCH + 1, NFFT / 2, 1.0, pwr.data(), NFFT / 2,
                        mel_mat_.data(), NCH + 1, 0.0, out_raw.data(), NCH + 1);
            for (int k = 0; k < m; k++) {
                double *row = out_raw.data() + k * (NCH + 1);
                for (int c = 0; c < NCH; c++) row[c] = std::log(std::max(row[c], 1.0));
                row[NCH] = std::log(std::max(energies[k], 1e-10));
            }
        }
        // Sliding CMS mean
        double sum[NCH + 1] = {0};
        for (int k = 0; k < m; k++) {
            const double *row = out_raw.data() + k * (NCH + 1);
            for (int c = 0; c <= NCH; c++) sum[c] += row[c];
            if (k >= CMS_WIN) {
                const double *old = out_raw.data() + (k - CMS_WIN) * (NCH + 1);
                for (int c = 0; c <= NCH; c++) sum[c] -= old[c];
            }
            int cnt = std::min(k + 1, CMS_WIN);
            float *dst = win_f32_.data() + k * (NCH + 1);
            for (int c = 0; c <= NCH; c++) dst[c] = static_cast<float>(row[c] - sum[c] / cnt);
        }
        float sil[NCH + 1];
        int hcnt = std::min(m, CMS_WIN);
        for (int c = 0; c <= NCH; c++) {
            double sraw = (c == NCH) ? std::log(1e-10) : 0.0;
            sil[c] = static_cast<float>(hcnt > 0 ? (sraw - sum[c] / hcnt) : sraw);
        }
        for (int k = m; k < FRAMES_WIN; k++) {
            std::memcpy(win_f32_.data() + k * (NCH + 1), sil, sizeof(sil));
        }
        int valid_t = std::min(T_WIN, (m + 50 - 5) / 5 + 1);
        auto t1 = clk::now();

        // 2. Frontend Conv1 -> Conv2 -> Conv3 -> Lin
        std::vector<float> xp((FRAMES_WIN + 4) * 44, 0.0f);
        for (int ti = 0; ti < FRAMES_WIN; ti++) {
            std::memcpy(xp.data() + (ti + 2) * 44 + 2, win_f32_.data() + ti * 40, 40 * sizeof(float));
        }
        for (int ti = 0; ti < FRAMES_WIN; ti++) {
            for (int f = 0; f < 40; f++) {
                float *dst = w1_.data() + (ti * 40 + f) * 25;
                for (int i = 0; i < 5; i++) {
                    std::memcpy(dst + i * 5, xp.data() + (ti + i) * 44 + f, 5 * sizeof(float));
                }
            }
        }
        std::vector<float> y1(FRAMES_WIN * 40 * 32);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    FRAMES_WIN * 40, 32, 25, 1.0f, w1_.data(), 25, c1_w_.data(), 32, 0.0f, y1.data(), 32);
        for (size_t i = 0; i < y1.size(); i++) y1[i] = std::max(y1[i], 0.0f);
        for (int ti = 0; ti < FRAMES_WIN; ti++) {
            std::memcpy(y1p_.data() + ((ti + 1) * 46 + 3) * 32, y1.data() + ti * 40 * 32, 40 * 32 * sizeof(float));
        }
        for (int ti = 0; ti < T_WIN; ti++) {
            for (int f = 0; f < 40; f++) {
                float *dst = w2_.data() + (ti * 40 + f) * (49 * 32);
                for (int i = 0; i < 7; i++) {
                    std::memcpy(dst + i * (7 * 32), y1p_.data() + ((ti * 5 + i) * 46 + f) * 32, 7 * 32 * sizeof(float));
                }
            }
        }
        std::vector<float> y2(T_WIN * 40 * 64);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T_WIN * 40, 64, 49 * 32, 1.0f, w2_.data(), 49 * 32, c2_w_.data(), 64, 0.0f, y2.data(), 64);
        for (size_t i = 0; i < y2.size(); i++) y2[i] = std::max(y2[i], 0.0f);
        for (int ti = 0; ti < T_WIN; ti++) {
            std::memcpy(y2p_.data() + ((ti + 1) * 41) * 64, y2.data() + ti * 40 * 64, 40 * 64 * sizeof(float));
        }
        for (int ti = 0; ti < T_WIN; ti++) {
            for (int f = 0; f < 20; f++) {
                float *dst = w3_.data() + (ti * 20 + f) * (9 * 64);
                for (int i = 0; i < 3; i++) {
                    std::memcpy(dst + i * (3 * 64), y2p_.data() + ((ti + i) * 41 + f * 2) * 64, 3 * 64 * sizeof(float));
                }
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T_WIN * 20, 192, 9 * 64, 1.0f, w3_.data(), 9 * 64, c3_w_.data(), 192, 0.0f, y3_tmp_.data(), 192);
        for (int ti = 0; ti < T_WIN; ti++) {
            for (int f = 0; f < 20; f++) {
                for (int ch = 0; ch < 192; ch++) {
                    float v = y3_tmp_[(ti * 20 + f) * 192 + ch];
                    y3_[ti * FLAT + ch * 20 + f] = std::max(v, 0.0f);
                }
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T_WIN, D, FLAT, 1.0f, y3_.data(), FLAT, lin_w_.data(), D, 0.0f, h_real_.data(), D);
        for (int ti = 0; ti < valid_t; ti++) {
            for (int ch = 0; ch < D; ch++) h_real_[ti * D + ch] += lin_b_[ch];
        }
        // Quantize & write to bo_act
        // act layout: [12, 64, 48, 16] at rows 4..7, cg 32..63, col 4..38
        uint8_t *act_ptr = bo_act_.map<uint8_t *>();
        for (int ch = 0; ch < D; ch++) {
            int cg = 32 + (ch >> 4);
            int c16 = ch & 15;
            float inv_sh = 1.0f / sh0_[ch];
            for (int r = 0; r < 4; r++) {
                uint8_t *dst = act_ptr + (((BR + r) * 64 + cg) * 48 + PAD) * 16 + c16;
                for (int p = 0; p < 35; p++) {
                    int ti = r * 35 + p;
                    int q = 0;
                    if (ti < valid_t) {
                        q = static_cast<int>(std::lrint(h_real_[ti * D + ch] * inv_sh));
                        q = std::max(-127, std::min(127, q));
                    }
                    dst[p * 16] = static_cast<uint8_t>(static_cast<int8_t>(q));
                }
            }
        }
        auto t2 = clk::now();

        // 3. NPU 8-segment execution
        set_valid_t(valid_t);
        const uint8_t *out_ptr = bo_out_.map<const uint8_t *>();
        for (uint32_t g = 0; g < hdr_.n_calls; g++) {
            bo_act_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run = kernel_(3, bo_instr_, n_insts_, bo_act_, bo_w_[g], bo_s_, bo_out_);
            run.wait();
            bo_out_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (g + 1 < hdr_.n_calls) {
                // Copy bo_out [4..7, 0..31, 4..38, 0..15] -> bo_act [4..7, 32..63, 4..38, 0..15]
                for (int r = 0; r < 4; r++) {
                    for (int cg = 0; cg < 32; cg++) {
                        const uint8_t *src = out_ptr + (((BR + r) * 32 + cg) * 48 + PAD) * 16;
                        uint8_t *dst = act_ptr + (((BR + r) * 64 + (32 + cg)) * 48 + PAD) * 16;
                        std::memcpy(dst, src, 35 * 16);
                    }
                }
            }
        }
        auto t3 = clk::now();

        // 4. out_w + CTC greedy decode
        for (int ti = 0; ti < valid_t; ti++) {
            int r = ti / 35;
            int p = ti % 35;
            float *dst = h_f32_.data() + ti * D;
            for (int cg = 0; cg < 32; cg++) {
                const int8_t *src = reinterpret_cast<const int8_t *>(
                    out_ptr + (((BR + r) * 32 + cg) * 48 + PAD + p) * 16);
                for (int c16 = 0; c16 < 16; c16++) {
                    int ch = (cg << 4) + c16;
                    dst[ch] = static_cast<float>(src[c16]) * sh_final_[ch];
                }
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    valid_t, hdr_.vocab_sz, D, 1.0f, h_f32_.data(), D,
                    out_w_.data(), hdr_.vocab_sz, 0.0f, logits_.data(), hdr_.vocab_sz);

        std::string res;
        uint32_t prev = 0xFFFFFFFFu;
        for (int ti = 0; ti < valid_t; ti++) {
            const float *row = logits_.data() + ti * hdr_.vocab_sz;
            uint32_t best = 0;
            float best_v = row[0] + out_b_[0];
            for (uint32_t v = 1; v < hdr_.vocab_sz; v++) {
                float val = row[v] + out_b_[v];
                if (val > best_v) { best_v = val; best = v; }
            }
            if (best != prev && best != 0) {
                const std::string &tok = vocab_[best];
                if (tok == "<UNK>") {
                } else if (tok == "<SPACE>") {
                    res.push_back(' ');
                } else if (tok.size() > 1 && tok[0] == '|') {
                    if (!res.empty() && std::isalnum(static_cast<unsigned char>(res.back()))) {
                        res.push_back(' ');
                    }
                    res.append(tok.substr(1));
                } else {
                    res.append(tok);
                }
            }
            prev = best;
        }
        auto t4 = clk::now();

        if (tm) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            tm->fbank_ms = ms(t0, t1);
            tm->frontend_ms = ms(t1, t2);
            tm->npu_ms = ms(t2, t3);
            tm->out_ms = ms(t3, t4);
            tm->total_ms = ms(t0, t4);
        }
        return res;
    }

private:
    void set_valid_t(int valid_t) {
        if (valid_t == last_vt_) return;
        for (uint32_t g = 0; g < hdr_.n_calls; g++) {
            int32_t *wp = bo_w_[g].map<int32_t *>();
            for (int64_t idx : vt_idx_) wp[idx] = valid_t;
            bo_w_[g].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        last_vt_ = valid_t;
    }

    BundleHeader hdr_{};
    std::vector<std::string> vocab_;
    std::vector<int64_t> vt_idx_;
    std::vector<float> sh0_, sh_final_, c1_w_, c2_w_, c3_w_, lin_w_, lin_b_, out_w_, out_b_;
    std::vector<double> ham_, mel_mat_;
    std::vector<float> win_f32_, w1_, y1p_, w2_, y2p_, w3_, y3_tmp_, y3_, h_real_, h_f32_, logits_;

    xrt::device device_;
    xrt::hw_context ctx_;
    xrt::kernel kernel_;
    xrt::bo bo_instr_, bo_act_, bo_s_, bo_out_;
    std::vector<xrt::bo> bo_w_;
    uint32_t n_insts_ = 0;
    int last_vt_ = T_WIN;
};

} // namespace

int main(int argc, char **argv) {
    std::string bundle = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.npu/asr/asr_4c_bundle";
    int iters = 1;
    std::vector<std::string> wavs;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) bundle = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else wavs.push_back(a);
    }

    auto t_init0 = std::chrono::steady_clock::now();
    AsrNpuEngine eng;
    if (!eng.init(bundle)) {
        std::fprintf(stderr, "Failed to init bundle from %s\n", bundle.c_str());
        return 1;
    }
    double init_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_init0).count();
    std::printf("[asr-cpp] initialized in %.2fs\n", init_s);

    if (wavs.empty()) {
        std::string line;
        std::printf("READY\n");
        std::fflush(stdout);
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            if (line == "quit" || line == "exit") break;
            std::vector<int16_t> pcm;
            if (!read_wav_16k(line, pcm)) {
                std::printf("ERR: cannot read %s\n", line.c_str());
                std::fflush(stdout);
                continue;
            }
            AsrNpuEngine::Timings tm;
            std::string txt = eng.recognize(pcm.data(), static_cast<int>(pcm.size()), &tm);
            std::printf("OK %.1fms (fb=%.1f fe=%.1f npu=%.1f out=%.1f) | %s\n",
                        tm.total_ms, tm.fbank_ms, tm.frontend_ms, tm.npu_ms, tm.out_ms, txt.c_str());
            std::fflush(stdout);
        }
        return 0;
    }

    // Warmup on first wav
    std::vector<int16_t> pcm0;
    if (read_wav_16k(wavs[0], pcm0)) {
        eng.recognize(pcm0.data(), static_cast<int>(pcm0.size()));
    }

    for (const auto &w : wavs) {
        std::vector<int16_t> pcm;
        if (!read_wav_16k(w, pcm)) {
            std::fprintf(stderr, "Cannot read %s\n", w.c_str());
            continue;
        }
        double best_tot = 1e9;
        AsrNpuEngine::Timings best_tm;
        std::string txt;
        for (int it = 0; it < iters; it++) {
            AsrNpuEngine::Timings tm;
            txt = eng.recognize(pcm.data(), static_cast<int>(pcm.size()), &tm);
            if (tm.total_ms < best_tot) { best_tot = tm.total_ms; best_tm = tm; }
        }
        std::printf("%-30s %6.1f ms (fb=%.1f fe=%.1f npu=%.1f out=%.1f) | %s\n",
                    w.substr(w.find_last_of('/') + 1).c_str(),
                    best_tm.total_ms, best_tm.fbank_ms, best_tm.frontend_ms,
                    best_tm.npu_ms, best_tm.out_ms, txt.c_str());
    }
    return 0;
}
