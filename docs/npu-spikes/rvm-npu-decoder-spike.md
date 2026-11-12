# RVM NPU：解码器（进行中）

接 alpha spike。功耗对比（overhead spike 末尾）说明整网上 NPU 预计比 780M 每秒省约 2 W，所以开始做解码器。

## 第一步：结构 + 浮点参考 + 量化方案（已完成）

- 解码器的完整数据流见 `npu/rvm/decoder_ref.py` 文件头：LRASPP → decode4（GRU）→ decode3..1（双线性 ×2 + 三路拼接 + 3×3 + GRU）
  → 全分辨率两个 3×3 + 1×1 投影 → DGF 细化。浮点参考连续 20 帧和 onnxruntime 最大差 ~1e-6。
- downsample_ratio = 1 时 DGF 细化不是恒等，和不细化的 alpha 在边缘差到 0.35 → NPU 版也要做。
- 量化方案（`npu/rvm/decoder_q.py`，伪量化）：GRU 状态固定 1/127、sigmoid / tanh 整层查表、上采样整数 3:1 加权、
  DGF 的 box 和乘积在 int32 里算。只看解码器，对浮点 alpha 的边缘带误差 1.05/255、MAE 0.08（编码器自己是 2.4/255）。
  细节和对照表在 skill 的 `references/quant.md`。
- 坑：伪量化里特征不量化（enc='float'）会让权重量化把其他输入通道压成 0，看起来像方案不行（边缘带 27–68），其实是模拟的 bug。

## 要补的 NPU 算子（按顺序）

1. 双线性 ×2 上采样 + 裁剪（新层类型，逐通道组，整数 3:1）。
2. 多路拼接（现在的拼接视图只支持两路；上采样的输出、编码器特征、原图池化写进同一个存储）。
3. sigmoid / tanh 查表激活（重新量化之后，int8 → 查表；sigmoid 输出 uint8）。
4. GRU 逐元素：`r·h`、`h + z·(c − h)`（三个张量的逐元素层）；LRASPP 的逐通道乘（和 SE 类似但乘输出）。
5. 全分辨率的 DGF 细化（box 3×3、逐元素乘减、三个 1×1、最后 `A·X + B`）。可以参考以前的 `rvm_tail` 行内核。
6. 原图池化：先放主机（3 通道，很便宜），以后再搬。

每个算子先写整数逐位的模拟（把 `decoder_q` 的伪量化换成精确整数），再上 NPU 逐字节比；最后接真实权重，
`alpha_check.py` 改成"NPU 整网 vs 浮点整网"验收。
