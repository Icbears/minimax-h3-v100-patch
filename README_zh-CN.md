# MiniMax H3 V100 混合精度 Custom Node v0.1.2

[English](README.md) | 简体中文

从 **0.1.2** 开始，本项目将经过 V100 实测的 MiniMax H3 混合精度方案改为 ComfyUI Custom Node 运行时加载。这是现在推荐的安装方式：不再改写 `comfy/ldm/minimax/model.py`，不需要修改启动命令，删除一个文件夹并重启即可卸载。

新版 Custom Node 已适配 **ComfyUI 0.31.1** 更新后的 MiniMax H3 audio carry/sampler 结构；用户已在 V100 上完成实际运行测试并确认工作正常。

此前会修改源码的安装器仍完整保存在 [`legacy_patcher/`](legacy_patcher/) 中，用于恢复、开发，以及确实需要源码补丁的场景。

## 推荐：使用 Custom Node

### 安装前

请确保 ComfyUI 的 `comfy/ldm/minimax/model.py` 是干净、未打补丁的版本。如果以前使用过本项目旧版 patcher，请先关闭 ComfyUI，再运行对应的恢复脚本：

```bat
legacy_patcher\restore_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
legacy_patcher\restore_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
```

为避免加载顺序决定精度行为，运行时扩展会主动拒绝叠加在旧源码补丁或其他 H3 monkey-patch 之上。

### 安装方法

1. 关闭 ComfyUI。
2. 下载或克隆本仓库。
3. 将整个仓库文件夹放入 `ComfyUI/custom_nodes/`。关键目录结构如下：

   ```text
   ComfyUI/
   └─ custom_nodes/
      └─ minimax-h3-v100-patch/
         ├─ __init__.py
         ├─ runtime_patch.py
         ├─ README_zh-CN.md
         └─ legacy_patcher/
   ```

4. 按原来的方式正常启动 ComfyUI。不要为本方案添加 `--fp16-unet`、`--force-fp16` 或其他全局精度参数。
5. 在启动日志中确认出现：

   ```text
   [MiniMax H3 V100] v0.1.2 runtime profile installed: FP32 base + FP16 QKV/text-video attention + FP32 audio attention
   [MiniMax H3 V100] no --fp16-unet flag is required; source files remain untouched
   ```

第一次加载 H3 模型时，日志还会显示已启用的主 DiT Block 数量。本扩展不会在工作流界面增加节点，现有 MiniMax H3 工作流照常使用。

### 更新与卸载

- 更新：关闭 ComfyUI，用新版本替换本仓库文件夹，然后重启。
- 卸载：从 `custom_nodes` 删除本仓库文件夹并重启。如果只使用过 Custom Node 模式，不需要恢复任何 ComfyUI 源码备份。

## 精度分配

50 个主要 DiT Block 继续采用已经验证的“FP32 海洋 + FP16 热点岛屿”方案：

- FP16：常驻的融合 QKV 权重、融合 QKV 投影，以及文本/视频查询的 Attention。
- FP32：目标/参考音频查询的 Attention、Q/K RMSNorm、RoPE、Attention 输出投影、MLP、AdaLN、残差累加和最终输出头。
- 保持源码计算路径：两个 Token Refiner Block。

音频 FP32 Attention 使用的 Q/K/V 数值来自 FP16 QKV 投影，但音频 Attention 运算本身使用 FP32。本方案只在推理期间、主 Block 输入为 CUDA FP32 时启用；CPU、BF16、全局 FP16 和训练路径均调用源码原始实现。

## 运行时安全机制

- 加载前检查当前 `Attention`、`DiTBlock`、`MLP`、`MiniMaxH3Model` 和 `PackedLayout` 结构。
- 检测到旧的源码级 V100 patch 或其他扩展已经接管关键 H3 方法时拒绝叠加。
- 如果其他扩展在本项目之后加载，也会再次检查，并让新模型实例保持未修改状态。
- 重复加载具有幂等性。
- 音频行信息缺失或无效时，该次调用安全回退到完整 FP32 Attention。
- QKV 权重不是 FP32/FP16 时不会强制转换，对应 Block 返回源码路径。
- 不会打开、替换、重命名或写入任何 ComfyUI 源码文件。

当前版本面向包含新版 audio carry/sampler 和 `PackedLayout.segments` 的 ComfyUI 0.31.1 H3 结构。如果未来 ComfyUI 重构内部实现，本扩展会优先自动禁用，而不是执行不完整的危险补丁。

## 性能依据

早期 V100 测试用例：0.2 百万像素、5 秒、24 fps。环境：Windows 10、NVIDIA Tesla V100 32 GB、150 W 功率限制；基线已经包含 TE-Speed。

| 方案 | 耗时 | 相对基线变化 | 结果 |
|---|---:|---:|---|
| TE-Speed 基线 | 317 秒 | — | 成功 |
| **MiniMax H3 V100 混合精度方案** | **206 秒** | **加速 35.0% / 1.54 倍** | **成功** |

这些数字用于说明早期核心精度方案，不代表所有 ComfyUI 工作流都能获得相同耗时。在后续音频安全版本中，相比推理期间反复转换 QKV 权重，QKV 权重常驻 FP16 额外获得了约 **5%–10%** 的实测提升。实际结果取决于工作负载、Attention 后端、卸载方式、功率限制和软件构建。

第一次 A/B 测试应固定随机种子、提示词、模型、采样器、步数、分辨率、帧数和 Attention 后端，并关闭无关的 TeaCache、SageAttention、编译、全局 FP16 和 FP16 accumulation 改动。丢弃一次预热并至少记录三次正式运行；遇到第一处黑屏、NaN、音频异常或报错时立即停止继续扩大精度边界。

## Legacy 源码 patcher

0.1.2 之前会修改源码的整套工具已收纳到 [`legacy_patcher/`](legacy_patcher/) 中，其中包括：

- 带安全检查的 origin/TE-Speed BAT 和 Python 安装器；
- 带验证的恢复脚本及精确相邻备份机制；
- origin/TE 结构识别、AST 检查、幂等性和 SHA-256 校验；
- 用于开发和手动移植的完整 V100 源码快照；
- 独立的 manifest 和原始验证来源说明。

普通用户应优先使用 Custom Node。只有在明确需要修改 `model.py`、安装 TE `block_loop` 源码钩子、恢复旧补丁，或检查完整源码快照时，才建议使用 legacy patcher。

Windows 入口：

```bat
legacy_patcher\patch_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
legacy_patcher\patch_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
legacy_patcher\restore_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
legacy_patcher\restore_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
```

Python 入口：

```text
python legacy_patcher/patch_te_v100.py /path/to/ComfyUI
python legacy_patcher/patch_h3_origin_v100.py /path/to/ComfyUI
```

完整开发源码：

- [`legacy_patcher/sources/origin_v100/model.py`](legacy_patcher/sources/origin_v100/model.py)：官方/origin Block Loop 加音频安全 V100 方案。
- [`legacy_patcher/sources/te_v100/model.py`](legacy_patcher/sources/te_v100/model.py)：TE-Speed Block Loop 钩子加相同精度方案。

受支持的干净 origin 源码基于包含 ComfyUI audio 修复 [`93cb5edb`](https://github.com/Comfy-Org/ComfyUI/commit/93cb5edb) 的实现，其 SHA-256 为 `1c9828ec3d38ac01398e45b1edf8d7db38fcc8148c5eb3ba8fb92b762147d0ce`。旧 patcher 的验证证据见 [`legacy_patcher/NOTICE.md`](legacy_patcher/NOTICE.md)，0.1.2 运行时边界见 [`NOTICE.md`](NOTICE.md)。

## 致谢

特别感谢 [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix)。这个项目展示了通过 ComfyUI Custom Node 干净地交付 MiniMax H3 加速的方法，并启发本项目在 0.1.2 中从持久修改源码转向运行时加载。

本项目继续采用自己此前独立测试的“**FP32 海洋 + FP16 热点岛屿**”精度策略；这里的致谢主要针对 Custom Node 的交付思路。同时感谢 ComfyUI、MiniMax 和 TE-Speed 贡献者提供的基础生态。

## 项目范围与许可

这是社区制作的混合精度扩展，不是 MiniMax、TE-Speed、ComfyUI 或上述致谢项目的官方版本。许可证 SPDX 标识符为 `GPL-3.0-only`，完整条款见 [`LICENSE`](LICENSE)；致谢项目使用其作者单独声明的许可证。
