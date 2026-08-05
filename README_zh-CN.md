# MiniMax H3 V100 混合精度补丁

[English](README.md) | 简体中文

本项目提供两个带版本识别和安全检查的安装脚本，用于安装经过 V100 实测的 MiniMax H3 混合精度方案：

- `patch_te_v100.bat`：用于已经包含 TE-Speed `block_loop` 钩子的 `model.py`。
- `patch_h3_origin_v100.bat`：用于不包含 TE 钩子的官方/原版 H3 `model.py`。
- `restore_te_v100.bat`：恢复补丁前的 TE 版精确备份。
- `restore_h3_origin_v100.bat`：恢复补丁前的官方/原版精确备份。

不要对同一个文件同时运行两个补丁。每个补丁都会先识别目标文件结构；如果选错版本，会在写入前拒绝执行。

## 实测结果

V100 测试用例：0.2 百万像素、5 秒、24 帧率。

测试环境：Windows 10、NVIDIA Tesla V100 32 GB，GPU 功率限制为 150 W。基线已经包含 TE-Speed 优化。在散热条件更好并提高功率限制时，性能可能进一步提升；预计 300 W 下会更快，但目前尚未进行实测，因此不计入下表结果。

| 方案 | 耗时 | 相对基线变化 | 结果 |
|---|---:|---:|---|
| TE-Speed 基线 | 317 秒 | — | 成功 |
| **MiniMax H3 V100 mixed-precision patch** | **206 秒** | **加速 35.0% / 1.54 倍** | **成功** |

本项目安装的是经过实测的 **MiniMax H3 V100 mixed-precision patch**。实际速度取决于工作负载、后端和软件版本；1.54 倍是上述环境中的实测结果，不代表所有环境都能获得相同提升。

## 精度分配

50 个主要 DiT Block 使用以下精度策略：

- FP16：融合 QKV 投影和优化后的注意力算子。
- FP32：Q/K RMSNorm、RoPE、注意力输出投影、MLP、AdaLN、残差累加和最终输出头。
- 保持源码计算精度：两个 Token Refiner Block。

补丁只会在主 Block 的输入张量为 CUDA FP32 时启用。BF16 和已经使用 FP16 的路径保持源码原有行为。

## Windows 一键使用

1. 关闭 ComfyUI。
2. 根据当前 `model.py` 选择对应 BAT：
   - 已经安装 TE-Speed 钩子：`patch_te_v100.bat`
   - 官方/原版 H3 文件：`patch_h3_origin_v100.bat`
3. 将 `ComfyUI\comfy\ldm\minimax\model.py` 拖到所选 BAT 文件上。
4. 确认控制台显示 `Patched SHA-256`，然后重新启动 ComfyUI。

如果要恢复补丁前的文件，请先关闭 ComfyUI，再将当前 `model.py` 拖到对应的恢复 BAT 上。恢复脚本会在写入前检查当前版本、V100 补丁标记以及匹配的干净备份。

BAT 启动器会按以下顺序寻找 Python：`COMFYUI_PYTHON`、附近便携版的 `python_embeded\python.exe`、Windows `py -3` 启动器，以及 `PATH` 中的 `python`。

命令行示例：

```bat
patch_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
patch_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py" --check
patch_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py" --revert

patch_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
patch_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py" --check
patch_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py" --revert

restore_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
restore_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
```

如果脚本位于 ComfyUI 安装目录内，也可以自动识别常见的 ComfyUI/便携版目录结构。当存在多个安装目录时，请使用 `--comfy-ui <根目录>` 或 `--model-file <文件>`。

## 备份与安全检查

首次写入前，安装脚本会在目标文件旁创建精确备份：

- TE 版：`model.py.v100_te.bak`
- 官方/原版：`model.py.v100_origin.bak`

安全机制包括：

- 写入前精确识别 TE/官方原版结构。
- 要求匹配受支持的 `Attention.forward` 锚点。
- 拒绝处理不完整的 TE 钩子、不完整的 V100 补丁以及选错版本的补丁脚本。
- 拒绝覆盖内容不同或已经过期的备份。
- 先写入临时文件，再进行原子替换。
- 写入前后进行 Python AST 语法检查。
- 输出输入文件、补丁后文件和恢复后文件的 SHA-256。
- 重复运行补丁不会重复修改文件。
- `--revert` 只会在当前文件可识别为 V100 补丁版本，且对应干净备份有效时执行。
- 独立恢复 BAT 调用的是同一条带安全检查的 `--revert` 路径，不会直接进行未经检查的文件复制。

## 性能测试建议

- 保持 H3 计算和残差流为 FP32；首次对比时不要叠加全局 `--force-fp16` 或 BF16。
- 固定随机种子、提示词、采样器、步数、分辨率、帧数、模型和注意力后端。
- 首次 A/B 测试时关闭无关的 TeaCache、SageAttention、编译和 FP16 累加修改。
- 丢弃第一次预热运行，并至少记录三次正式运行。
- 记录总耗时、每步耗时、显存峰值、视频有效性、音频有效性以及第一条警告/错误。

## Python 入口

BAT 文件只是轻量的 Windows 启动器。无第三方依赖的 Python 安装脚本也可以直接在 Windows 或 Linux 上运行：

```text
python patch_te_v100.py /path/to/ComfyUI
python patch_h3_origin_v100.py /path/to/ComfyUI
```

推荐使用 Python 3.8 或更高版本。

## 项目范围与署名

这是社区制作的混合精度补丁，不是 MiniMax、TE-Speed 或 ComfyUI 的官方版本。它会修改上游 ComfyUI 的 MiniMax H3 模型实现；使用 TE 专用安装脚本时会保留已有的 TE 钩子。源码哈希和验证来源见 `NOTICE.md`。

许可证的 SPDX 标识符为 `GPL-3.0-only`，与被修改实现所使用的上游 ComfyUI 许可证保持一致。完整条款见 `LICENSE`。
