# MiniMax H3 V100 混合精度补丁

[English](README.md) | 简体中文

本项目提供两个带版本识别和安全检查的安装脚本，用于安装经过 V100 实测的 MiniMax H3 混合精度方案：

- `patch_te_v100.bat`：用于 TE-Speed 版本；如果目标是受支持的官方 origin 文件，会先自动安装经过校验的 TE-Speed `block_loop` 钩子，再应用 V100 补丁。
- `patch_h3_origin_v100.bat`：用于不包含 TE 钩子的官方/原版 H3 `model.py`。
- `restore_te_v100.bat`：恢复补丁前的 TE 版精确备份。
- `restore_h3_origin_v100.bat`：恢复补丁前的官方/原版精确备份。

不要对同一个文件同时运行两个补丁。TE 安装脚本可以自动把受支持的干净 origin 文件转换为 TE 版本；origin 安装脚本仍只接受官方原版结构。

## ComfyUI 0.31.1 音频兼容更新

旧版补丁会把 H3 联合序列中的全部 Attention 都放到 FP16，其中也包括音频查询。在 ComfyUI 0.31.1 更新 MiniMax audio carry/sampler 路径后，旧精度边界可能导致声音异常，即使同一份未打补丁的新版 `model.py` 声音完全正常。

本次更新已经针对新版音频路径完成适配：

- 文本和视频查询的 Attention 继续使用 FP16 加速。
- 目标音频和参考音频查询单独使用 FP32 Attention 重新计算，并只覆盖对应的输出行。
- 完整保留新版 audio carry/sampler 逻辑；不修改 `audio_vae.py`，也不修改 VAE 启动参数。
- 主 DiT 的 QKV 计算和 50 层 QKV 常驻权重使用 FP16。V100 实测表明，QKV 权重常驻 FP16 还可以根据工作负载和卸载方式额外提速约 **5%–10%**。

origin 版和带 TE 钩子的版本均已通过本轮 V100 视频及音频测试。如果安装脚本显示 `legacy`，请先使用对应的 `--revert` 恢复，必要时更新 ComfyUI，然后重新安装本版本。为保证备份安全，旧补丁不会被直接原地升级。

## 性能依据

V100 测试用例：0.2 百万像素、5 秒、24 帧率。

测试环境：Windows 10、NVIDIA Tesla V100 32 GB，GPU 功率限制为 150 W。基线已经包含 TE-Speed 优化。在散热条件更好并提高功率限制时，性能可能进一步提升；预计 300 W 下会更快，但目前尚未进行实测，因此不计入下表结果。

| 方案 | 耗时 | 相对基线变化 | 结果 |
|---|---:|---:|---|
| TE-Speed 基线 | 317 秒 | — | 成功 |
| **MiniMax H3 V100 mixed-precision patch** | **206 秒** | **加速 35.0% / 1.54 倍** | **成功** |

317 秒和 206 秒来自早期 V100 混合精度核心方案的实测，可用于说明保留下来的视频侧 FP16 路径具有明显加速潜力，但不应视为本次音频兼容更新的精确端到端耗时。

在当前音频兼容版本中，相比每次推理时重新转换 QKV 权重的相同方案，QKV 权重常驻 FP16 额外获得了约 **5%–10%** 的实测提升。本轮没有提供统一的绝对耗时，因此不对所有工作流承诺固定速度。

实际速度取决于工作负载、注意力后端和软件版本；1.54 倍以及额外 5%–10% 都不代表所有环境都能获得相同提升。

## 精度分配

50 个主要 DiT Block 使用以下精度策略：

- FP16：常驻的融合 QKV 权重、融合 QKV 投影，以及文本/视频查询的 Attention。
- FP32：目标/参考音频查询的 Attention、Q/K RMSNorm、RoPE、注意力输出投影、MLP、AdaLN、残差累加和最终输出头。
- 保持源码计算精度：两个 Token Refiner Block。

音频 FP32 Attention 使用的 Q/K/V 数值来自 FP16 QKV 投影，但音频 Attention 运算本身使用 FP32。补丁只会在主 Block 的输入张量为 CUDA FP32 时启用。BF16 和已经使用 FP16 的路径保持源码原有行为。

## 随附完整源码

项目内提供两份完整的 V100 加速版 `model.py`，便于开发者检查代码、按需修改，或移植到自己的 ComfyUI 构建：

- [`sources/origin_v100/model.py`](sources/origin_v100/model.py)：当前官方/origin MiniMax H3 结构，加上经过实测的音频安全 V100 方案；不包含 TE-Speed `block_loop` 钩子。
- [`sources/te_v100/model.py`](sources/te_v100/model.py)：包含 TE-Speed `block_loop` 钩子，并使用相同的音频安全 V100 方案。

两份源码均基于包含 [`93cb5edb`](https://github.com/Comfy-Org/ComfyUI/commit/93cb5edb) audio carry 修复的新版 ComfyUI MiniMax H3 实现。受支持的干净 origin `model.py` SHA-256 为 `1c9828ec3d38ac01398e45b1edf8d7db38fcc8148c5eb3ba8fb92b762147d0ce`。

如果 ComfyUI 构建与该源码版本匹配，可以先备份，再将所选文件作为 `comfy/ldm/minimax/model.py` 使用。对于更新、较旧或已自行修改的构建，应把这两份文件作为参考源码，将 Attention 精度改动以及可选的 TE Block Loop 钩子移植到本地实现，不建议直接覆盖。替换前请关闭 ComfyUI，并在正式使用前验证 Python 语法和模型输出。随附源码主要用于手动开发；对于受支持的版本，带安全检查的 BAT/Python 安装器仍是更稳妥的选择。

## Windows 一键使用

1. 关闭 ComfyUI。
2. 根据当前 `model.py` 选择对应 BAT：
   - 需要 TE-Speed 版本（当前文件可以是干净 origin 或已安装 TE 钩子）：`patch_te_v100.bat`
   - 官方/原版 H3 文件：`patch_h3_origin_v100.bat`
3. 直接双击所选 BAT。本定制版默认目标为 `C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py`；也仍可将其他 `model.py` 拖到 BAT 上。
4. 确认控制台显示 `Patched SHA-256`，然后重新启动 ComfyUI。

如果要恢复补丁前的文件，请先关闭 ComfyUI，再运行对应的恢复 BAT。若 TE 安装从 origin 文件开始，`restore_te_v100.bat` 会精确恢复该 origin 文件；若从已有 TE 文件开始，则精确恢复原 TE 文件。

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

- TE 版：`model.py.v100_te.bak`（保存点击 TE 安装前的精确文件，可为 origin 或 TE）
- 官方/原版：`model.py.v100_origin.bak`

安全机制包括：

- 写入前精确识别 TE/官方原版结构，并只在两个 TE 转换锚点均匹配时执行 origin→TE 转换。
- 要求匹配受支持的 `Attention.forward` 锚点。
- 要求存在新版 MiniMax audio carry/sampler 锚点；旧源码会被拒绝，必须先更新 ComfyUI。
- 旧的音频不兼容 V100 补丁会被识别为 `legacy`；可以安全恢复，但不会被静默原地升级。
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
