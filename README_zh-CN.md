# MiniMax H3 V100 — Custom Node v0.1.4

[English](README.md) | 简体中文

0.1.4 修复 `_patched_final_forward() takes 5 positional arguments but 8 were given`，保留独立 L3 的 FP16 存储/计算分支及 FP32 保护。目前完成本地结构、参数传递和精度流回归；**V100 实际推理、画面、音频、速度和显存仍待服务器测试**。

## 安装 / 升级

1. 完全关闭 ComfyUI。
2. 从 `custom_nodes` 删除旧 H3 补丁目录，包括旧的 `minimax-h3-v100-l3-clean`、`minimax-h3-v100-patch` 或 preview 副本；移除已安装的 `minimax-h3-v100-l3-te` 和 TE-Speed。只留一个 H3 精度补丁。
3. 将 `minimax-h3-v100-v0.1.4.zip` 解压到实际使用的 `custom_nodes`。你提供的报错对应 `C:\Comfyui\custom_nodes`。正确目录层级为：

   ```text
   custom_nodes/minimax-h3-v100-l3-clean/__init__.py
   custom_nodes/minimax-h3-v100-l3-clean/runtime_patch.py
   ```

4. 正常启动 ComfyUI，无需添加工作流节点或 `--fp16-unet`。ComfyUI 的 `comfy/ldm/minimax/model.py` 应保持官方原版。
5. 启动日志确认 `v0.1.4 runtime profile installed`；模型加载时确认 `enabled v0.1.4` 及主 DiT block 数量。

本包不含 TE 适配版或旧式源码修改器。如果之前手动修改过 ComfyUI 源码，先恢复当前 ComfyUI 版本对应的官方文件。仅安装过 Custom Node 的情况下，卸载只需删除插件目录并重启。

## 本次修复

官方 `v0.34.5` 标签版仍是四输入 FinalLayer 接口。你的报告标注 `0.34.0`，但实际 H3 源码已调用新增 `sigma`、`sample_sigmas`、`shifts` 的七输入接口，因此按实际源码接口兼容，不单凭版本号判断。

- FinalLayer 将隐藏状态和时间嵌入转成 FP32 后，完整传递所有位置参数、关键字参数给原生实现。
- 保留上游调制行处理、PDD 输出头选择、采样时间表检查和音视频各自的 shift 权重逻辑。
- 适配新版 DiT block 的 `attention` 参数；旧版调用未提供该参数时，不向旧实现额外传入。
- Attention/MLP/DiT 的重写接口出现未知参数时，在注册 FP16 之前停止安装，避免不完整覆盖。

核对源码：[ComfyUI v0.34.5](https://github.com/Comfy-Org/ComfyUI/blob/v0.34.5/comfy/ldm/minimax/model.py) 和 [PDD 主线固定提交 15eb748b](https://github.com/Comfy-Org/ComfyUI/blob/15eb748b3ec5f8a0a2d470b7fb280e2d7579f916/comfy/ldm/minimax/model.py)。完整哈希见 `tests/fixtures/PROVENANCE.json`。

## 精度策略

- ComfyUI 原生 FP16 权重加载、预取及卸载。
- 主 DiT 的 attention/MLP 分支使用 FP16；目标及参考音频 attention 使用 FP32 重算。
- Attention 输出：`/64 → FP16 投影 → FP32 ×64`。
- MLP：FP16 `fc1`、FP32 SwiGLU、`/256 → FP16 fc2 → FP32 ×256`。
- 残差、归一化/调制、condition 输入、Token Refiner 和最终输出头保留 FP32 保护。
- 保留 CUDA 7.0 设备限制、重复补丁检查；不使用临时 `weight.data` 转换。

Attention 和 MLP 计算函数的 AST 哈希与 0.1.3 一致。此前 0.1.3 文档记录了 ComfyUI 0.33.2 实测通过和驻留显存下降；这些历史结果不能代替 0.1.4 的服务器实测。

## 服务器测试

先复跑这次报错的短工作流，固定模型、种子、采样器、步数、分辨率及 offload 分配。确认采样完成、画面不黑、音频正常，再记录耗时及显存。如果继续使用 Sol-Attn，保持报告中的 `fp16_safe=False`，避免叠加另一套 H3 精度补丁。稀疏注意力组合效果和速度以服务器结果为准。

## 本地验证与打包

```text
python -B -m unittest discover -s tests -v
python -B tools/build_release.py
```

回归测试用 trace tensor 执行真实上游 FinalLayer 函数体，覆盖普通/PDD 分支、参数传递及采样表检查；PDD 输出头算术使用调用记录替身，因此不等同于 PyTorch 数值验证或 GPU 推理。打包脚本使用明确文件清单，生成 manifest，核对 ZIP 内容并输出 SHA-256 校验文件。

## 致谢与许可

感谢 ComfyUI、MiniMax 以及 [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix) 的 Custom Node 交付模式及相关混合精度工作。本项目为独立社区扩展。许可：`GPL-3.0-only`，见 `LICENSE` 和 `NOTICE.md`。
