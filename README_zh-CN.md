# MiniMax H3 V100 — Custom Node v0.1.4

[English](README.md) | 简体中文

0.1.4 修复ComfyUI 0.34.5中出现的 `_patched_final_forward() takes 5 positional arguments but 8 were given`问题，保留独立的 FP16 存储/计算分支及 FP32 保护。目前完成本地结构、参数传递和精度流回归。

## 安装 / 升级

1. 完全关闭 ComfyUI。
2. 从 `custom_nodes` 删除旧 H3 补丁目录。
3. 将 `minimax-h3-v100-v0.1.4.zip` 解压到实际使用的 `custom_nodes`。你提供的报错对应 `C:\Comfyui\custom_nodes`。正确目录层级为：

   ```text
   custom_nodes/minimax-h3-v100-l3-clean/__init__.py
   custom_nodes/minimax-h3-v100-l3-clean/runtime_patch.py
   ```

4. 正常启动 ComfyUI，无需添加工作流节点或 `--fp16-unet`。ComfyUI 的 `comfy/ldm/minimax/model.py` 应保持官方原版。
5. 启动日志确认 `v0.1.4 runtime profile installed`；模型加载时确认 `enabled v0.1.4` 及主 DiT block 数量。

本包不含 TE 适配版或旧式源码修改器。如果之前手动修改过 ComfyUI 源码，先恢复当前 ComfyUI 版本对应的官方文件。仅安装过 Custom Node 的情况下，卸载只需删除插件目录并重启。

## 精度策略

- ComfyUI 原生 FP16 权重加载、预取及卸载。
- 主 DiT 的 attention/MLP 分支使用 FP16；目标及参考音频 attention 使用 FP32 重算。
- Attention 输出：`/64 → FP16 投影 → FP32 ×64`。
- MLP：FP16 `fc1`、FP32 SwiGLU、`/256 → FP16 fc2 → FP32 ×256`。
- 残差、归一化/调制、condition 输入、Token Refiner 和最终输出头保留 FP32 保护。
- 保留 CUDA 7.0 设备限制、重复补丁检查；不使用临时 `weight.data` 转换。

Attention 和 MLP 计算函数的 AST 哈希与 0.1.3 一致。此前 0.1.3 文档记录了 ComfyUI 0.33.2 实测通过和驻留显存下降；这些历史结果不能代替 0.1.4 的服务器实测。

## 致谢与许可

感谢 ComfyUI、MiniMax 以及 [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix) 的 Custom Node 交付模式及相关混合精度工作。本项目为独立社区扩展。许可：`GPL-3.0-only`，见 `LICENSE` 和 `NOTICE.md`。
