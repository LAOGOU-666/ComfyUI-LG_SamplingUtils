"""
ZImage Noise Injection 节点

通过 CFG 机制将参考图像的特征注入到生成中。
可以让生成结果"学习"参考图像的某些特质（如水珠、纹理、质感等）。
"""

import torch
import torch.nn.functional as F
import logging
import comfy.model_management
import comfy.latent_formats


class LGNoiseInjection:
    """
    LG Noise Injection - 特征注入
    
    工作原理：
    参考图像的 latent 特征会被注入到 CFG 过程中，
    模型会"学习"参考图像的某些特质并应用到生成结果上。
    
    适用场景：
    - 添加水珠、汗珠等表面细节
    - 添加纹理、材质感
    - 添加光泽、反射效果
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "模型"
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE 编码器"
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "参考图像（含有你想要注入的特征）"
                }),
                "strength": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "注入强度。0.1-0.2 轻微，0.2-0.4 明显"
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "开始注入的采样进度"
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "结束注入的采样进度"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "遮罩，白色区域会被注入特征"
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"
    DESCRIPTION = "将参考图像的特征（如水珠、纹理等）注入到生成结果中。"

    def apply(self, model, vae, reference_image, strength, start_percent, end_percent, mask=None):
        if strength <= 0:
            return (model,)
        
        m = model.clone()
        
        # 编码参考图像（使用模型的 latent format）
        ref_latent = self._encode_reference(vae, reference_image, model)
        
        # 预处理 mask
        mask_latent = None
        if mask is not None:
            mask_latent = self._prepare_mask(mask)
        
        step_counter = [0]
        
        def cfg_function(args):
            """在 CFG 合并后注入参考特征"""
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            
            step_counter[0] += 1
            
            # 标准 CFG
            cfg_result = uncond + cond_scale * (cond - uncond)
            
            # 计算进度
            sigma = float(timestep[0]) if timestep.dim() > 0 else float(timestep)
            progress = 1.0 - min(sigma, 1.0)
            
            # 检查是否在作用范围内
            if progress < start_percent or progress > end_percent:
                return cfg_result
            
            # 准备 reference
            ref = ref_latent.to(device=cfg_result.device, dtype=cfg_result.dtype)
            
            if ref.shape[2:] != cfg_result.shape[2:]:
                # 根据张量维度选择插值模式
                if len(ref.shape) == 5:  # 5D张量 [B, C, T, H, W]
                    ref = F.interpolate(ref, size=cfg_result.shape[2:], mode='trilinear', align_corners=False)
                else:  # 4D张量 [B, C, H, W]
                    ref = F.interpolate(ref, size=cfg_result.shape[2:], mode='bilinear', align_corners=False)
            
            if ref.shape[0] != cfg_result.shape[0]:
                if ref.shape[0] == 1:
                    # 根据张量维度调整expand
                    if len(ref.shape) == 5:
                        ref = ref.expand(cfg_result.shape[0], -1, -1, -1, -1)
                    else:
                        ref = ref.expand(cfg_result.shape[0], -1, -1, -1)
                else:
                    ref = ref[:cfg_result.shape[0]]
            
            # 准备 mask
            current_mask = None
            if mask_latent is not None:
                current_mask = mask_latent.to(device=cfg_result.device, dtype=cfg_result.dtype)
                # 调整 mask 尺寸到 latent 空间
                if current_mask.shape[2:] != cfg_result.shape[2:]:
                    # 根据张量维度选择插值模式
                    if len(current_mask.shape) == 5:  # 5D张量 [B, C, T, H, W]
                        current_mask = F.interpolate(current_mask, size=cfg_result.shape[2:], mode='trilinear', align_corners=False)
                    else:  # 4D张量 [B, C, H, W]
                        current_mask = F.interpolate(current_mask, size=cfg_result.shape[2:], mode='bilinear', align_corners=False)
                # 调整 batch size
                if current_mask.shape[0] != cfg_result.shape[0]:
                    if current_mask.shape[0] == 1:
                        # 根据张量维度调整expand
                        if len(current_mask.shape) == 5:
                            current_mask = current_mask.expand(cfg_result.shape[0], -1, -1, -1, -1)
                        else:
                            current_mask = current_mask.expand(cfg_result.shape[0], -1, -1, -1)
                    else:
                        current_mask = current_mask[:cfg_result.shape[0]]
            
            # 计算有效强度（线性衰减）
            if end_percent > start_percent:
                range_progress = (progress - start_percent) / (end_percent - start_percent)
                decay = 1.0 - range_progress
            else:
                decay = 1.0
            effective_strength = strength * decay
            
            # 计算特征注入方向
            feature_direction = ref - cfg_result
            
            # 控制注入幅度，避免过度偏移
            cfg_std = cfg_result.std()
            feature_std = feature_direction.std()
            if feature_std > cfg_std * 3:
                feature_direction = feature_direction * (cfg_std * 3 / feature_std)
            
            # 应用遮罩
            if current_mask is not None:
                # mask: 1 = 注入区域, 0 = 保持原样
                feature_direction = feature_direction * current_mask
            
            # 应用特征注入
            injected = cfg_result + feature_direction * effective_strength
            
            if step_counter[0] <= 3:
                mask_info = "with mask" if current_mask is not None else "no mask"
                logging.warning(f"[FeatureInj] step={step_counter[0]} | progress={progress:.2f} | eff_str={effective_strength:.3f} | {mask_info}")
            
            return injected
        
        m.set_model_sampler_cfg_function(cfg_function)
        
        return (m,)
    
    def _encode_reference(self, vae, reference_image, model):
        """编码参考图像"""
        loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
        latent = vae.encode(reference_image)
        # 使用模型自己的 latent format，而不是硬编码 Flux
        latent_format = model.get_model_object("latent_format")
        latent = latent_format.process_in(latent)
        comfy.model_management.load_models_gpu(loaded_models)
        logging.warning(f"[FeatureInj] Reference encoded: shape={latent.shape}")
        return latent
    
    def _prepare_mask(self, mask):
        """准备遮罩，转换为 latent 空间格式"""
        # mask 输入格式: [B, H, W] 或 [H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        # 添加 channel 维度: [B, H, W] -> [B, 1, H, W]
        mask = mask.unsqueeze(1)
        
        logging.warning(f"[FeatureInj] Mask prepared: shape={mask.shape}")
        return mask


class LGNoiseInjectionLatent:
    """
    LG Noise Injection (Latent) - 直接使用 Latent 的特征注入
    
    工作原理：
    直接输入参考 latent，其特征会被注入到 CFG 过程中。
    如果 latent 包含 noise_mask，则自动使用该遮罩。
    
    适用场景：
    - 添加水珠、汗珠等表面细节
    - 添加纹理、材质感
    - 添加光泽、反射效果
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "模型"
                }),
                "reference_latent": ("LATENT", {
                    "tooltip": "参考 latent（含有你想要注入的特征）"
                }),
                "strength": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "注入强度。0.1-0.2 轻微，0.2-0.4 明显"
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "开始注入的采样进度"
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "结束注入的采样进度"
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"
    DESCRIPTION = "直接输入 latent 进行特征注入，自动使用 latent 的 noise_mask 作为遮罩。"

    def apply(self, model, reference_latent, strength, start_percent, end_percent):
        if strength <= 0:
            return (model,)
        
        m = model.clone()
        
        # 获取 latent samples
        ref_latent = reference_latent["samples"]
        
        # 获取 noise_mask（如果存在）
        mask_latent = None
        if "noise_mask" in reference_latent:
            mask_latent = self._prepare_mask(reference_latent["noise_mask"])
            logging.warning(f"[FeatureInjLatent] Using noise_mask from latent")
        
        logging.warning(f"[FeatureInjLatent] Reference latent: shape={ref_latent.shape}")
        
        step_counter = [0]
        
        def cfg_function(args):
            """在 CFG 合并后注入参考特征"""
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            
            step_counter[0] += 1
            
            # 标准 CFG
            cfg_result = uncond + cond_scale * (cond - uncond)
            
            # 计算进度
            sigma = float(timestep[0]) if timestep.dim() > 0 else float(timestep)
            progress = 1.0 - min(sigma, 1.0)
            
            # 检查是否在作用范围内
            if progress < start_percent or progress > end_percent:
                return cfg_result
            
            # 准备 reference
            ref = ref_latent.to(device=cfg_result.device, dtype=cfg_result.dtype)
            
            if ref.shape[2:] != cfg_result.shape[2:]:
                # 根据张量维度选择插值模式
                if len(ref.shape) == 5:  # 5D张量 [B, C, T, H, W]
                    ref = F.interpolate(ref, size=cfg_result.shape[2:], mode='trilinear', align_corners=False)
                else:  # 4D张量 [B, C, H, W]
                    ref = F.interpolate(ref, size=cfg_result.shape[2:], mode='bilinear', align_corners=False)
            
            if ref.shape[0] != cfg_result.shape[0]:
                if ref.shape[0] == 1:
                    # 根据张量维度调整expand
                    if len(ref.shape) == 5:
                        ref = ref.expand(cfg_result.shape[0], -1, -1, -1, -1)
                    else:
                        ref = ref.expand(cfg_result.shape[0], -1, -1, -1)
                else:
                    ref = ref[:cfg_result.shape[0]]
            
            # 准备 mask
            current_mask = None
            if mask_latent is not None:
                current_mask = mask_latent.to(device=cfg_result.device, dtype=cfg_result.dtype)
                # 调整 mask 尺寸到 latent 空间
                if current_mask.shape[2:] != cfg_result.shape[2:]:
                    # 根据张量维度选择插值模式
                    if len(current_mask.shape) == 5:  # 5D张量 [B, C, T, H, W]
                        current_mask = F.interpolate(current_mask, size=cfg_result.shape[2:], mode='trilinear', align_corners=False)
                    else:  # 4D张量 [B, C, H, W]
                        current_mask = F.interpolate(current_mask, size=cfg_result.shape[2:], mode='bilinear', align_corners=False)
                # 调整 batch size
                if current_mask.shape[0] != cfg_result.shape[0]:
                    if current_mask.shape[0] == 1:
                        # 根据张量维度调整expand
                        if len(current_mask.shape) == 5:
                            current_mask = current_mask.expand(cfg_result.shape[0], -1, -1, -1, -1)
                        else:
                            current_mask = current_mask.expand(cfg_result.shape[0], -1, -1, -1)
                    else:
                        current_mask = current_mask[:cfg_result.shape[0]]
            
            # 计算有效强度（线性衰减）
            if end_percent > start_percent:
                range_progress = (progress - start_percent) / (end_percent - start_percent)
                decay = 1.0 - range_progress
            else:
                decay = 1.0
            effective_strength = strength * decay
            
            # 计算特征注入方向
            feature_direction = ref - cfg_result
            
            # 控制注入幅度，避免过度偏移
            cfg_std = cfg_result.std()
            feature_std = feature_direction.std()
            if feature_std > cfg_std * 3:
                feature_direction = feature_direction * (cfg_std * 3 / feature_std)
            
            # 应用遮罩
            if current_mask is not None:
                # mask: 1 = 注入区域, 0 = 保持原样
                feature_direction = feature_direction * current_mask
            
            # 应用特征注入
            injected = cfg_result + feature_direction * effective_strength
            
            if step_counter[0] <= 3:
                mask_info = "with noise_mask" if current_mask is not None else "no mask"
                logging.warning(f"[FeatureInjLatent] step={step_counter[0]} | progress={progress:.2f} | eff_str={effective_strength:.3f} | {mask_info}")
            
            return injected
        
        m.set_model_sampler_cfg_function(cfg_function)
        
        return (m,)
    
    def _prepare_mask(self, mask):
        """准备遮罩，转换为 latent 空间格式"""
        # mask 输入格式: [B, H, W] 或 [H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        # 添加 channel 维度: [B, H, W] -> [B, 1, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        return mask


NODE_CLASS_MAPPINGS = {
    "LGNoiseInjection": LGNoiseInjection,
    "LGNoiseInjectionLatent": LGNoiseInjectionLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LGNoiseInjection": "🎈LG Noise Injection",
    "LGNoiseInjectionLatent": "🎈LG Noise Injection (Latent)",
}
