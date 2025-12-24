"""
Interactive Sigmas Editor Node
Allows real-time adjustment of sigmas curve by dragging points
"""

import torch
import numpy as np
import json
import os
import folder_paths
from server import PromptServer
from aiohttp import web

# Set web directory for custom UI
WEB_DIRECTORY = "./web"


class SigmasEditor:
    """Interactive editor for adjusting sigmas curve"""
    
    # 类级别缓存，存储每个节点上次接收的输入sigmas（用于检测输入是否变化）
    _last_sent_data = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"tooltip": "Input sigmas schedule to edit"}),
                "sigmas_adjustments": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "dynamicPrompts": False,
                    "tooltip": "JSON array of adjusted sigma values for each step"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("adjusted_sigmas",)
    FUNCTION = "adjust_sigmas"
    CATEGORY = "🎈LAOGOU/Sampling Utils"
    DESCRIPTION = "Interactively adjust sigmas curve by dragging control points"
    
    def adjust_sigmas(self, sigmas, sigmas_adjustments="[]", unique_id=None):
        # Convert sigmas to numpy
        if isinstance(sigmas, torch.Tensor):
            sigmas_np = sigmas.cpu().numpy()
        else:
            sigmas_np = np.array(sigmas)
        
        # Parse adjusted sigma values from JSON
        try:
            adjusted_values = json.loads(sigmas_adjustments)
        except:
            adjusted_values = []
        
        # If no adjustments or length mismatch, use original sigmas
        if len(adjusted_values) != len(sigmas_np):
            adjusted_sigmas = sigmas_np.copy()
        else:
            # Use the adjusted sigma values directly
            adjusted_sigmas = np.array(adjusted_values, dtype=np.float64)
        
        # Ensure last sigma is still 0 if original was 0
        if sigmas_np[-1] == 0:
            adjusted_sigmas[-1] = 0
        
        result_tensor = torch.FloatTensor(adjusted_sigmas)
        
        # Send sigmas data to frontend via PromptServer (只有输入sigmas改变时才发送)
        if unique_id is not None:
            # 只根据输入的sigmas创建缓存键（不包括adjustments）
            # 使用 hash(sigmas_np.tobytes()) 避免浮点数精度比较问题
            current_sigmas_key = hash(sigmas_np.tobytes())
            
            # 检查是否与上次输入的sigmas相同
            last_sigmas_key = self._last_sent_data.get(unique_id)
            
            # 只有输入sigmas改变时才发送数据到前端
            if last_sigmas_key != current_sigmas_key:
                # 当输入sigmas改变时，不发送旧的调整值，让前端重置为原始值
                PromptServer.instance.send_sync("sigmas_editor_update", {
                    "node_id": unique_id,
                    "sigmas_data": {
                        "original": sigmas_np.tolist(),
                        "adjusted": None,  # 重置为None，让前端使用新的原始值
                    }
                })
                
                # 更新缓存（只缓存输入的sigmas）
                self._last_sent_data[unique_id] = current_sigmas_key
        
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "SigmasEditor": SigmasEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SigmasEditor": "Sigmas Editor 🎚️",
}

