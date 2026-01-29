# new_models/utils_check.py
import torch
import logging

def print_model_structure_info(model, logger=None):
    """
    打印模型的关键结构信息，用于确认维度是否符合预期。
    """
    log_func = logger.info if logger else print
    
    log_func("\n" + "="*30 + " 模型结构维度检查 " + "="*30)
    
    # 1. 检查 Image Encoder
    if hasattr(model, 'image_encoder'):
        enc = model.image_encoder
        log_func(f"[Image Encoder] Type: {type(enc).__name__}")
        if hasattr(enc, 'embed_dims'):
            log_func(f"  - embed_dims: {enc.embed_dims}")
        if hasattr(enc, 'patch_embed') and hasattr(enc.patch_embed, 'proj'):
            w = enc.patch_embed.proj.weight
            log_func(f"  - Actual PatchEmbed Weight: {w.shape} (OutCh should match {enc.embed_dims[0]})")
        
        # 尝试检查第一层的具体维度
        if hasattr(enc, 'patch_embed') and hasattr(enc.patch_embed, 'proj'):
            weight = enc.patch_embed.proj.weight
            log_func(f"  - PatchEmbed Output Dim (from weight): {weight.shape[0]} (Expected embed_dim)")
            
    # 2. 检查 Video Encoder
    if hasattr(model, 'video_encoder'):
        v_enc = model.video_encoder
        log_func(f"[Video Encoder] Type: {type(v_enc).__name__}")
        if hasattr(v_enc, 'hidden_dim'):
            log_func(f"  - hidden_dim: {v_enc.hidden_dim}")

    # 3. 检查 Decoder
    if hasattr(model, 'image_decoder'):
        dec = model.image_decoder
        if hasattr(dec, 'embed_dims'):
             log_func(f"[Image Decoder] embed_dims: {dec.embed_dims}")

    log_func("="*80 + "\n")

def check_state_dict_compatibility(model, state_dict, logger=None):
    """
    在加载权重前，显式对比当前模型和 Checkpoint 的维度。
    返回: (bool, mismatch_details)
    """
    model_state = model.state_dict()
    mismatch_keys = []
    
    log_func = logger.info if logger else print
    log_func(">>>正在检查权重形状兼容性...")

    is_compatible = True
    
    for key, param in state_dict.items():
        if key not in model_state:
            # 忽略不存在的key（可能是不同版本的冗余参数）
            continue
            
        model_shape = model_state[key].shape
        ckpt_shape = param.shape
        
        if model_shape != ckpt_shape:
            is_compatible = False
            error_msg = f"  Mismatch: {key}\n    - Model expect: {model_shape}\n    - Checkpoint has: {ckpt_shape}"
            mismatch_keys.append(error_msg)

    if not is_compatible:
        log_func("\n!!!!!! 致命错误：权重维度不匹配 (DIMENSION MISMATCH) !!!!!!")
        log_func("这意味着你正在尝试将一个【小模型的权重】加载到【大模型】中，或者反之。")
        for err in mismatch_keys[:20]: # 只打印前20个错误
            log_func(err)
        if len(mismatch_keys) > 20:
            log_func(f"... 还有 {len(mismatch_keys) - 20} 个不匹配项。")
        log_func("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return False
    
    log_func(">>> 兼容性检查通过：所有匹配键的形状一致。\n")
    return True