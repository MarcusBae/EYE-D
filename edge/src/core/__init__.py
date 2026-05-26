import torch

# Try to import intel_extension_for_pytorch (IPEX) to register XPU device backend in PyTorch.
# Catch BaseException to handle version-mismatch prints/crashes (such as the 'os.exit' AttributeError in IPEX).
try:
    import intel_extension_for_pytorch as ipex
except BaseException:
    pass

# Monkey-patch BoxMOT's assert_cuda_available to bypass CUDA validation
# and allow XPU (Intel GPU) or other hardware execution support
try:
    import boxmot.utils.torch_utils as boxmot_torch
    boxmot_torch.assert_cuda_available = lambda device: None
except Exception:
    pass

# Monkey-patch Ultralytics select_device to natively support XPU without raising ValueError
try:
    import ultralytics.utils.torch_utils as ultralytics_torch
    _orig_select_device = ultralytics_torch.select_device
    
    def _patched_select_device(device="", *args, **kwargs):
        if str(device).strip().lower() == 'xpu':
            return torch.device('xpu')
        return _orig_select_device(device, *args, **kwargs)
        
    ultralytics_torch.select_device = _patched_select_device
except Exception:
    pass
