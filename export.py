# Update model

print('\nKhadas/RK3588')
for k, m in model.named_modules():
    if isinstance(m, Detect):                
        m.inplace = inplace
        m.onnx_dynamic = dynamic
        m.forward = m.forward_export
        m.export = True

