import onnx
model = onnx.load('inswapper_128.onnx')

print("=== MODEL METADATA ===")
print(f"Input Nodes: {[i.name for i in model.graph.input]}")
print(f"Output Nodes: {[o.name for o in model.graph.output]}")

for i in model.graph.input:
    shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f"Input '{i.name}' Shape: {shape}")

print("\n=== VERIFICATION ===")
print("1. If inputs are 'target' and 'source', you are using the standard InsightFace.")
print("2. The standard InsightFace requires: Input: BGR, Scale: [0, 1].")
print("3. If this script shows different input names, your code's 'targetKey'/'sourceKey' are wrong.")