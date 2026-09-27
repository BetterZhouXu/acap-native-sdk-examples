"""Cut the float32 trigonometry from this YOLOv8 OBB TFLite export.

Only the verified model fingerprint below is supported: tensor IDs are export-specific.
The resulting model outputs quantized DFL distances, class scores and angles.
"""

import argparse
import hashlib
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

SOURCE_SHA256 = "0ac7fdc4e5c39fb6d23ca38ed3c4864b4f85f4f4c0a8d99bccec9812dbe534b1"
OUTPUT_TENSORS = (455, 475, 450)  # [1,4,8400] ltrb, [1,4,8400] classes, [1,1,8400] angle


def export(source: Path, destination: Path) -> None:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unsupported model export: inspect its graph before changing output tensors")

    model = schema.ModelT.InitFromPackedBuf(raw)
    if len(model.subgraphs) != 1 or list(model.subgraphs[0].outputs) != [477]:
        raise ValueError("Unexpected TFLite graph")
    graph = model.subgraphs[0]
    if (list(graph.tensors[455].shape), list(graph.tensors[475].shape),
            list(graph.tensors[450].shape)) != ([1, 4, 8400], [1, 4, 8400], [1, 1, 8400]):
        raise ValueError("Unexpected OBB head dimensions")

    producers = {int(t): i for i, op in enumerate(graph.operators) for t in op.outputs}
    needed_tensors = set(graph.inputs) | set(OUTPUT_TENSORS)
    needed_ops = set()
    pending = list(OUTPUT_TENSORS)
    while pending:
        tensor = int(pending.pop())
        if tensor not in producers:
            continue
        op_index = producers[tensor]
        if op_index in needed_ops:
            continue
        needed_ops.add(op_index)
        for input_tensor in graph.operators[op_index].inputs:
            if input_tensor >= 0:
                pending.append(int(input_tensor))

    graph.operators = [op for i, op in enumerate(graph.operators) if i in needed_ops]
    for op in graph.operators:
        needed_tensors.update(int(i) for i in op.inputs if i >= 0)
        needed_tensors.update(int(i) for i in op.outputs if i >= 0)
        if op.intermediates is not None:
            needed_tensors.update(int(i) for i in op.intermediates if i >= 0)

    tensor_indices = sorted(needed_tensors)
    tensor_map = {old: new for new, old in enumerate(tensor_indices)}
    graph.tensors = [graph.tensors[i] for i in tensor_indices]
    graph.inputs = [tensor_map[int(i)] for i in graph.inputs]
    graph.outputs = [tensor_map[i] for i in OUTPUT_TENSORS]
    for op in graph.operators:
        op.inputs = [tensor_map[int(i)] if i >= 0 else -1 for i in op.inputs]
        op.outputs = [tensor_map[int(i)] for i in op.outputs]
        if op.intermediates is not None:
            op.intermediates = [tensor_map[int(i)] for i in op.intermediates]

    used_codes = sorted({op.opcodeIndex for op in graph.operators})
    code_map = {old: new for new, old in enumerate(used_codes)}
    for op in graph.operators:
        op.opcodeIndex = code_map[op.opcodeIndex]
    model.operatorCodes = [model.operatorCodes[i] for i in used_codes]
    model.signatureDefs = None  # Original signature refers to the removed single output.

    # This export stores weights after the FlatBuffer and refers to absolute file
    # offsets. Repacking changes those offsets; inline the weights instead.
    for buffer in model.buffers:
        if buffer.offset:
            if buffer.offset + buffer.size > len(raw):
                raise ValueError("Invalid external TFLite buffer")
            buffer.data = np.frombuffer(raw[buffer.offset:buffer.offset + buffer.size],
                                        dtype=np.uint8)
            buffer.offset = 0
            buffer.size = 0

    if any(t.type == schema.TensorType.FLOAT32 for t in graph.tensors):
        raise ValueError("Rewritten graph still contains FLOAT32 tensors")
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(builder.Output())
    print(f"Wrote {destination}: {len(graph.operators)} ops, "
          f"{len(graph.tensors)} tensors, three INT8 outputs, no FLOAT32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
