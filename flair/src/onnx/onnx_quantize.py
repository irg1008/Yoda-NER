from os import path

from onnxruntime.quantization.quant_utils import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from pathlib import Path


# TODO


def quantize_onnx_model(onnx_model_path: str):
    """Quantize an ONNX model.

    Args:
        onnx_model_path: Path to the ONNX model.
    """
    quantized_model_path = onnx_model_path.replace(".onnx", "_quantized.onnx")

    # onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(
        Path(onnx_model_path),
        Path(quantized_model_path),
        weight_type=QuantType.QUInt8,
    )

    print(f"quantized model saved to: {quantized_model_path}")


def main():
    model_path = path.join(path.dirname(__file__), "../../models/lite")
    best_model_path = path.join(model_path, "best-model.pt")
    onnx_model_path = best_model_path.replace(".pt", ".onnx")

    quantize_onnx_model(onnx_model_path)


if __name__ == "__main__":
    main()
