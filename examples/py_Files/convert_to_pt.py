import torch
import os
import subprocess
import torchvision.models as models
from pathlib import Path

ONNX_DIR = "../../models/onnx_dir"


# this needs to get more general prurpose than this otherwise it becomees hyper specific
def convert():
    """
    this function converts a model with a .pth extension
    to .pt to allow it to be handled by burn_vision
    """
    try:
        if not os.path.exists(ONNX_DIR):
            print(f"Creating dir {ONNX_DIR}")
            subprocess.run(["mkdir", ONNX_DIR])

        for model_path in Path("../models").iterdir():
            if "googlenet" in f"{model_path}":
                print(f"{model_path}")
                name = "googlenet"
                actual_model = models.googlenet()
                if os.path.exists(f"{ONNX_DIR}/{name}.onnx"):
                    print(f"{model_path} already converted")
                else:
                    convert_model(model_path, (1, 3, 224, 244), name, actual_model)
                    print(f"successfully converted {model_path} to onnx format")

                # Basically had to convert it directly after its installation
                # it is incapable of recognizing certain state dicts as is implemented

            # elif "densenet121" in f"{model_path}":
            #     print(f"{model_path}")
            #     name = "densenet121"
            #     densenet_model = models.densenet121()
            #     if os.path.exists(f"{ONNX_DIR}/{name}.onnx"):
            #         print(f"{model_path} already converted")
            #     else:
            #         convert_model(model_path, (1, 3, 224, 244), name, densenet_model)
            #         print(f"successfully converted {model_path} to onnx format")

            elif "inception" in f"{model_path}":
                print(f"{model_path}")
                name = "inceptionv3"
                inception_model = models.inception_v3()
                if os.path.exists(f"{ONNX_DIR}/{name}.onnx"):
                    print(f"{model_path} already converted")
                else:
                    convert_model(model_path, (1, 3, 299, 299), name, inception_model)
                    print(f"successfully converted {model_path} to onnx format")

            # maxvit - this one i find baffling, failed to execute here for some reason; And even after successfully individual conversion
            # failed to extract model layout from file; for some reason it has o for its input dims

            # elif "maxvit" in f"{model_path}":
            #     print(f"{model_path}")
            #     name = "maxvit"
            #     max_model = models.maxvit_t()
            #     if os.path.exists(f"{ONNX_DIR}/{name}.onnx"):
            #         print(f"{model_path} already converted")
            #     else:
            #         convert_model(model_path, (1, 3, 224, 244), name, max_model)
            #         print(f"successfully converted {model_path} to onnx format")

            elif "regnet" in f"{model_path}":
                print(f"{model_path}")
                name = "regnet_x_16gf"
                regnet_model = models.regnet_x_16gf()
                if os.path.exists(f"{ONNX_DIR}/{name}.onnx"):
                    print(f"{model_path} already converted")
                else:
                    convert_model(model_path, (1, 3, 224, 244), name, regnet_model)
                    print(f"successfully converted {model_path} to onnx format")

            # model require reduction along several dimensions - something that onnx2burn cannot handle

            elif "mnasnet" in f"{model_path}":
                print(f"{model_path}")
                name = "mnasnet0_5"
                mnas_model = models.mnasnet0_5()
                if os.path.exists(f"{ONNX_DIR}/{name}.onnx"):
                    print(f"{model_path} already converted")
                else:
                    convert_model(model_path, (1, 3, 224, 244), name, mnas_model)
                    print(f"successfully converted {model_path} to onnx format")

            elif "efficient" in f"{model_path}":
                print(f"{model_path}")
                name = "efficientnet_b0"
                eff_model = models.efficientnet_b0()
                if os.path.exists(f"{ONNX_DIR}/{name}.onnx"):
                    print(f"{model_path} already converted")
                else:
                    convert_model(model_path, (1, 3, 224, 244), name, eff_model)
                    print(f"successfully converted {model_path} to onnx format")

            # Yielded file format has discrepancies with what the onnx2burn tool is expecting - specifically an assertion fails

            elif "vit_b" in f"{model_path}":
                print(f"{model_path}")
                name = "vit_b_16"
                vit_model = models.vit_b_16()
                if os.path.exists(f"{ONNX_DIR}/{name}.onnx"):
                    print(f"{model_path} already converted")
                else:
                    convert_model(model_path, (1, 3, 224, 224), name, vit_model)
                    print(f"successfully converted {model_path} to onnx format")

    except Exception as e:
        raise e


# decorator for conversion
def convert_wrapper(func):
    def wrap(*args, **kwargs):
        model_path, in_tensor, name, model = func(*args, **kwargs)
        print(f"{model_path}")
        state = torch.load(model_path)
        dummy_input = torch.randn(in_tensor)
        model.load_state_dict(state)
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            f"{ONNX_DIR}/{name}.onnx",
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],  # Input tensor name
            output_names=["output"],  # Output tensor name
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },  # Allow dynamic batch size
        )

        subprocess.run(["onnx2burn", f"{ONNX_DIR}/{name}.onnx", ONNX_DIR])

    return wrap


@convert_wrapper
def convert_model(model_path, in_tensor, name, actual_model):
    return model_path, in_tensor, name, actual_model


if __name__ == "__main__":
    convert()
