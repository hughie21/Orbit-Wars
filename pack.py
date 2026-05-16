import zipfile
import os
import model
import argparse

FRAMEWORK = "./framework"
MODEL = "./model"
MODEL_WEIGHTS_ARCNAME = "model/ppo_weights.pt"


def get_files(path):
    files = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if not os.path.isdir(full_path):
            files.append(full_path)
    return files


def make_main(model_name: str, model_path: str | None = None) -> str:
    """Generate main.py content for the submission zip."""
    if model_name == "ppo_agent":
        weight_line = ""
        if model_path:
            weight_line = f'set_model_path("{model_path}")\n'
        return (
            "from model.ppo_agent import set_model_path, ppo_agent\n"
            "\n"
            f"{weight_line}"
            "\n"
            "def agent(obs, config=None):\n"
            "    return ppo_agent(obs, config)\n"
        )
    else:
        return (
            f"from model import {model_name}\n"
            "\n"
            "def agent(obs, config=None):\n"
            f"    return {model_name}(obs, config)\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Pack up the submission")
    parser.add_argument("model", choices=model.__all__, help="Select a model to export")
    parser.add_argument("--output", "-o", default="submission.zip", help="output file")
    parser.add_argument(
        "--model_path", "-m", default=None,
        help="Trained model weights file (.pt) to include in the submission "
             "(required for ppo_agent)",
    )

    args = parser.parse_args()

    # Validate: PPO requires a model weights file
    if args.model == "ppo_agent" and args.model_path is None:
        print("Error: ppo_agent requires --model_path to point to a trained .pt file")
        exit(1)

    frameworks = get_files(FRAMEWORK)
    models = get_files(MODEL)

    main_content = make_main(args.model, MODEL_WEIGHTS_ARCNAME)

    with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Framework files
        for f in frameworks:
            zf.write(f)
        # Model source files
        for m in models:
            zf.write(m)
        # Trained weights (included inside model/ directory)
        if args.model_path:
            zf.write(args.model_path, arcname=MODEL_WEIGHTS_ARCNAME)
        # Entry point
        zf.writestr("main.py", main_content)

    print(f"导出成功,文件已保存至：{args.output}")
    if args.model_path:
        print(f"  模型权重: {args.model_path} -> {MODEL_WEIGHTS_ARCNAME}")


if __name__ == "__main__":
    main()
