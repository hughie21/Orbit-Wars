import zipfile
import os
import model
import argparse

FRAMEWORK = "./framework"
MODEL = "./model"
PPO_WEIGHTS_ARCNAME = "model/ppo_weights.pt"
STRATEGIC_WEIGHTS_ARCNAME = "model/strategic_weights.pkl"


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
    elif model_name == "strategic_agent":
        lines = [
            "from model.strategic_agent import StrategicAgent",
            "",
        ]
        if model_path:
            lines += [
                f'WEIGHTS_PATH = "{model_path}"',
                "_agent = None",
                "",
                "def agent(obs, config=None):",
                "    global _agent",
                "    if _agent is None:",
                "        _agent = StrategicAgent(player_id=obs.get('player', 0), enable_learning=True)",
                f"        _agent.load_weights(WEIGHTS_PATH)",
                "    return _agent.act(obs)",
            ]
        else:
            lines += [
                "from model.strategic_agent import strategic_agent",
                "",
                "def agent(obs, config=None):",
                "    return strategic_agent(obs, config)",
            ]
        return "\n".join(lines) + "\n"
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
        help="Trained model weights file to include in the submission "
             "(.pt for ppo_agent, .pkl for strategic_agent)",
    )

    args = parser.parse_args()

    # Validate weight format
    if args.model_path:
        if args.model == "ppo_agent" and not args.model_path.endswith(".pt"):
            print("Error: ppo_agent requires a .pt weights file")
            exit(1)
        if args.model == "strategic_agent" and not args.model_path.endswith(".pkl"):
            print("Error: strategic_agent requires a .pkl weights file")
            exit(1)

    # Determine arcname for weights inside the zip
    if args.model == "strategic_agent":
        weights_arcname = STRATEGIC_WEIGHTS_ARCNAME
    else:
        weights_arcname = PPO_WEIGHTS_ARCNAME

    frameworks = get_files(FRAMEWORK)
    models = get_files(MODEL)

    main_content = make_main(args.model, weights_arcname if args.model_path else None)

    with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Framework files
        for f in frameworks:
            zf.write(f)
        # Model source files
        for m in models:
            zf.write(m)
        # Trained weights (included inside model/ directory)
        if args.model_path:
            zf.write(args.model_path, arcname=weights_arcname)
        # Entry point
        zf.writestr("main.py", main_content)

    print(f"导出成功,文件已保存至：{args.output}")
    if args.model_path:
        print(f"  模型权重: {args.model_path} -> {weights_arcname}")


if __name__ == "__main__":
    main()
