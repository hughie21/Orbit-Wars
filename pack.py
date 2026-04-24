import zipfile
import os
import model
import argparse

FRAMEWORK = "./framework"
MODEL = "./model"
MAINFILE_TEMPLATE = """
from model import {model_name}
_agent_instance = None

def agent(obs, config=None):
    global _agent_instance

    try:
        if hasattr(obs, 'player'):
            player = obs.player
        else:
            player = obs.get("player", 0)

        if _agent_instance is None:
            _agent_instance = {model_name}(player_id=player)
        elif _agent_instance.player_id != player:
            _agent_instance = {model_name}(player_id=player)

        return _agent_instance.compute_moves(obs)
    except Exception as e:
        # Return empty moves on any error
        return []
"""


def get_files(path):
    files = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if not os.path.isdir(full_path):
            files.append(full_path)
    return files


def main():
    parser = argparse.ArgumentParser(description="Pack up the submission")
    parser.add_argument("model", choices=model.__all__, help="Select a model to export")
    parser.add_argument("--output", "-o", default="submission.zip", help="output file")

    args = parser.parse_args()

    frameworks = get_files(FRAMEWORK)
    models = get_files(MODEL)

    main_content = MAINFILE_TEMPLATE.format(model_name=args.model)

    with zipfile.ZipFile(args.output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in frameworks:
            zf.write(f)
        for m in models:
            zf.write(m)

        zf.writestr("main.py", main_content)
    print(f"导出成功,文件已保存至：{args.output}")


if __name__ == "__main__":
    main()
