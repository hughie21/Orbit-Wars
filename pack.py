import zipfile
import os

FRAMEWORK = "./framework"
MODEL = "./model"

def get_files(path):
    files = []
    for entry in os.listdir(path):
        full_path = os.path.join(path,entry)
        if not os.path.isdir(full_path):
            files.append(full_path)
    return files

frameworks = get_files(FRAMEWORK)
models = get_files(MODEL)

with zipfile.ZipFile("submission.zip", mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in frameworks:
        zf.write(f)
    for m in models:
        zf.write(m)

    zf.write("./main.py")

