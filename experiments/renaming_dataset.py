import os
from translator.utils import pad

rename_dir = "to_rename"
for file in os.listdir(rename_dir):
    name, ext = file.split(".")
    name_split = name.split("_")
    idx = int(name_split[len(name_split) - 1])
    idx_padded = pad(idx)
    name_split[len(name_split) - 1] = idx_padded
    name = "_".join(name_split)
    new_file = ".".join([name, ext])
    os.rename(os.path.join(rename_dir, file), os.path.join(rename_dir, new_file))
