import re
from pathlib import Path
from shutil import move

out_bin = Path('output')
all_files: list[Path] = list(out_bin.glob("*.png"))

label = input("Enter label to assign: ")
print("Enter exclusion pattern, or ENTER for no pattern:")
exclusion_pattern = input("(?i): ") or r"(?!)"
matcher = re.compile(exclusion_pattern, re.IGNORECASE)

for file in all_files:
    if matcher.search(str(file)):
        continue
    out = file.with_stem(file.stem + "_" + label)
    move(file, out)
    print(file, '->', out)
