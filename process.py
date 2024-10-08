import shutil
from math import atan
from pathlib import Path
from typing import cast

import cv2 as cv
import cv2.typing
import numpy
import numpy as np
from cv2.typing import MatLike
from rich.console import Console

richconsole = Console(force_terminal=True)
rp = richconsole.print

INPUT_DIR = Path("media")
INPUTS = list(INPUT_DIR.glob("*.png"))
OUTPUT_DIR = Path("output")
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def stripesAlgorithm(fp: Path, display: bool):
    def imshow(name: str, image: MatLike):
        if display:
            cv.imshow(name, image)

    def out(name: str, image: MatLike):
        if not display:
            out_to = OUTPUT_DIR / fp.relative_to(INPUT_DIR)
            out_to = out_to.with_name(out_to.stem + f"_{name}.png")
            rp(f"[bright_cyan]    [bold]Output[/] {name}[/] [bright_black]to[/] [cyan]{out_to}[/]")
            cv.imwrite(str(out_to), image)

    def stop():
        if display:
            cv.destroyAllWindows()

    original = cv.imread(str(fp))
    compressed = cv.resize(original, (0, 0), fx=0.5, fy=0.5)
    hsv = cv.cvtColor(compressed, cv.COLOR_BGR2HSV)
    grey = cv.cvtColor(compressed, cv.COLOR_BGR2GRAY)

    min_len = 64
    lower_bound = 0
    upper_bound = 64

    set_a = grey >= lower_bound
    set_b = grey <= upper_bound
    both = np.bitwise_and(set_a, set_b)

    output_bin = np.zeros(grey.shape, np.uint8)

    for i, row in enumerate(both):
        active = False
        start_at = 0
        for j, pos in enumerate(row):
            if pos and not active:
                start_at = j
                active = True
            if not pos and active:
                active = False
                length = j - start_at
                if length > min_len:
                    output_bin[i, start_at:j] = [255] * length

    imshow('original', compressed)
    imshow('result', output_bin)
    out('result', output_bin)

    if display:
        while cv.waitKey(0) != 27: pass
    stop()


def process(fp: Path, display: bool):
    names = []

    def imshow(name: str, image: MatLike):
        if display:
            names.append(name)
            cv.imshow(name, image)

    def out(name: str, image: MatLike):
        if not display:
            out_to = OUTPUT_DIR / fp.relative_to(INPUT_DIR)
            out_to = out_to.with_name(out_to.stem + f"_{name}.png")
            rp(f"[bright_cyan]    [bold]Output[/] {name}[/] [bright_black]to[/] [cyan]{out_to}[/]")
            cv.imwrite(str(out_to), image)

    def stop():
        if display:
            cv.destroyAllWindows()

    rp(f'[bold green]Processing[/] [bright_green]{fp}[/]'
       f' [bright_black]in {"interactive" if display else "automatic"} mode[/]')
    original = cv.imread(str(fp))
    image = cv.resize(original, (0, 0), fx=0.5, fy=0.5)
    # imshow("Original", image)
    # Look for black bits with contrast?
    n = 48
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blackish = cv.inRange(gray, np.array([0]), np.array([64]))
    # noinspection PyTypeChecker
    larger = cv.dilate(blackish, None, iterations=1)
    # masked = cv.bitwise_and(image, image, mask=blackish)
    canned = cv.Canny(gray, 50, 200)
    canned_roi = cv.bitwise_and(canned, canned, mask=larger)
    annotated = image.copy()
    lines = cv.HoughLinesP(canned_roi, 10, np.pi / 180, 10, minLineLength=80, maxLineGap=10)
    drawn = 0
    if lines is not None:
        N = lines.shape[0]
        for i in range(N):
            x1 = cast(int, lines[i][0][0])
            y1 = cast(int, lines[i][0][1])
            x2 = cast(int, lines[i][0][2])
            y2 = cast(int, lines[i][0][3])
            if (x2 - x1) == 0:
                continue
            angle = atan((y2 - y1) / (x2 - x1))
            if abs(angle) > 0.174532925:
                continue
            drawn += 1
            cv.line(annotated, [x1, y1], [x2, y2], [255, 0, 0], 2)
    if drawn == 0:
        cv.putText(annotated, "no matches", (100, 100), 0, fontScale=2, color=(0, 0, 255), thickness=4)

    imshow("grayscale", gray)
    imshow("hsv", hsv)
    imshow("Edges", canned_roi)
    imshow("GrayscaleM", blackish)
    imshow("annotated", annotated)
    out("annotated", annotated)
    # out("mask", blackish)
    # out("edge", canned)
    if display:
        while cv.waitKey(0) != 27: pass
    rp()
    stop()


def main():
    bulk = True
    if bulk:
        for file in INPUTS:
            stripesAlgorithm(file, False)
    else:
        file = next(x for x in INPUTS if 'blue_v2' in x.stem)
        stripesAlgorithm(file, True)


if __name__ == '__main__':
    main()
