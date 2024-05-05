#!/usr/bin/env python3
import random
import copy
import io
import math
import os

from pprint import pprint
from glob import glob
import xml.etree.ElementTree as ET

import numpy as np
import cairosvg
from svgpathtools import svg2paths2, wsvg
from PIL import Image
from tqdm import tqdm

NAMESPACES = {
    "svg": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
}
C_WIDTH_MM = 17.5
C_HEIGHT_MM = 17.5
SYMBOLS = [
    ("A", "A", "Ubuntu"),
    ("B", "B", "Ubuntu"),
    ("C", "C", "Ubuntu"),
    ("D", "D", "Ubuntu"),
    ("E", "E", "Ubuntu"),
    ("F", "F", "Ubuntu"),
    ("sc", ";:", "Ubuntu"),
    ("inf", "∞", "Ubuntu"),
    ("sum", "∑", "Ubuntu"),
    ("angle", "∢", "Ubuntu"),
    ("root", "√", "Ubuntu"),
    ("plus", "+", "Ubuntu"),
    ("subset", "⊆", "Ubuntu"),
]


def read_svg(svg_file, layer=None):
    # Parse the original SVG file into an ElementTree
    tree = ET.parse(svg_file)
    root = tree.getroot()

    if layer is not None:
        for g_element in root.findall(
            ".//svg:g[@inkscape:groupmode='layer']", NAMESPACES
        ):
            if g_element.get(f"{{{NAMESPACES['inkscape']}}}label") != layer:
                root.remove(g_element)

    return tree


def read_letter(svg_file, letter, font):
    tree = read_svg(svg_file)
    root = tree.getroot()
    text = root.find(".//svg:tspan", NAMESPACES)
    text.text = letter
    return tree


def filter_svg(tree, path_id_to_keep):
    # Make a deep copy of the original tree
    new_tree = copy.deepcopy(tree)
    new_root = new_tree.getroot()

    paths_to_remove = [
        path
        for path in new_root.findall(".//svg:path", NAMESPACES)
        if path.get("id") not in path_id_to_keep
    ]

    for path in paths_to_remove:
        new_root.remove(path)

    return new_tree


def split_svg(tree, output_width):
    root = tree.getroot()

    # Create a list of all paths within this group
    paths = root.findall(".//svg:path", NAMESPACES)
    path_ids = [path.get("id") for path in paths if path.get("id") is not None]

    # Iterate over each path
    for path_id in path_ids:
        new_tree = filter_svg(tree, path_id)

        svg_string = ET.tostring(new_tree.getroot(), encoding="unicode")

        png_data = cairosvg.svg2png(
            bytestring=svg_string.encode("utf-8"), output_width=output_width
        )

        png_stream = io.BytesIO(png_data)

        image = Image.open(png_stream)
        white_background = Image.new("RGBA", image.size, "WHITE")
        image = Image.alpha_composite(white_background, image).convert("L")

        bitmap = (255 - np.array(image)) / 255

        yield (path_id, bitmap)


def svgtree_to_bitmap(tree, output_width=1024):
    svg_string = ET.tostring(tree.getroot(), encoding="unicode")

    png_data = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"), output_width=output_width
    )

    png_stream = io.BytesIO(png_data)

    image = Image.open(png_stream)
    white_background = Image.new("RGBA", image.size, "WHITE")
    image = Image.alpha_composite(white_background, image).convert("L")

    bitmap = (255 - np.array(image)) / 255

    return bitmap


def fill_paths(tree, path_ids):
    tree = copy.deepcopy(tree)
    root = tree.getroot()

    path_ids = set(path_ids)
    for path in root.findall(".//svg:path", NAMESPACES):
        path.set("stroke", "#bbbbbb")
        if path.get("id") in path_ids:
            path.set("fill", "#bbbbbb")
            path.set("fill-opacity", "1")
        else:
            path.set("fill", "none")

    return tree


def get_translate_coordinates(svg_root):
    """
    Parses the SVG ElementTree to find the first <g> element with a 'transform'
    attribute and extracts the translation coordinates from it.

    :param svg_root: An ElementTree root of an SVG document
    :return: A tuple of (x, y) coordinates or None if not found
    """
    g_element = svg_root.find(".//svg:g", NAMESPACES)
    transform = g_element.get("transform")
    if transform and "translate" in transform:
        # Attempt to extract the translation coordinates
        try:
            # Extracting the numbers from the string
            prefix = "translate("
            suffix = ")"
            start = transform.find(prefix) + len(prefix)
            end = transform.find(suffix, start)
            numbers = transform[start:end].split(",")
            x = float(numbers[0].strip())
            y = (
                float(numbers[1].strip()) if len(numbers) > 1 else 0
            )  # Default y to 0 if not provided
            return (x, y)
        except (IndexError, ValueError) as e:
            print(f"Error parsing transform attribute: {e}")
            return None
    # Return None if no appropriate <g> element is found
    return None


def get_square(paths, attributes, svg_attributes, outputfn):
    attributes = attributes.copy()
    svg_attributes = svg_attributes.copy()
    doc_xmin, doc_ymin, doc_width, doc_height = (
        float(v) for v in svg_attributes["viewBox"].split()
    )
    doc_width_mm = svg_attributes["width"]
    doc_height_mm = svg_attributes["height"]
    if not doc_width_mm.endswith("mm") or not doc_height_mm.endswith("mm"):
        raise ValueError("document dimensions not in mm")
    px_width = float(doc_width_mm[:-2]) / doc_width
    px_height = float(doc_height_mm[:-2]) / doc_height
    c_width = C_WIDTH_MM / px_width
    c_height = C_HEIGHT_MM / px_height
    box_xmin = random.uniform(doc_xmin, doc_xmin + doc_width - c_width)
    box_xmax = box_xmin + c_width
    box_ymin = random.uniform(doc_ymin, doc_ymin + doc_height - c_height)
    box_ymax = box_ymin + c_height
    to_export_paths = []
    to_export_attributes = []
    for path, a in zip(paths, attributes):
        bbox = path.bbox()
        path_xmin, path_xmax, path_ymin, path_ymax = path.bbox()
        if (not path_xmax < box_xmin and not path_xmin > box_xmax) and (
            not path_ymax < box_ymin and not path_ymin > box_ymax
        ):
            to_export_paths.append(path)
            to_export_attributes.append(a)
    svg_attributes["viewBox"] = " ".join(
        str(x) for x in [box_xmin, box_ymin, c_width, c_height]
    )
    svg_attributes["width"] = f"{C_WIDTH_MM}mm"
    svg_attributes["height"] = f"{C_HEIGHT_MM}mm"
    # print()
    # print(len(to_export_paths), len(to_export_attributes))
    wsvg(
        to_export_paths,
        attributes=to_export_attributes,
        svg_attributes=svg_attributes,
        filename=outputfn,
    )


def filled(bitmap):
    return np.sum(bitmap >= 0.5) / bitmap.size

def cleanup():
    for f in glob("out/*.svg") + glob("out/preview/*.png"):
        os.remove(f)

if __name__ == "__main__":
    cleanup()
    random.seed(123)
    w = 1024
    letters = {
        nm: {
            "bitmap": svgtree_to_bitmap(read_letter("a.svg", sym, font), w),
            "sym": sym,
        }
        for nm, sym, font in SYMBOLS
    }
    # letters_fill = [filled(x['bitmap']) for x in letters.values()]
    # print("letter fill between", min(letters_fill), "-", max(letters_fill))
    fn = "square_pattern_stroke.svg"
    translate = get_translate_coordinates(read_svg(fn))
    paths, attributes, svg_attributes = svg2paths2(fn)
    for i in range(len(attributes)):
        attributes[i].update(
            {
                "fill": "#000000",
                "stroke": "#000000",
                "stroke-width": "0.271926",
                "stroke-linejoin": "round",
            }
        )
        del attributes[i]["style"]
    paths = [path.translated(complex(translate[0], translate[1])) for path in paths]
    for i in tqdm(list(range(5)), desc="outer", position=0):
        square_fn = f"out/square{i}.svg"
        get_square(paths, attributes, svg_attributes, square_fn)
        square = read_svg(square_fn)
        tiles = dict(split_svg(square, w))
        # tiles_fill = [filled(b) for b in tiles.values()]
        # print()
        # print("files fill between", min(tiles_fill), "-", max(tiles_fill))
        for letter in tqdm(letters, desc="inner", position=1):
            keep_ids = []
            score = 0.0
            for tile in tiles:
                polygon_pixels = np.sum(tiles[tile] >= 0.5)
                if polygon_pixels == 0:
                    continue
                shared_pixels = np.sum(
                    np.logical_and(tiles[tile] >= 0.5, letters[letter]["bitmap"] >= 0.5)
                )
                overlap_ratio = shared_pixels / polygon_pixels
                if overlap_ratio > 0.5:
                    keep_ids.append(tile)
                    score += overlap_ratio
                else:
                    score -= overlap_ratio
            filled = fill_paths(square, keep_ids)
            filled.write(
                f"out/{letter}_{i}_{score:.0f}.svg",
                encoding="utf-8",
                xml_declaration=True,
            )
