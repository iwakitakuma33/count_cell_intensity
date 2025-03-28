import csv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import cv2
from extract_data import SELECT_COLOR, DEFAULT_COLOR
import imageio
from copy import deepcopy


ROOT_DIR = "/Users/atsushi/Downloads/count_cell_intensity"
# ROOT_DIR = "/Users/iwakitakuma/count_cell_intensity"
POSITION_DIR = ROOT_DIR + "/positions/"
EXTRACTED_DIR = ROOT_DIR + "/extracted_data/"
THRESHOLD_DIR = ROOT_DIR + "/threshold/"


if not os.path.exists(POSITION_DIR):
    os.makedirs(POSITION_DIR)
if not os.path.exists(EXTRACTED_DIR):
    os.makedirs(EXTRACTED_DIR)
if not os.path.exists(THRESHOLD_DIR):
    os.makedirs(THRESHOLD_DIR)

csv_filename = POSITION_DIR + "positions.csv"
NORM_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


with open(csv_filename, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

data_dict = {
    d["file_name"]: d
    for d in data
    if d["color"] == SELECT_COLOR.get(d["file_name"], DEFAULT_COLOR)
}

print(data_dict)
dna_results = {}
target_results = {}
ratio_results = {}
for file_name, position in data_dict.items():
    print("--------------")
    print("start: " + file_name)
    dna_data = []
    target_data = []
    prefix = file_name.rstrip(".czi")

    input_dir = EXTRACTED_DIR + file_name + "/"
    dna_path = input_dir + file_name + "-dna.csv"
    if not os.path.exists(dna_path):
        print("not analyze " + file_name)
        continue

    with open(dna_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        dna_data = [
            {
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "Intensity": float(row["Intensity"]),
                "Key": row["X"] + row["Y"],
            }
            for row in reader
        ]

    output_dir = THRESHOLD_DIR + file_name + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_path = input_dir + prefix + "-dna.png"
    output_path = output_dir + prefix + "-dna-gray.png"

    image = imageio.v3.imread(input_path, mode="L")
    plt.imsave(output_path, image, cmap="gray", format="png")

    dna_data_intensity = np.array(
        [float(d["Intensity"]) for d in dna_data], dtype=np.float32
    )
    mean, std = np.mean(dna_data_intensity), np.std(dna_data_intensity)

    metadata_list = ["color: " + position["color"]]
    for norm_ppf in NORM_LIST:
        _dna_data = deepcopy(dna_data)
        g_image = imageio.v3.imread(output_path)
        g_image = g_image.astype(np.uint8)
        threshold = norm.ppf(norm_ppf, mean, std)
        threshold = max(0, threshold)

        metadata = ["=============="]
        metadata.append("norm ppf p: " + str(norm_ppf))
        metadata.append("threshold: " + str(threshold))
        metadata_list.append("\n".join(metadata))
        _dna_data = [d for d in _dna_data if int(d["Intensity"]) > threshold]
        dna_x_y = [(int(d["X"]), int(d["Y"])) for d in _dna_data]
        max_x = max(x for x, y in dna_x_y) + 1
        max_y = max(y for x, y in dna_x_y) + 1
        _b_img = np.zeros((max_y, max_x), dtype=np.uint8)
        for x, y in dna_x_y:
            _b_img[y, x] = 255
        contours, _ = cv2.findContours(
            _b_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outer_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(_b_img)
        cv2.drawContours(mask, [outer_contour], -1, 255, thickness=cv2.FILLED)
        inner_pixels = np.argwhere(mask == 255)
        outer_contour_points = [(x, y) for x, y in outer_contour[:, 0, :]]
        inner_pixels = [(x, y) for y, x in inner_pixels]
        all_pixels = outer_contour_points + inner_pixels
        for x, y in all_pixels:
            if 0 <= y < g_image.shape[0] and 0 <= x < g_image.shape[1]:
                g_image[y, x] = [0, 220, 0, 20]

        _output_path = output_dir + prefix + "-dna-" + str(norm_ppf) + ".png"
        imageio.v3.imwrite(_output_path, g_image)

    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    x, y = max(0, int(position["x"])), max(0, int(position["y"]))
    w, h = (
        min(width - int(position["x"]), int(position["w"])),
        min(height - int(position["y"]), int(position["h"])),
    )
    roi = image[y : y + h, x : x + w]
    roi_thre, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thre, b_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_pixels = np.column_stack(np.where(b_img == 255))
    filtered_pixels = bright_pixels

    _dna_data = [d for d in dna_data if int(d["Intensity"]) > roi_thre]
    dna_x_y = [(int(d["X"]), int(d["Y"])) for d in _dna_data]
    g_image = imageio.v3.imread(output_path)
    for x, y in dna_x_y:
        if 0 <= y < g_image.shape[0] and 0 <= x < g_image.shape[1]:
            g_image[y, x] = [0, 220, 0, 20]

    _output_path = output_dir + prefix + "-dna-otsu-" + str(thre) + ".png"
    plt.imsave(_output_path, g_image, cmap="gray", format="png")
    txt_filename = output_dir + position["color"] + ".csv"
    with open(txt_filename, mode="w") as f:
        f.write("\n".join(metadata_list))
