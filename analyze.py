import cv2
import csv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from extract_data import SELECT_COLOR, DEFAULT_COLOR


ROOT_DIR = "/Users/iwakitakuma/count_cell_intensity"
# ROOT_DIR = "/Users/atsushi/Downloads/count_cell_intensity"
DATA_DIR = ROOT_DIR + "/data/"
POSITION_DIR = ROOT_DIR + "/positions/"
EXTRACTED_DIR = ROOT_DIR + "/extracted_data/"
RESULT_DIR = ROOT_DIR + "/results/"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(POSITION_DIR):
    os.makedirs(POSITION_DIR)
if not os.path.exists(EXTRACTED_DIR):
    os.makedirs(EXTRACTED_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

csv_filename = POSITION_DIR + "positions.csv"
NORM_PPF_P = 0.5
PPF_DICT = {"Untitled189.czi": 0.5}
# PPF_DICT = {"Untitled189.czi": 0.4}

NUM_BINS = 15
with open(csv_filename, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

data_dict = {
    d["file_name"]: d
    for d in data
    if d["color"] == SELECT_COLOR.get(d["file_name"], DEFAULT_COLOR)
}


def line_equation(start, end):
    """直線の方程式を求める (y = ax + b) または (垂直線の場合)"""
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        return None, x1
    else:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a, b


def perpendicular_intersection(x, y, a, b):
    if a is None:
        return b, y
    else:
        x_p = (x + a * (y - b)) / (1 + a**2)
        y_p = a * x_p + b
        return x_p, y_p


def get_intersection_order(points, start, end, file_name):
    a, b = line_equation(start, end)
    _intersections = []

    for x, y, value in points:
        x_p, y_p = perpendicular_intersection(x, y, a, b)
        distance = np.sqrt((x_p - end[0]) ** 2 + (y_p - end[1]) ** 2)
        _intersections.append((x_p, y_p, value, distance))

    _intersections.sort(key=lambda p: p[3], reverse=True)
    start = _intersections[0]
    intersections = []
    for intersection in _intersections:
        distance = np.sqrt(
            (intersection[0] - start[0]) ** 2 + (intersection[1] - start[1]) ** 2
        )
        intersections.append(
            (intersection[0], intersection[1], intersection[2], distance)
        )
    return intersections


def compute_average_values(intersections, num_bins=NUM_BINS):
    """直線を num_bins 個に分割し、それぞれの区間内の value の平均値を計算"""
    if not intersections:
        return []

    start_distance = intersections[0][3]
    end_distance = intersections[-1][3]
    bin_size = (end_distance - start_distance) / num_bins

    bins = [[] for _ in range(num_bins)]

    for x_p, y_p, value, distance in intersections:
        bin_index = min(int((distance - start_distance) / bin_size), num_bins - 1)
        bins[bin_index].append(value)
    average_values = [np.mean(bin) if bin else 0 for bin in bins]
    return average_values


dna_results = {}
target_results = {}
ratio_results = {}
for file_name, position in data_dict.items():
    print("--------------")
    print("start: " + file_name)
    dna_data = []
    target_data = []
    dir_path = EXTRACTED_DIR + file_name + "/"
    dna_path = dir_path + file_name + "-dna.csv"
    target_path = dir_path + file_name + "-target.csv"
    prefix = file_name.rstrip(".czi")
    if not os.path.exists(dna_path):
        print("not analyze " + file_name + " because not found csv file")
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
    with open(target_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        target_data = [
            {
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "Intensity": float(row["Intensity"]),
                "Key": row["X"] + row["Y"],
            }
            for row in reader
        ]

    dna_data_intensity = np.array(
        [float(d["Intensity"]) for d in dna_data], dtype=np.float32
    )
    mean, std = np.mean(dna_data_intensity), np.std(dna_data_intensity)
    threshold = norm.ppf(PPF_DICT.get(file_name, NORM_PPF_P), mean, std)
    threshold = max(0, threshold)

    print("threshold: " + str(threshold))
    print("row dna data pixels: " + str(len(dna_data)))
    print("row target data pixels: " + str(len(target_data)))
    dna_data = [d for d in dna_data if float(d["Intensity"]) > threshold]
    dna_x_y = [(int(d["X"]), int(d["Y"])) for d in dna_data]
    max_x = max(x for x, y in dna_x_y) + 1
    max_y = max(y for x, y in dna_x_y) + 1
    _b_img = np.zeros((max_y, max_x), dtype=np.uint8)
    for x, y in dna_x_y:
        _b_img[y, x] = 255
    contours, _ = cv2.findContours(_b_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(_b_img)
    cv2.drawContours(mask, [outer_contour], -1, 255, thickness=cv2.FILLED)
    inner_pixels = np.argwhere(mask == 255)
    outer_contour_points = [(x, y) for x, y in outer_contour[:, 0, :]]
    inner_pixels = [(x, y) for y, x in inner_pixels]
    all_pixels = outer_contour_points + inner_pixels
    all_pixels_dict = {
        str(x) + str(y): {
            "Key": str(x) + str(y),
            "X": x,
            "Y": y,
            "Intensity": 0,
        }
        for x, y in all_pixels
    }
    print("filtered dna all pixels: " + str(len(all_pixels_dict)))
    dna_data_keys = {str(x) + str(y): (x, y) for x, y in all_pixels}
    target_keys = {d["Key"]: d for d in target_data if d["Key"] in dna_data_keys}
    all_pixels_dict.update(target_keys)
    target_data = all_pixels_dict.values()
    print("filled target pixels: " + str(len(all_pixels_dict)))

    try:
        int(position["circle_center_x"])
        int(position["circle_center_y"])
    except:
        print("not analyze " + file_name + " because not selected circle center")
        continue
    start = (int(position["centroid_x"]), int(position["centroid_y"]))
    end = (int(position["circle_center_x"]), int(position["circle_center_y"]))

    dna_data_values = [
        (int(d["X"]), int(d["Y"]), int(d["Intensity"])) for d in dna_data
    ]
    dna_ordered_intersections = get_intersection_order(
        dna_data_values, start, end, file_name
    )
    dna_average_values = compute_average_values(dna_ordered_intersections)

    target_data_values = [
        (int(d["X"]), int(d["Y"]), int(d["Intensity"])) for d in target_data
    ]
    target_ordered_intersections = get_intersection_order(
        target_data_values, start, end, file_name
    )
    target_average_values = compute_average_values(target_ordered_intersections)

    ratio_values = [
        t / d if d != 0 else 0
        for t, d in zip(target_average_values, dna_average_values)
    ]

    x_values = list(range(len(ratio_values)))
    y_values = ratio_values

    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(y_values) * 1.1)
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", label="Ratio")

    plt.legend()
    plt.grid(True)
    dir_path = RESULT_DIR + file_name + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(dir_path + file_name + "-ratio.png")

    dna_results[prefix] = dna_average_values
    target_results[prefix] = target_average_values
    ratio_results[prefix] = ratio_values

    x_values = list(range(len(dna_average_values)))
    y_values = dna_average_values
    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(y_values) * 1.1)
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", label="Ratio")

    plt.legend()
    plt.grid(True)
    plt.savefig(dir_path + file_name + "-dna.png")

    x_values = list(range(len(target_average_values)))
    y_values = target_average_values
    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(y_values) * 1.1)
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", label="Ratio")

    plt.legend()
    plt.grid(True)
    plt.savefig(dir_path + file_name + "-target.png")
print("---------- end save imgs")

txt_filename = RESULT_DIR + "dna.csv"
with open(txt_filename, "w", encoding="utf-8") as txtfile:
    writer = csv.writer(txtfile)
    writer.writerows([[k, *v] for k, v in dna_results.items()])
txt_filename = RESULT_DIR + "target.csv"
with open(txt_filename, "w", encoding="utf-8") as txtfile:
    writer = csv.writer(txtfile)
    writer.writerows([[k, *v] for k, v in target_results.items()])

txt_filename = RESULT_DIR + "ratio.csv"
with open(txt_filename, "w", encoding="utf-8") as txtfile:
    writer = csv.writer(txtfile)
    writer.writerows([[k, *v] for k, v in ratio_results.items()])

print("end")
