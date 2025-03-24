import cv2
import csv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from extract_data import SELECT_COLOR, DEFAULT_COLOR

ROOT_DIR = "/Users/atsushi/Downloads/count_cell_intensity"
# ROOT_DIR = "/Users/iwakitakuma/count_cell_intensity"
DATA_DIR = ROOT_DIR + "/data/"
POSITION_DIR = ROOT_DIR + "/positions/"
EXTRACTED_DIR = ROOT_DIR + "/extracted_data/"
RESULT_DIR = ROOT_DIR + "/circle_results/"

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


NUM_BINS = 12
with open(csv_filename, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

data_dict = {
    d["file_name"]: d
    for d in data
    if d["color"] == SELECT_COLOR.get(d["file_name"], DEFAULT_COLOR)
}


def is_point_in_sector_triangle(point, center, theta1, theta2, radius):
    """
    三角形（中心 + 2つの角）内に点が含まれるかをチェック
    """
    px, py = point
    cx, cy = center

    # 三角形の他の2点（角度→座標に変換）
    p1 = (cx + radius * np.cos(theta1), cy + radius * np.sin(theta1))
    p2 = (cx + radius * np.cos(theta2), cy + radius * np.sin(theta2))

    # ベクトルを定義
    def vec(a, b):
        return np.array([b[0] - a[0], b[1] - a[1]])

    # 外積の符号を使って判定
    v0 = vec(center, p1)
    v1 = vec(center, p2)
    v2 = vec(center, (px, py))

    cross1 = np.cross(v0, v2)
    cross2 = np.cross(v1, v2)

    # 点が扇の「扇形三角形」にあるためには：
    # v2 が v0 と v1 の間にある必要がある（クロス積の符号でわかる）
    cross_sector = np.cross(v0, v1)

    # cross1 と cross2 が cross_sector と同じ向きなら内側
    return np.sign(cross1) == np.sign(cross_sector) and np.sign(cross2) != np.sign(
        cross_sector
    )


def classify_points(points, center, ref_point, radius=100000):
    initial_angle = np.arctan2(
        ref_point[1] - center[1], ref_point[0] - center[0]
    ) - np.deg2rad(180 // NUM_BINS)

    sectors = [[] for _ in range(NUM_BINS)]
    point_sector_map = [[] for _ in range(NUM_BINS)]
    sector_angles = []

    for i in range(NUM_BINS):
        theta1 = initial_angle + np.deg2rad(i * 360 // NUM_BINS)
        theta2 = initial_angle + np.deg2rad((i + 1) * 360 // NUM_BINS)
        sectors[i] = []
        sector_angles.append((theta1, theta2))

    for x, y, v in points:
        for i, (theta1, theta2) in enumerate(sector_angles):
            if is_point_in_sector_triangle((x, y), center, theta1, theta2, radius):
                sectors[i].append(v)
                point_sector_map[i].append((x, y, v))
                break
    return point_sector_map, sectors, sector_angles


def zigzag_reorder(data):
    result = []
    left = 0
    right = len(data) - 1
    while left <= right:
        result.append(data[left])
        left += 1
        if left <= right:
            result.append(data[right])
            right -= 1
    return result


def compute_average_values(sectors):
    if not sectors:
        return []
    average_values = [np.mean(sector) if sector else 0 for sector in sectors]
    result = zigzag_reorder(average_values)
    return result


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
    _, dna_ordered_intersections, _ = classify_points(dna_data_values, start, end)
    dna_average_values = compute_average_values(dna_ordered_intersections)
    target_data_values = [
        (int(d["X"]), int(d["Y"]), int(d["Intensity"])) for d in target_data
    ]
    _, target_ordered_intersections, _ = classify_points(target_data_values, start, end)
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
