import csv
import numpy as np
from scipy.stats import norm  # ガウス分布のパーセンタイル計算用
import matplotlib.pyplot as plt
import os
from extract_data import SELECT_COLOR, DEFAULT_COLOR

ROOT_DIR = "/Users/iwakitakuma/count_cell_intensity"
# ROOT_DIR = "/Users/atsushi/Downloads/count_cell_intensity"
DATA_DIR = ROOT_DIR + "/data/"
POSITION_DIR = ROOT_DIR + "/positions/"
EXTRACTED_DIR = ROOT_DIR + "/extracted_data/"
RESULT_DIR = ROOT_DIR + "/results/"


csv_filename = POSITION_DIR + "positions.csv"
THRESHOLD = 70
NORM_PPF_P = 0.2

NUM_BINS = 25
with open(csv_filename, "r") as csvfile:
    reader = csv.DictReader(csvfile)  # 各行を辞書として読み込む
    data = [row for row in reader]  # リストに変換

data_dict = {
    d["file_name"]: d
    for d in data
    if d["file_name"] in SELECT_COLOR
    and d["color"] == SELECT_COLOR.get(d["file_name"], DEFAULT_COLOR)
}


def line_equation(start, end):
    """直線の方程式を求める (y = ax + b) または (垂直線の場合)"""
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:  # 垂直線の場合
        return None, x1  # 垂直線ならx = x1が直線
    else:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a, b


def perpendicular_intersection(x, y, a, b):
    """点 (x, y) から直線 y = ax + b への垂線の交点を求める"""
    if a is None:  # 直線が垂直の場合
        return b, y  # 交点は (b, y)
    else:
        x_p = (x + a * (y - b)) / (1 + a**2)
        y_p = a * x_p + b
        return x_p, y_p


def get_intersection_order(points, start, end, file_name):
    """各点から直線への垂線の交点を求め、start から end までの順番で並べる"""
    a, b = line_equation(start, end)
    _intersections = []

    for x, y, value in points:
        x_p, y_p = perpendicular_intersection(x, y, a, b)
        distance = np.sqrt((x_p - end[0]) ** 2 + (y_p - end[1]) ** 2)  # 直線上の距離
        _intersections.append((x_p, y_p, value, distance))

    _intersections.sort(key=lambda p: p[3], reverse=True)  # 距離でソート
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

    start_distance = intersections[0][3]  # 最初の交点の距離
    end_distance = intersections[-1][3]  # 最後の交点の距離
    bin_size = (end_distance - start_distance) / num_bins  # 区間の長さ

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
    with open(dna_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)  # 各行を辞書として読み込む
        dna_data = [
            {
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "Intensity": float(row["Intensity"]),
                "Key": row["X"] + row["Y"],
            }
            for row in reader
        ]  # リストに変換
    with open(target_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)  # 各行を辞書として読み込む
        target_data = [
            {
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "Intensity": float(row["Intensity"]),
                "Key": row["X"] + row["Y"],
            }
            for row in reader
        ]  # リストに変換

    threshold = THRESHOLD
    if not threshold:
        dna_data_intensity = np.array(
            [float(d["Intensity"]) for d in dna_data], dtype=np.float32
        )
        mean, std = np.mean(dna_data_intensity), np.std(dna_data_intensity)
        threshold = norm.ppf(NORM_PPF_P, mean, std)
        threshold = max(0, threshold)  # 負の値にならないように制限

    print("threshold: " + str(threshold))
    print("row dna data pixels: " + str(len(dna_data)))
    dna_data = [d for d in dna_data if float(d["Intensity"]) > threshold]
    dna_data_keys = set(d["Key"] for d in dna_data)
    print("row target data pixels: " + str(len(target_data)))
    print("filtered dna data pixels: " + str(len(dna_data)))
    target_data = [d for d in target_data if d["Key"] in dna_data_keys]
    print("filtered target data pixels: " + str(len(target_data)))
    target_keys = set(d["Key"] for d in target_data)
    missing_keys = dna_data_keys - target_keys
    for key in missing_keys:
        matching_dna = next(d for d in dna_data if d["Key"] == key)
        target_data.append(
            {
                "Key": key,
                "X": matching_dna["X"],
                "Y": matching_dna["Y"],
                "Intensity": 0,  # Intensityは0にする
            }
        )
    print("filled target data pixels: " + str(len(target_data)))

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
        t / d if d != 0 else 0  # 0除算を避けるため、dが0のときは0をセット
        for t, d in zip(target_average_values, dna_average_values)
    ]

    x_values = list(range(len(ratio_values)))  # X軸: 各区間のインデックス
    y_values = ratio_values  # Y軸: 計算された比率

    # グラフの描画
    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(y_values) * 1.1)
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", label="Ratio")

    # 軸ラベルとタイトル
    plt.legend()
    plt.grid(True)
    dir_path = RESULT_DIR + file_name + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(dir_path + file_name + "-ratio.png")

    prefix = file_name.rstrip(".czi")
    dna_results[prefix] = dna_average_values
    target_results[prefix] = target_average_values
    ratio_results[prefix] = ratio_values

    # グラフの描画
    x_values = list(range(len(dna_average_values)))  # X軸: 各区間のインデックス
    y_values = dna_average_values  # Y軸: 計算された比率
    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(y_values) * 1.1)
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", label="Ratio")

    # 軸ラベルとタイトル
    plt.legend()
    plt.grid(True)
    plt.savefig(dir_path + file_name + "-dna.png")

    x_values = list(range(len(target_average_values)))  # X軸: 各区間のインデックス
    y_values = target_average_values  # Y軸: 計算された比率
    plt.figure(figsize=(10, 5))
    plt.ylim(0, max(y_values) * 1.1)
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", label="Ratio")

    # 軸ラベルとタイトル
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
