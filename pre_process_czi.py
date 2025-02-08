import czifile
import cv2
import numpy as np
from matplotlib.widgets import Cursor
import csv
import matplotlib.pyplot as plt

ROOT_DIR = "/Users/atsushi/Downloads/count_cell_intensity"
DATA_DIR = ROOT_DIR + "/data/"
POSITION_DIR = ROOT_DIR + "/positions/"
DEFAULT_MARGIN = 1
file_names = ["Untitled189.czi", "Untitled197.czi"]
# 全部読み込むようにする。
# file_namesがあればそれのみにする。
MARGIN_DICT = {"Untitled189.czi": 1}

colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "yellow": (0, 255, 255),
    "gray": (128, 128, 128),
    "lightgray": (192, 192, 192),
    "orange": (0, 165, 255),
}
color_list = list(colors.values())
color_name_list = list(colors.keys())


def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def object_position(file_name):
    IMG_URL = DATA_DIR + file_name
    czi = czifile.CziFile(IMG_URL)
    image_data = czi.asarray()
    second_image = image_data[0, 1, :, :, 0]
    second_image = (second_image / np.max(second_image) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(second_image, (5, 5), 0)
    _, binary_black = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    contours_black, _ = cv2.findContours(
        binary_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = 50000
    filtered_black_contours = [
        cnt for cnt in contours_black if cv2.contourArea(cnt) > min_area
    ]

    if filtered_black_contours:
        largest_black_contour = max(filtered_black_contours, key=cv2.contourArea)

        # 黒い領域のマスクを作成
        black_mask = np.zeros_like(second_image)
        cv2.drawContours(
            black_mask, [largest_black_contour], -1, (255), thickness=cv2.FILLED
        )

        # 黒い領域内の白い部分を抽出
        white_in_black = cv2.bitwise_and(second_image, second_image, mask=black_mask)

        # 白いオブジェクトを検出
        _, binary_white = cv2.threshold(white_in_black, 50, 255, cv2.THRESH_BINARY)
        contours_white, _ = cv2.findContours(
            binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # **小さすぎるオブジェクトを削除**
        min_white_object_area = 500  # 小さなノイズを削除する閾値
        filtered_white_objects = [
            (x, y, w, h)
            for (x, y, w, h) in [cv2.boundingRect(cnt) for cnt in contours_white]
            if w * h > min_white_object_area
        ]

        results = []
        if filtered_white_objects:
            fig, ax = plt.subplots(figsize=(8, 8))
            curor = Cursor(ax, useblit=True, color="red", linewidth=1)
            ax.set_title(f"Detected Object in {file_name} (Brightness Increased)")
            ax.axis("off")
            bright_image = adjust_gamma(second_image, gamma=1.8)  # 表示用のガンマ補正
            image_with_objects = cv2.cvtColor(bright_image, cv2.COLOR_GRAY2BGR)

            for idx, white_object in enumerate(filtered_white_objects):
                margin = MARGIN_DICT.get(file_name, DEFAULT_MARGIN)
                x, y, w, h = white_object
                x1, y1 = max(0, x - margin), max(0, y - margin)  # 左上
                x1, y1 = max(0, x - margin), max(0, y - margin)  # 左上
                w_new, h_new = w + 2 * margin, h + 2 * margin  # 調整後の幅と高さ
                x4, y4 = x1 + w + 2 * margin, y1 + h + 2 * margin  # 右下

                centroid_x = (x1 + x4) // 2
                centroid_y = (y1 + y4) // 2

                cv2.rectangle(
                    image_with_objects, (x1, y1), (x4, y4), color_list[idx], 2
                )
                cv2.circle(
                    image_with_objects, (centroid_x, centroid_y), 5, (0, 0, 255), -1
                )

                results.append(
                    {
                        "file_name": file_name,
                        "x": x1,
                        "y": y1,
                        "w": w_new,
                        "h": h_new,
                        "centroid_x": centroid_x,
                        "centroid_y": centroid_y,
                        "color": color_name_list[idx],
                    }
                )
            circle_center = {
                "x": None,
                "y": None,
            }

            def on_click(event):
                if event.xdata is not None and event.ydata is not None:
                    circle_center["x"] = int(event.xdata)
                    circle_center["y"] = int(event.ydata)

            # クリックイベントを登録
            fig.canvas.mpl_connect("button_press_event", on_click)
            ax.imshow(image_with_objects, cmap="gray")
            print(
                "どの色を選択するのかメモして"
                + str(color_name_list[0 : len(filtered_white_objects)])
            )
            plt.savefig(POSITION_DIR + file_name + ".png")
            plt.show()
            for result in results:
                result["circle_center_x"] = circle_center["x"]
                result["circle_center_y"] = circle_center["y"]
            return results
    else:
        print(f"error: {file_name}")
        return None


positions = []
for file_name in file_names:
    position = object_position(file_name)
    if position:
        positions.extend(position)


# CSV ファイルに保存
csv_filename = POSITION_DIR + "positions.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = positions[0].keys()  # ヘッダーを取得
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # ヘッダーを書き込む
    writer.writerows(positions)  # リストの各辞書を1行ずつ書き込む

print(f"{csv_filename} に保存しました！")
