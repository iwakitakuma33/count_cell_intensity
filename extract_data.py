from loci.plugins import BF
from ij import IJ
from ij.gui import Roi
from loci.plugins.in import ImporterOptions
import csv
import os


ROOT_DIR = "/Users/atsushi/Downloads/count_cell_intensity"
DATA_DIR = ROOT_DIR + "/data/"
POSITION_DIR = ROOT_DIR + "/positions/"
EXTRACTED_DIR = ROOT_DIR + "/extracted_data/"

csv_filename = POSITION_DIR + "positions.csv"

SELECT_COLOR = {
    "Untitled189.czi": "green",
    "Untitled197.czi": "green"
}
with open(csv_filename, "r") as csvfile:
    reader = csv.DictReader(csvfile)  # 各行を辞書として読み込む
    data = [row for row in reader]  # リストに変換

data_dict = {d["file_name"]: d for d in data if d["file_name"] in SELECT_COLOR and d['color'] == SELECT_COLOR[d["file_name"]]}

for file_name, position in data_dict.items():
    IMG_URL = DATA_DIR + file_name
    def extract(imp,name):
        save_path = os.path.join(EXTRACTED_DIR, file_name.replace(".czi", "-" + name + ".png"))
        IJ.saveAs(imp, "PNG", save_path)
        IJ.run(imp, "8-bit", "")

        x1, y1 = int(position['x']),  int(position['y'])
        width, height = int(position['w']),  int(position['h'])

        imp.setRoi(x1, y1, width, height)

        ip = imp.getProcessor()
        pixels = []

        for y in range(y1, y1 + height):
            for x in range(x1, x1 + width):
                intensity = ip.getPixel(x, y)
                pixels.append((x, y, intensity))

        output_path = EXTRACTED_DIR + file_name + "-" + name + ".csv"
        with open(output_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["X", "Y", "Intensity"])  # ヘッダー行
            writer.writerows(pixels)  # データを書き込み

    options = ImporterOptions()
    options.setId(IMG_URL)
    options.setSplitChannels(True)  # チャンネルを分ける
    options.setColorMode(ImporterOptions.COLOR_MODE_DEFAULT)  # 各チャンネルを個別に取得
    imps = BF.openImagePlus(options)  # チャンネルごとに ImagePlus を取得
    print(len(imps))
    extract(imps[0], "target")
    extract(imps[1], "dna")
print("End extract")
