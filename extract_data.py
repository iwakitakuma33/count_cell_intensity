ROOT_DIR = "/Users/atsushi/Downloads/count_cell_intensity"
# ROOT_DIR = "/Users/iwakitakuma/count_cell_intensity"
DATA_DIR = ROOT_DIR + "/data/"
POSITION_DIR = ROOT_DIR + "/positions/"
EXTRACTED_DIR = ROOT_DIR + "/extracted_data/"


csv_filename = POSITION_DIR + "positions.csv"

DEFAULT_COLOR = "green"
SELECT_COLOR = {
    "Untitled195.czi": "blue",
    "Untitled153.czi": "red",
    "Untitled225.czi": "red",
    "Untitled123.czi": "blue",
    "Untitled240.czi": "red",
    "Untitled297.czi": "red",
    "Untitled339.czi": "blue",
    "Untitled312.czi": "blue",
    "Untitled138.czi": "blue",
}
if __name__ == "__main__":
    try:
        from loci.plugins import BF
        from ij import IJ
        import importlib
    except ImportError as e:
        print(e)

    import csv
    import os

    with open(csv_filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]

    data_dict = {
        d["file_name"]: d
        for d in data
        if d["color"] == SELECT_COLOR.get(d["file_name"], DEFAULT_COLOR)
    }

    for file_name, position in data_dict.items():
        IMG_URL = DATA_DIR + file_name

        def extract(imp, name):
            dir_path = EXTRACTED_DIR + file_name + "/"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            save_path = os.path.join(
                dir_path, file_name.replace(".czi", "-" + name + ".png")
            )
            IJ.saveAs(imp, "PNG", save_path)
            IJ.run(imp, "8-bit", "")

            x1, y1 = int(position["x"]), int(position["y"])
            width, height = int(position["w"]), int(position["h"])

            imp.setRoi(x1, y1, width, height)

            ip = imp.getProcessor()
            pixels = []

            for y in range(y1, y1 + height):
                for x in range(x1, x1 + width):
                    intensity = ip.getPixel(x, y)
                    pixels.append((x, y, intensity))

            output_path = dir_path + file_name + "-" + name + ".csv"
            with open(output_path, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["X", "Y", "Intensity"])
                writer.writerows(pixels)

        ImporterOptions = importlib.import_module("loci.plugins.in").ImporterOptions
        options = ImporterOptions()
        options.setId(IMG_URL)
        options.setSplitChannels(True)
        options.setColorMode(ImporterOptions.COLOR_MODE_DEFAULT)
        imps = BF.openImagePlus(options)
        print(len(imps))
        extract(imps[0], "target")
        extract(imps[1], "dna")
    print("End extract")
