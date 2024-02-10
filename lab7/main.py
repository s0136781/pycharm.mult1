import csv
import pathlib
import cv2
import pytesseract
from pytesseract import Output
from difflib import SequenceMatcher
import statistics

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


# Построить абсолютный путь до файла относительно местоположения скрипта
def rel_path(rel_path):
    path = pathlib.Path(__file__).parent / rel_path
    return path

#создание пустой строки и словаря(результаты распознавания и метки изображения)
def test_recognition(rec_type, val_type, dataset_name, show_img=False):
    output_str = ""
    labels = {}
    images_count = 0
    correct_guesses = 0
    similarities = []

    with open(
        str(rel_path(dataset_name + "/labels.csv")), newline="", encoding="utf-8"
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="'")
        for row in reader:
            labels[row[0]] = row[1]

    img_files = list(
        pathlib.Path(str(rel_path(dataset_name + "/samples/"))).glob("*.jpg")
    )

    for img_file in img_files:
        img = cv2.imread(str(img_file.resolve()), 0)
        groud_truth = labels[img_file.name]

        if rec_type == "straight_recognition":
            result = pytesseract.image_to_string(img, lang="rus+eng")

        if rec_type == "boxes_recognition":
            h, w = img.shape
            boxes = pytesseract.image_to_boxes(img, lang="rus+eng")

            for box in boxes.splitlines():
                box_data = box.split(" ")
                cv2.rectangle(
                    img,
                    (int(box_data[1]), h - int(box_data[2])),
                    (int(box_data[3]), h - int(box_data[4])),
                    (0, 255, 0),
                    2,
                )

            result = "".join([sym_data.split(" ")[0] for sym_data in boxes.split("\n")])

        result = "".join(result.splitlines())

        output_str += f"{img_file.name} | {groud_truth} | {result}\n"

        if val_type == "binary_correct":
            if result.lower() == groud_truth.lower():
                correct_guesses += 1

        if val_type == "similarity":
            similarity = SequenceMatcher(
                None, groud_truth.lower(), result.lower()
            ).ratio()
            similarities.append(similarity)

        images_count += 1

        print(result)
        if show_img:
            cv2.imshow("capthca", img)
            cv2.waitKey()

    output_str += "\n"

    if val_type == "binary_correct":
        output_str += f"Статистика: угадано {correct_guesses} / {images_count} капч"

    if val_type == "similarity":
        output_str += (
            f"Статистика: средняя схожесть: {statistics.fmean(similarities) * 100}%"
        )

    with open(
        str(
            rel_path(
                "results_" + val_type + "_" + rec_type + "_" + dataset_name + ".txt"
            )
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(output_str)


def main():
    test_recognition(
        "straight_recognition", "binary_correct", "dataset", show_img=False
    )
    test_recognition("straight_recognition", "similarity", "dataset", show_img=False)
    test_recognition(
        "straight_recognition", "binary_correct", "dataset2", show_img=False
    )
    test_recognition("straight_recognition", "similarity", "dataset2", show_img=False)


if __name__ == "__main__":
    main()