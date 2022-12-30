#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def apply_yolo_object_detection(image_to_process):
    """
    Распознавание и определение координат объектов на изображении
    :param image_to_process: исходное изображение
    :return: изображение с отмеченными объектами и подписями к ним
    """

    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    # Начало поиска объектов на изображении
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Выбор
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        # Для отладки рисуем объекты, входящие в нужные классы
        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_bounding_box(image_to_process,
                                                        class_index, box)

    final_image = draw_object_count(image_to_process, objects_count)
    return final_image


def draw_object_bounding_box(image_to_process, index, box):
    """
    Отрисовка границ объекта с подписями
    :param image_to_process: исходное изображение
    :param index: индекс класса объекта, определенного с помощью YOLO
    :param box: координаты местности вокруг объекта
    :return: изображение с отмеченными объектами
    """

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font,
                              font_size, color, width, cv2.LINE_AA)

    return final_image


def draw_object_count(image_to_process, objects_count):
    """
    Подпись количества найденных объектов на изображении
    :param image_to_process: исходное изображение
    :param objects_count: количество объектов желаемого класса
    :return: изображение с помеченным количеством найденных объектов
    """

    start = (10, 120)
    font_size = 1.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)

    # Вывод текста с обводкой (чтобы было видно при разном освещении картинки)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    return final_image


def start_image_object_detection(img_path):
    """
    Анализ изображения
    """

    try:
        # Применение методов распознавания объектов на изображении с помощью YOLO
        image = cv2.imread(img_path)
        image = apply_yolo_object_detection(image)

        # Отображение обработанного изображения на экране
        cv2.imshow("Image", image)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    # Загрузка весов YOLO из файлов и настройка сети
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg",
                                     "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Загрузка из файла классов объектов, которые YOLO может обнаружить
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    # Определение классов, которые будут приоритетными для поиска по изображению
    # Имена находятся в файле coco.names.txt

    image = input("Укажите путь к изображению: ")
    look_for = input("Что мы ищем: ").split(',')

    # Удаление пробелов
    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    start_image_object_detection(image)
