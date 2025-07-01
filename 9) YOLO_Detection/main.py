import cv2
import numpy as np

def load_data_and_preprocess(path):
    """
    Loads an image from the given path and preprocesses it for YOLO model input.

    Args:
        path (str): Path to the input image.

    Returns:
        tuple: Original image, preprocessed image blob, image height, and image width.
    """
    img = cv2.imread(path)
    h, w = img.shape[:2]
    preprocess_img = cv2.dnn.blobFromImage(
        img,
        scalefactor=1/255.0,
        size=(416, 416),
        swapRB=True,
        crop=False
    )
    return img, preprocess_img, h, w


def read_models_labels(labelPath):
    """
    Loads YOLO model and class labels.

    Args:
        labelPath (str): Path to the file containing class names (e.g., coco.names).

    Returns:
        tuple: Loaded YOLO network and list of class labels.
    """
    weightPath = r"yolo files\yolov3.weights"
    configPath = r"yolo files\yolov3.cfg"
    net = cv2.dnn.readNet(weightPath, configPath)
    labels = open(labelPath).read().strip().split("\n")
    return net, labels


def inference(img, preprocess_img, h, w, net, labels):
    """
    Performs a forward pass through the YOLO network.

    Args:
        img (numpy.ndarray): Original image.
        preprocess_img (numpy.ndarray): Preprocessed blob for YOLO.
        h (int): Height of the image.
        w (int): Width of the image.
        net (cv2.dnn_Net): Loaded YOLO model.
        labels (list): List of class labels.

    Returns:
        list: Output predictions from specified YOLO output layers.
    """
    net.setInput(preprocess_img)
    output_layers = ["yolo_82", "yolo_94", "yolo_106"]  # YOLOv3 output layers
    predictions = net.forward(output_layers)
    return predictions


def post_processing(predictions, w, h):
    """
    Extracts bounding boxes, class IDs, and confidences from YOLO predictions.

    Args:
        predictions (list): YOLO output predictions.
        w (int): Image width.
        h (int): Image height.

    Returns:
        tuple: classIDs, confidences, and bounding boxes of detected objects.
    """
    classIDs = []
    confidences = []
    boxes = []

    for layer in predictions:
        for detected_object in layer:
            scores = detected_object[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.3:  # Minimum confidence threshold
                box = detected_object[:4] * np.array([w, h, w, h])  # Scale box to image size
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([x, y, int(width), int(height)])

    return classIDs, confidences, boxes


def show_result(img, classIDs, confidences, boxes):
    """
    Applies Non-Maximum Suppression and visualizes detection results on the image.

    Args:
        img (numpy.ndarray): Original image.
        classIDs (list): Detected class IDs.
        confidences (list): Confidence scores of detected objects.
        boxes (list): Bounding boxes for detected objects.
    """
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)

    for i in idxs.flatten():
        (x, y) = boxes[i][0], boxes[i][1]
        (w, h) = boxes[i][2], boxes[i][3]

        # Draw bounding box and label
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        text = "{} : {:.2f}".format(labels[classIDs[i]], confidences[i])
        cv2.putText(img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---- Main Execution ----
img, preprocess_img, h, w = load_data_and_preprocess("cat.jpg")
net, labels = read_models_labels(r"yolo files\coco.names")
predictions = inference(img, preprocess_img, h, w, net, labels)
classIDs, confidences, boxes = post_processing(predictions, w, h)
show_result(img, classIDs, confidences, boxes)