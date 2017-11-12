import numpy as np
import time
import cv2


# Method: Used to slide across the image at a specified height
def sliding_window(img, step_size=20, win_size=(200, 200), y_height=0):
    img_height, img_width = int(img.shape[0]), int(img.shape[1])

    # Slide a window across the image
    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            # Yield the current window
            yield (x, y_height, image[y:y + win_size[1], x:x + win_size[0]])


# Method: Used to check if two sliding windows overlap
def is_overlap_between_windows(win_a, win_b, win_size):
    """
    :param win_a: Sliding window a
    :param win_b: Sliding window b
    :param win_size: Sliding window size
    :return: True if overlap, else False
    """
    if win_a[0] > (win_b[0] + win_size[0]) or win_b[0] > (win_a[0] + win_size[0]):
        return False
    if win_a[1] > (win_b[1] + win_size[1]) or win_b[1] > (win_a[1] + win_size[1]):
        return False
    return True


# Method: Used to return the region of overlap between sliding windows
def get_overlapping_region(win_a, win_b, win_size):
    """
    :param win_a: Sliding window a
    :param win_b: Sliding window b
    :param win_size: Sliding window size
    :return: Overlapping region
    """
    if not is_overlap_between_windows(win_a, win_b, win_size):
        return None, None, None

    x, y = max(win_a[0], win_b[0]), max(win_a[1], win_b[1])
    size = (min(win_a[0] + win_size[0], win_b[0] + win_size[0]) - x,
            min(win_a[1] + win_size[1], win_b[1] + win_size[1]) - y)
    return x, y, size


# File paths - GoogleNet
model_path = 'models/bvlc_googlenet.caffemodel'
image_path = 'images/fruit.jpg'
labels_path = 'synset_words.txt'
prototxt_path = 'models/bvlc_googlenet.prototxt'
image_data = [[]]
prediction_labels, prediction_confs, prediction_winds = [], [], []

# Sliding window variables
step_size = 20
win_size = (120, 120)
y_height = 0

# Load labels
rows = open(labels_path).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
print('[INFO]: Labels loaded')

# Load image
image = cv2.imread(image_path)
image = image.copy()
print('[INFO]: Image loaded')

# Load model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
print('[INFO]: Model loaded')

# Slide across the image
for (x, y, window) in sliding_window(image, step_size=step_size, win_size=win_size, y_height=y_height):
    # If the window does not meet our desired window size, ignore it
    if window.shape[0] != win_size[0] or window.shape[1] != win_size[1]:
        continue

    # CNN required fixed dimensions for our input image (227x227)
    # Mean subtraction is performed to normalize the input (104, 117, 123)
    blob = cv2.dnn.blobFromImage(window, 1, (227, 227), (104, 117, 123))

    # Set the blob as an input to the network
    net.setInput(blob)

    # Perform a forward-pass to obtain our output classification
    predictions = net.forward()

    # Get top prediction
    predictions = predictions.reshape((1, len(classes)))
    top_prediction_index = int(np.argsort(predictions[0])[::-1][:1])

    # Prediction information for each window
    prediction_labels.append(classes[top_prediction_index])
    prediction_confs.append(predictions[0][top_prediction_index]*100)
    prediction_winds.append((x, y, win_size))

    # -- DELETE ME --
    # clone = image.copy()
    # cv2.rectangle(clone, (x, y), (x + win_size[0], y + win_size[1]), (0, 255, 0), 2)
    # cv2.imshow("Window", clone)
    # cv2.waitKey(1)
    # time.sleep(0.025)
    # ---------------

    if (x+win_size[0]+step_size) >= image.shape[1]:
        print('[INFO]: Finished')
        break


print(prediction_confs.index(max(prediction_confs)))
print(prediction_labels[16], prediction_confs[16])
