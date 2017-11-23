import os
import cv2
import time
import json
import numpy as np
import Dropbox as db
from string import capwords
from pubnub import Pubnub


# Method: Used to slide across the image at a specified height
def sliding_window(img, win_size, step_size=20, crop_size=(0, 0)):
    """
    :param img: Image file
    :param step_size: Step size
    :param win_size: Window size
    :param crop_size: Area of the image for the window to slide over
    :return: (x, y, window)
    """
    # Slide a window across the image
    for y in np.arange(crop_size[0], img.shape[0]-crop_size[0], step_size*2):
        for x in np.arange(crop_size[1], img.shape[1]-crop_size[1], step_size):
            # Yield the current window
            yield (x, y, img[y:y + win_size[1], x:x + win_size[0]])


# Method: Used to get a list of unique class predictions
def get_prediction_labels(labels, confidences):
    """
    :param labels: Predicted labels for each window
    :param confidences: Predicted confidences for each window
    :return: Top unique prediction labels
    """
    unique_labels = list(set(labels))
    unique_max_conf = np.zeros(len(unique_labels))

    for i, label in enumerate(labels):
        for j, unique_label in enumerate(unique_labels):
            if unique_label is label:
                # Check for highest conf
                if unique_max_conf[j] <= confidences[i]:
                    unique_max_conf[j] = confidences[i]
                    break

    return unique_labels, unique_max_conf


# Method: Used to draw boxes around classified food items
def draw_boxes_around_predictions(img, top_confidences, confidences, labels, windows):
    """
    :param img: Image file
    :param top_confidences: Top prediction confidences
    :param confidences: All prediction confidences
    :param labels: All prediction labels
    :param windows: All prediction windows
    :return: Output image
    """
    for i, confidence in enumerate(confidences):
        for j, top_confidence in enumerate(top_confidences):
            if confidence == top_confidence:
                # Draw boxes (with text) around top predictions
                wind = windows[i]
                x, y, size = wind[0], wind[1], wind[2]
                cv2.rectangle(img=img, pt1=(x, y), pt2=(x+size[0], y+size[1]), color=(0, 255, 0))
                cv2.putText(img=img, text='{}'.format(labels[i]), org=(x, y),
                            color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)

    return img


# Method: Used to send a string with the predictions to PubNub
def send_message_to_pubnub(top_labels):
    msg = ''

    if top_labels:
        # Generate PubNub message
        for i in np.arange(len(top_labels)):
            msg += top_labels[i] + ','
    else:
        msg = 'No items detected! :( Please double check the image!'

    json_msg = json.dumps(msg)
    channel = 'PantryPalToPhone'

    pubnub.publish(channel=channel, message=json_msg)


# Method: Used to get the image for classification
def get_image(img_path, capture=False):
    """
    :param img_path: Path to image if loading
    :param capture: True = Capture image, False = Load Image
    :return:
    """
    # Capture image from Raspberry Pi
    if capture:
        import picamera

        save_path = 'images/image.jpg'

        camera = picamera.PiCamera()
        camera.resolution = (1024, 768)
        time.sleep(2)
        camera.capture(save_path)
        image = cv2.imread(save_path)
    # Load image from disk
    else:
        image = cv2.imread(img_path)

    return image


# Method: Used to classify the contents of the captured image
def classification(message, channel):
    """
    :param message: Required for PubNub
    :param channel: Required for PubNub
    :return: Classified image for DropBox and message for PubNub
    """
    print('[INFO]: Classification starting....')
    image = get_image(img_path='images/16.jpg', capture=capture)
    # Start time
    start_time = time.time()

    for win_size in np.array([(220, 220), (280, 280), (360, 360), (420, 420)]):  # Sorry, not sorry!
        for (x, y, window) in sliding_window(image, win_size, step_size=60, crop_size=(50, 50)):
            # If the window does not meet our desired window size, ignore it
            if window.shape[0] != win_size[0] or window.shape[1] != win_size[1]:
                continue

            # GoogleNet CNN required fixed dimensions for our input image (224x224)
            # Mean subtraction is performed to normalize the input (104, 117, 123)
            blob = cv2.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))

            # Set the blob as an input to the network
            net.setInput(blob)

            # Perform a forward-pass to obtain our output classification
            predictions = net.forward()

            # Get top prediction
            predictions = predictions.reshape((1, len(classes)))
            best_index = int(np.argsort(predictions[0])[::-1][0])

            # Only save prediction information if prediction confidence is > 50% and it's 'NotFood'
            if classes[best_index] != 'Notfood' and predictions[0][best_index] >= 0.5:
                prediction_labels.append(classes[best_index])
                prediction_confs.append(predictions[0][best_index] * 100)
                prediction_winds.append((x, y, win_size))

            # Decide to display sliding window illustration or not
            if display:
                clone = image.copy()
                cv2.rectangle(clone, (x, y), (x + win_size[0], y + win_size[1]), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)

    # Calculate total classification time
    total_time = time.time() - start_time
    print('[INFO]: Classification time: {0:.2f}s'.format(total_time))

    # Get the labels to be sent to PubNub
    top_labels, top_confs = get_prediction_labels(prediction_labels, prediction_confs)

    # Draw boxes for top predictions on the image
    classified_image = draw_boxes_around_predictions(img=image,
                                                     top_confidences=top_confs,
                                                     confidences=prediction_confs,
                                                     labels=prediction_labels,
                                                     windows=prediction_winds)

    # Save image
    cv2.imwrite("output/pantry.jpg", classified_image)

    # Upload classified file to DropBox
    db.delete_files_from_dropbox_dir(dbox_dir='')
    db.upload_files(local_dir='C://PythonProjects/PantryPal/output/', dbox_dir='/')

    # Send prediction to PubNub
    send_message_to_pubnub(top_labels=top_labels)
    print('[INFO]: Message sent to PubNub')
    print('[INFO]: Ready to classify....')


# Connect to PubNub
pubnub = Pubnub(publish_key="pub-c-935c97ba-71d6-4dd1-b500-e1ea1f85e0a5",
                subscribe_key="sub-c-7ce45822-aff9-11e7-8f6d-3a18aff742a6")

# File paths - GoogleNet
model_path = 'models/bvlc_googlenet.caffemodel'
prototxt_path = 'models/bvlc_googlenet.prototxt'
labels_path = 'modified_image_net_labels.txt'

# Other variables
prediction_labels, prediction_confs, prediction_winds = [], [], []
display = False
capture = False

# Load labels
rows = open(labels_path).read().strip().split("\n")
classes = [capwords(r[r.find(" ") + 1:].split(",")[0].strip()) for r in rows]

# Load model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Subscribe to PubNub
pubnub.subscribe(channels="PhoneToPantryPal", callback=classification)
print('[INFO]: Subscribed to PubNub')
