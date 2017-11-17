import cv2
import time
import numpy as np
import Dropbox as db


# Method: Used to slide across the image at a specified height
def sliding_window(img, step_size=20, win_size=(200, 200)):
    # Slide a window across the image
    for y in range(0, img.shape[0], step_size):
        for x in range(0, img.shape[1], step_size):
            # Yield the current window
            yield (x, y, img[y:y + win_size[1], x:x + win_size[0]])


# Method: Used to find the highest confidences for each unique class prediction
def find_max_confidence_for_unique_labels(labels, confidences):
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
def draw_boxes_for_unique_labels(img, top_confidences, confidences, labels, windows):
    for i, confidence in enumerate(confidences):
        for j, top_confidence in enumerate(top_confidences):
            if confidence == top_confidence:
                # Draw boxes (with text) around top predictions
                wind = windows[i]
                x, y, size = wind[0], wind[1], wind[2]
                cv2.rectangle(img=img, pt1=(x, y), pt2=(x+size[0], y+size[1]), color=(0, 255, 0))
                cv2.putText(img=img, text='{}: {:.2f}'.format(labels[i], float(confidences[i])), org=(x, y),
                            color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)

    cv2.imwrite("image.jpg", img)


# Method: Used to send a string with the predictions to PubNub
def send_message_to_pubnub(top_labels, top_confs):
    pubnub_msg = ''

    # Generate PubNub message
    for i in range(len(top_labels)):
        pubnub_msg += str(top_labels[i]) + '_' + str('{0:.2f}%'.format(float(top_confs[i]))) + '\n'

    # TODO: Figure out how to send the string to PubNub
    print(pubnub_msg)


# Start time
start_time = time.time()

# File paths - GoogleNet
model_path = 'models/bvlc_googlenet.caffemodel'
image_path = 'images/image4.jpg'
labels_path = 'image_net_labels.txt'
prototxt_path = 'models/bvlc_googlenet.prototxt'

# Other variables
prediction_labels, prediction_confs, prediction_winds = [], [], []
display = True

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

for win_size in [(150, 150), (225, 225), (300, 300)]:  # Sorry, not sorry! (far from ideal)
    for (x, y, window) in sliding_window(image, step_size=100, win_size=win_size):
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
        top_prediction_index = int(np.argsort(predictions[0])[::-1][:1])

        # Only save prediction information if prediction confidence is > 60%
        if predictions[0][top_prediction_index] > 0.6:
            prediction_labels.append(classes[top_prediction_index])
            prediction_confs.append(predictions[0][top_prediction_index] * 100)
            prediction_winds.append((x, y, win_size))

        # Decide to display sliding window illustration or not
        if display:
            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + win_size[0], y + win_size[1]), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)

# Get the labels to be sent to
msg_labels, msg_confs = find_max_confidence_for_unique_labels(prediction_labels, prediction_confs)

# Draw boxes for top predictions on the image
draw_boxes_for_unique_labels(img=image,
                             top_confidences=msg_confs,
                             confidences=prediction_confs,
                             labels=prediction_labels,
                             windows=prediction_winds)

# Upload classified file to DropBox
db.upload_files(local_dir='output/',
                dbox_dir='/')

# Send prediction to pubnub
send_message_to_pubnub(top_labels=msg_labels,
                       top_confs=msg_confs)

# Calculate total execution time
total_time = time.time() - start_time
print('Execution time: {0:.2f}s'.format(total_time))
