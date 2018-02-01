# PantryPal

## Overview
_PantryPal_ is a Raspberry Pi based project which classifies fruit and vegetables (in a _Pantry_) using a simplified SqueezeNet CNN with a sliding window for multi-object classification in an image. The resulting image was uploaded to DropBox by the Raspberry Pi and a message containing the contents of the _Pantry_ was sent by the CNN (using PubNub) to a simple Android application. The message (containing the list of classified items) and image was displayed on an Android application which also allowed the user to define their weekly shopping list and compare that list to the list of classified fruit and vegatables from the image.

## Description
This project is the computation application. It downloads the image from DropBox taken by the Raspberry Pi (_PantryPi_), classifies the contents of the image using a simplified GoogleNet CNN with a sliding window for multi-object classification, and uploads the image with bounding boxes surrounding the detected food items along with sending a message containing a list of classified food items to PubNub for the Android application.

## Software Versions
- python    `3.6.1`
- cv2       `3.3.0`
- numpy     `1.2.1`
- dropbox   `8.4.0`
- json      `2.6.0`
- pubnub    `3.9.0`
