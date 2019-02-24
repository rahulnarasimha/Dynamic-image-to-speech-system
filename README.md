# Dynamic-image-to-speech-system
A device to help the blind read non braille books. 
A finger-mounted device with a built-in camera which has the ability to convert written text into audio. 
As the user follows a line of text with their finger, the device reads it out.

First, the system looks to locate the fingertip as a cursor for finding characters, words and lines. 

After selecting the word closest to the finger, we crop it and if the text in the image is not straight we rotate the image by the required angle.

We select the contour closest to the finder tip for further processing:

We then reduce our region of focus by cropping he area above the finger. 
We then find contours of the image thus identifying all the words in the image. 
We consider the word closest to the fingertip for further processing.

After selecting the word closest to the finger, we crop it.

We crop the image and if the text in the image is not straight we rotate the image by the required angle.

We then binarize the image using Otsu’s thresholding. 

We then use pytesseract to extract the text from the image.

As the reader moves his finger the above process is performed repeatedly, thus enabling to read one word at a time.

