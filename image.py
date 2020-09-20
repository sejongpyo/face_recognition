import cv2

images_paths = {
    'john' : 'images/john.jpg',
    'gabriel' : 'images/gabriel.jpg',
    'lucifer' : 'images/lucifer.jpg'
}

# insert records
descs = {
    'john' : None,
    'gabriel' : None,
    'lucifer' : None
}

for name, images_path in images_paths.items():
    image_bgr = cv2.imread(images_path) # load image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # convert bgr to rgb which is colored image

    image_shapes = 