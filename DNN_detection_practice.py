import cv2

# haar cascade model
harr_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture('sample.mp4')

# play video
'''
while (cap.isOpened()):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
# save video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('new.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (680, 480))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 0)
        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

#fourcc = cv2.VideoWriter_fourcc('m', 'o', 'v', 'v')