import cv2


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        if not ret or img is None:
            break

        red = cv2.Laplacian(img, cv2.CV_64F)

        cv2.imshow("origin", img)
        cv2.imshow("red", red)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
