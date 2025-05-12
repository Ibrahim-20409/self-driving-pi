import cv2
import numpy as np
import time




def blackenEdge(or_frame):
    gray = cv2.cvtColor(or_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(blurred)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=5)
    inverted = 255 - dilated

    h, w = inverted.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    floodfilled = inverted.copy()
    cv2.floodFill(floodfilled, mask, (0, 0), 255)

    floodfilled_inv = 255 - floodfilled
    filled_objects = cv2.bitwise_or(dilated, floodfilled_inv)

    result = np.full_like(or_frame, 255)
    result[filled_objects > 0] = 0
    return result

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def warpImg(img, points, width, height, inverse=False):
    src = np.float32(points)
    dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    if inverse:
        matrix = cv2.getPerspectiveTransform(dst, src)
    else:
        matrix = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, matrix, (width, height))



def getHistogram(img, show_histogram=False, min_threshold=0.1, scale=4):
    column_brightness = np.sum(img, axis=0)
    max_brightness = np.max(column_brightness)
    min_brightness = min_threshold * max_brightness
    bright_columns = np.where(column_brightness >= min_brightness)

    if bright_columns[0].size > 0:
        center_x = int(np.average(bright_columns))
    else:
        center_x = img.shape[1] // 2

    if show_histogram:
        hist_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        for x, brightness in enumerate(column_brightness):
            color = (255, 0, 255) if brightness > min_brightness else (0, 0, 255)
            bar_height = brightness // 255 // scale
            cv2.line(hist_img, (x, img.shape[0]), (x, img.shape[0] - bar_height), color, 1)

        cv2.circle(hist_img, (center_x, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        hist_img = cv2.resize(hist_img, (320, 320))
        cv2.imshow('Histogram', hist_img)

    return center_x


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0]) if isinstance(imgArray[0], list) else 1

    if isinstance(imgArray[0], list):
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]

        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (width, height))
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        combined_rows = [np.hstack(row) for row in imgArray]
        final_image = np.vstack(combined_rows)
    else:
        width = imgArray[0].shape[1]
        height = imgArray[0].shape[0]

        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (width, height))
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        final_image = np.hstack(imgArray)

    return cv2.resize(final_image, (0, 0), fx=scale, fy=scale)

def processFrame(img, display=1):
    img = cv2.resize(img, (wT, hT))
    fullimage = warpImg(img, points, wT, hT)
    image = blackenEdge(fullimage)

    hsv = cv2.cvtColor(fullimage, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([179,30,255]))
    gray_mask = cv2.inRange(hsv, np.array([0,0,40]), np.array([179,50,200]))
    black_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([179,50,40]))

    combined_mask = cv2.bitwise_or(white_mask, gray_mask)
    combined_mask = cv2.bitwise_or(combined_mask, black_mask)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = cv2.bitwise_and(gray_image, combined_mask)
    imgWarp = result

    basePoint = getHistogram(imgWarp, show_histogram=True)
    curveRaw = basePoint - wT // 2
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    if curve ==-1 :
        direction = "STRAIGHT"
    elif curve< 5:
        direction = "LEFT"
    elif curve > 5:
        direction = "RIGHT"
    else:
        direction = "BACKWARD"

    if display:
        imgInvWarp = warpImg(imgWarp, points, wT, hT, inverse=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[:hT//3, :] = 0

        lane_color = np.zeros_like(img)
        lane_color[:] = (0, 255, 0)
        lane_color = cv2.bitwise_and(imgInvWarp, lane_color)

        final_img = cv2.addWeighted(img, 1, lane_color, 1, 0)
        cv2.putText(final_img, f"Curve: {curve}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(final_img, f"Direction: {direction}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.line(final_img, (wT//2, 450), (wT//2 + (curve*3), 450), (255,0,255), 5)

        if display == 2:
            stacked = stackImages(0.7, [[img, imgWarp, result], [imgInvWarp, lane_color, final_img]])
            cv2.imshow('ImageStack', stacked)
        else:
            cv2.imshow('Result', final_img)

    return curve



def processFrameWithFPS(img, display=1, ground_truth_curves=None):
    start_time = time.time()
    curve = processFrame(img, display=2)
    end_time = time.time()
    if ground_truth_curves is not None:
        accuracy = evaluateAccuracy(curve, ground_truth_curves)
        print(f"Accuracy: {accuracy:.2f}%")
    # Calculate FPS
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}")


    return curve

def evaluateAccuracy(detected_curve, ground_truth_curve):
    # Compare detected curve with ground truth
    if ground_truth_curve is not None:
        accuracy = 100 - abs(detected_curve - ground_truth_curve)  # Simple accuracy metric
    else:
        accuracy = 0  # No ground truth, no accuracy
    return accuracy


def simulateLightingChange(img, brightness_factor=1.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * brightness_factor  # Adjust brightness
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_bright

wT, hT = 640, 680
points = np.float32([[110, 208], [wT - 110, 208], [0, hT], [wT, hT]])
curveList = []
avgVal = 10

cap = cv2.VideoCapture(r'D:\6th Sem\DIP\project\PXL_20250325_044746327.TS.mp4')
frameCounter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameCounter += 1
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0

    # processFrame(frame, display=2)
    # frame = simulateLightingChange(frame, brightness_factor=2)  # Simulate brighter lighting
    processFrameWithFPS(frame, display=2,ground_truth_curves=15)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
