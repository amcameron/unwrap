#!/usr/bin/env python3
import argparse
import sys

import cv2
import numpy as np

def unwrap(cap: cv2.VideoCapture):

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    smoothed_theta = None
    smoothing_alpha = 0.4  # Smoothing factor (0 < alpha <= 1)
    # Read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("No frames found in video.")
        return
    h, w, c = frame.shape
    # Prepare output image: height x total_frames x 3
    output_columns = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

    while True:
        print(f"\rProcessing frame: {frame_number}/{total_frames}", end='', flush=True)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        edges = cv2.Canny(gray, 100, 255)

        # Hough transform to find lines
        lines_hough = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150, min_theta=-np.pi / 6, max_theta=np.pi / 6)
        median_theta = None
        if lines_hough is not None:
            thetas = [line[0][1] for line in lines_hough]
            if thetas:
                median_theta = np.median(thetas)

        # Compute smoothed theta
        if median_theta is not None:
            if smoothed_theta is None:
                smoothed_theta = median_theta
            else:
                delta = np.angle(np.exp(1j * (median_theta - smoothed_theta)))
                smoothed_theta = smoothed_theta + smoothing_alpha * delta

            # Sample pixels along the yellow line (smoothed_theta through center)
            a = np.cos(smoothed_theta)
            b = np.sin(smoothed_theta)
            x_c, y_c = w // 2, h // 2
            # Find two edge points for the line
            points = []
            for x_edge in [0, w-1]:
                if abs(b) > 1e-6:
                    y = int(round(y_c - ((x_edge - x_c) * a) / b))
                    if 0 <= y < h:
                        points.append((x_edge, y))
            for y_edge in [0, h-1]:
                if abs(a) > 1e-6:
                    x = int(round(x_c - ((y_edge - y_c) * b) / a))
                    if 0 <= x < w:
                        points.append((x, y_edge))
            # Pick the two most distant points
            if len(points) >= 2:
                max_dist = -1
                pt1, pt2 = points[0], points[1]
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        dist = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
                        if dist > max_dist:
                            max_dist = dist
                            pt1, pt2 = points[i], points[j]
                # Sample pixels along the line from pt1 to pt2
                line_length = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
                if line_length > 0:
                    x_vals = np.linspace(pt1[0], pt2[0], line_length)
                    y_vals = np.linspace(pt1[1], pt2[1], line_length)
                    # Sample pixels along the line
                    sampled = np.zeros((line_length, 1, 3), dtype=frame.dtype)
                    for i in range(line_length):
                        x = round(x_vals[i])
                        y = round(y_vals[i])
                        if 0 <= x < w and 0 <= y < h:
                            sampled[i, 0, :] = frame[y, x, :]
                        else:
                            sampled[i, 0, :] = (0, 0, 0)
                    # Interpolate to output height
                    # Use numpy.interp to resample to output height
                    y_indices = np.linspace(0, line_length - 1, h)
                    sampled_resized = np.zeros((h, 1, 3), dtype=frame.dtype)
                    for ch in range(3):
                        sampled_resized[:, 0, ch] = np.interp(
                            y_indices,
                            np.arange(line_length),
                            sampled[:, 0, ch]
                        )
                    output_columns.append(sampled_resized)
        frame_number += 1
    print()

    cap.release()

    if frame_number == 0:
        print("No frames found in video.")
        return

    # Save output image
    output_img = np.hstack(output_columns)
    return output_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A commandline tool that processes an input file.")
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('--output', type=str, default='output.png', help='Path to the output image file')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input_file)
    if not cap.isOpened():
        sys.exit(f"Error: Cannot open video file {args.input_file}")

    output_img = unwrap(cap)
    cv2.imwrite(args.output, output_img)
    print(f"Saved output image with {output_img.shape[1]} columns to {args.output}")
