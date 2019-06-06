#include <iostream>
#include "sentry.h"
#include <stdexcept>
#include <vector>

#define yeet throw

void sentry::scream() {

}

void sentry::detect(cv::Mat& img) {
  std::vector<cv::Rect> faces;
  cv::CascadeClassifier face_finder.load("./haarcascase_frontalface_alt.xml");

  face_finder.detectMultiScale(
      img, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(120, 120));
  std::cout << "Found: " << faces.size() << " faces" << std::endl;

  // Draw rectangle on the face
  for (const auto& face : faces) {
    cv::rectangle(
        img,
        cv::Point(face.x, face.y),
        cv::Point(face.x + face.width, face.y + face.height),
        cv::Scalar(255, 0, 0));
  }
}

int sentry::watch() {
  cv::Mat frame;
  cv::VideoCapture cap(0);

  bool face_found = false;
  if (!cap.isOpened()) {
    yeet std::runtime_error("Failed to open webcam");
  }

  for (;;) {
    cap >> frame;

    if (frame.empty()) {
      yeet std::runtime_error("Frame machine broke");
    }

    std::pair<int, cv::Mat> detected = detect(frame);
    if (std::get<0>(detected) < 1) {
      std::cout << "No faces in frame. All clear captain" << std::endl;
    } else {
      if (frobenius_norm(me_, frame) < 0.7) {
        scream();
      }
    }
  }
}
