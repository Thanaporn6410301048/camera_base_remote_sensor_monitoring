import cv2
import numpy as np

def capture_images_for_calibration(chessboard_size=(9, 6), required_images=15, delay=1.0):
    """
    Captures images for camera calibration, ensuring geometric diversity and focus.
    
    Parameters:
    - chessboard_size: Tuple of the number of inner corners per chessboard row and column.
    - required_images: Total images to capture with diverse angles.
    - delay: Delay in seconds between captures to prevent blurring.
    """
    # ตั้งค่าเก็บข้อมูลและค่าพารามิเตอร์ต่าง ๆ
    cap = cv2.VideoCapture(0)
    captured_images = []
    pattern_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    
    print("Starting camera calibration capture. Adjust the checkerboard and hold it steady...")
    
    while len(captured_images) < required_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboard_size,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if found:
            # ตรวจสอบความคมชัดและแสดงมุมที่ตรวจพบ
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            frame = cv2.drawChessboardCorners(frame, chessboard_size, corners_subpix, found)
            
            # คำนวณความคมชัดโดยใช้ Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            focus_score = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

            # ตั้งค่าความคมชัดขั้นต่ำ
            focus_threshold = 100  # Adjust this value based on testing
            if focus_score > focus_threshold:
                print(f"Image captured (sharpness: {focus_score:.2f}).")
                captured_images.append((gray, corners_subpix))
                
                # เว้นช่วงเวลาเพื่อให้การตรวจจับไม่ทับซ้อนกัน
                cv2.waitKey(int(delay * 1000))

        cv2.imshow('Camera Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_images, pattern_points

def calibrate_camera(images, pattern_points):
    """
    Calibrates the camera based on captured images and pattern points.
    """
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    for (gray, corners) in images:
        obj_points.append(pattern_points)
        img_points.append(corners)

    # คำนวณค่าพารามิเตอร์การคาลิเบรต
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print("Calibration completed.")
    print("Camera Matrix:", mtx)
    print("Distortion Coefficients:", dist)
    
    return mtx, dist, rvecs, tvecs

# เรียกใช้ฟังก์ชันเพื่อเก็บภาพและคาลิเบรตกล้อง
images, pattern_points = capture_images_for_calibration()
camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(images, pattern_points)
