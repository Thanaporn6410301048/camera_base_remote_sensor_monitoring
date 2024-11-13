import cv2
import numpy as np
import glob

# กำหนดขนาดของ checkerboard (จำนวนจุดในแนวนอน x แนวตั้ง)
CHECKERBOARD = (9, 6)

# กำหนดพารามิเตอร์สำหรับการหาค่าความแม่นยำ
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# สร้างอาเรย์สำหรับเก็บจุด 3D และ 2D
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # จุด 3D ในโลก
imgpoints = []  # จุด 2D ในภาพ

# โหลดภาพจากโฟลเดอร์
images = glob.glob('snapshots\snapshot_img\*.png')

if not images:
    print("ไม่พบภาพในโฟลเดอร์ snapshot_img")
else:
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # หา checkerboard
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"พบ checkerboard ในภาพ: {fname}")
            # วาดมุมที่ตรวจจับได้ลงในภาพ
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Checkerboard', img)
            cv2.waitKey(500)  # แสดงภาพเป็นเวลา 500ms
        else:
            print(f"ไม่พบ checkerboard ในภาพ: {fname}")

# ปิดหน้าต่างภาพ
cv2.destroyAllWindows()

# ตรวจสอบว่ามีภาพที่ใช้ในการคำนวณหรือไม่
if len(objpoints) > 0 and len(imgpoints) > 0:
    # ใช้ภาพสุดท้ายในการคำนวณขนาด
    gray_shape = gray.shape[::-1]  # ขนาดของภาพสุดท้ายที่ใช้
    # คำนวณเมตริกซ์ของกล้องและพารามิเตอร์การบิดเบี้ยว
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # แสดงค่าเมตริกซ์ของกล้อง
    print("Camera Matrix:")
    print(mtx)

    # แสดงพารามิเตอร์การบิดเบี้ยว
    print("Distortion Coefficients:")
    print(dist)
else:
    print("ไม่สามารถคำนวณเมตริกซ์ของกล้องได้ เนื่องจากไม่มีภาพที่ใช้ในการคำนวณ")
