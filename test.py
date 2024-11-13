import cv2
import time
import os

# กำหนดขนาดของ checkerboard (จำนวนจุดในแนวนอน x แนวตั้ง)
CHECKERBOARD = (9, 6)  # ปรับค่าตามขนาดจริงของ checkerboard ที่ใช้

# สร้างโฟลเดอร์สำหรับเก็บภาพถ้าไม่มีอยู่
output_dir = 'snapshots/snapshot_img'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# เปิดการเชื่อมต่อกับกล้องเว็บแคม
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

img_counter = 0
start_time = time.time()

try:
    while img_counter < 30:
        ret, frame = cap.read()

        if not ret:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        # ปรับความสว่างและคอนทราสต์ของภาพ
        alpha = 1.5  # คอนทราสต์ (ค่า > 1 ทำให้คอนทราสต์สูงขึ้น)
        beta = 50    # ความสว่าง (ค่า > 0 ทำให้สว่างขึ้น)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # แปลงภาพเป็นขาวดำ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ใช้ Gaussian Blur ลด Noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # ใช้ Adaptive Threshold เพื่อลดผลกระทบจากแสงที่ไม่สม่ำเสมอ
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # ตรวจจับ checkerboard
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            # ใช้ cornerSubPix เพื่อทำให้การตรวจจับมุมละเอียดขึ้น
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # วาดเส้นที่ตรวจจับได้ลงในภาพ
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
            # แสดงจำนวนมุมที่ตรวจจับได้
            cv2.putText(frame, f"Detected: {len(corners)} corners", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # บันทึกภาพทุกๆ 1 วินาที
            if time.time() - start_time >= 1:
                img_name = f"{output_dir}/snapshot_{img_counter}.png"
                cv2.imwrite(img_name, frame)
                print(f"บันทึกภาพ {img_name} แล้ว")
                img_counter += 1
                start_time = time.time()

        # แสดงภาพจากกล้อง
        cv2.imshow("Webcam Feed with Checkerboard Detection", frame)

        # หยุดการทำงานด้วยการกด 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("หยุดการทำงาน")

finally:
    cap.release()
    cv2.destroyAllWindows()
