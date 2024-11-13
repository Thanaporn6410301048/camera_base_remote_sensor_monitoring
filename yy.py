import cv2
import numpy as np
import time

def show_video_with_board_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    print("เริ่มแสดงวิดีโอจากกล้อง... กด 'q' เพื่อหยุด")

    # กำหนดอัตราส่วนการวัด (1 unit = 10 cm)
    scale_factor = 10.0
    last_snapshot_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ไม่สามารถอ่านเฟรมจากกล้องได้")
            break

        # แปลงเฟรมเป็นสีเทา
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ใช้ GaussianBlur เพื่อลดนอยส์
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # ใช้ Canny edge detection
        edges = cv2.Canny(blurred_frame, 50, 150)

        # ค้นหาคอนทัวร์
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # คำนวณพื้นที่ของคอนทัวร์
            area = cv2.contourArea(contour)
            if area > 1000:  # กำหนดขนาดขั้นต่ำของคอนทัวร์
                # หาขอบเขตของคอนทัวร์
                x, y, w, h = cv2.boundingRect(contour)

                # วาดกรอบรอบวัตถุ
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # คำนวณค่าเมทริกและเซนติเมตร
                metric_value = (w / scale_factor) * 2  # คูณ 2 เพื่อให้ค่าเป็น 2 เท่าของความเป็นจริง
                cm_value = metric_value  # ค่าที่ได้เป็นเซนติเมตร

                # แสดงผลค่าเมทริกในกรอบ
                cv2.putText(frame, f'Metric: {metric_value:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # ตรวจสอบเวลาที่ผ่านมาเพื่อจับภาพทุกๆ 4 วินาที
                current_time = time.time()
                if current_time - last_snapshot_time >= 4:
                    last_snapshot_time = current_time
                    snapshot = frame.copy()  # ทำการจับภาพ
                    cv2.imwrite(f'snapshot_{int(current_time)}.png', snapshot)  # บันทึกภาพ

        # แสดงเฟรมจากกล้อง
        cv2.imshow('Video Feed', frame)

        # ออกจากลูปเมื่อกด 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_video_with_board_detection()