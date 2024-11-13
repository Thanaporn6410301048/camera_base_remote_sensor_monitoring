import cv2
import numpy as np
import time

def show_video_with_board_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    print("เริ่มแสดงวิดีโอจากกล้อง... กด 'q' เพื่อหยุด")

    # กำหนดขนาดของตารางหมากฮอสในเซนติเมตร
    actual_table_width_cm = 30  # ความกว้างของตารางจริง
    actual_table_height_cm = 30  # ความสูงของตารางจริง

    # กำหนดอัตราส่วนการวัด (1 unit = 10 cm)
    scale_factor = 10.0  # เปลี่ยนได้ตามความเหมาะสม

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

                # คำนวณพื้นที่ในเซนติเมตร
                width_cm = (w / scale_factor)  # ความกว้างในเซนติเมตร
                height_cm = (h / scale_factor)  # ความสูงในเซนติเมตร
                area_cm2 = width_cm * height_cm  # พื้นที่ในตารางเซนติเมตร

                # แสดงผลค่าเมทริกในกรอบ
                cv2.putText(frame, f'Area: {area_cm2:.2f} cm²', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # แสดงเฟรมจากกล้อง
        cv2.imshow('Video Feed', frame)

        # ออกจากลูปเมื่อกด 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_video_with_board_detection()