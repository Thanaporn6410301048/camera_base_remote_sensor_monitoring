# ปรับความสว่างและคอนทราสต์ของภาพ
        alpha = 1.5  # คอนทราสต์ (ค่า > 1 ทำให้คอนทราสต์สูงขึ้น)
        beta = 50    # ความสว่าง (ค่า > 0 ทำให้สว่างขึ้น)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
