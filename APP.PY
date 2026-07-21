import cv2
import numpy as np
import streamlit as st

# ייבוא בסיסי של mediapipe
import mediapipe as mp
# פקודה שמכריחה את ה-solutions להיטען בצורה מפורשת למניעת שגיאות
import mediapipe.solutions.face_detection as mp_face_detection

# עיצוב כותרות האתר
st.set_page_config(page_title="אפליקציית שני ראשים", page_icon="👤")
st.title("👤 אפליקציית שני ראשים")
st.write("העלה תמונה והפוך לאדם עם שני ראשים בשניות!")

# כפתור העלאת קובץ באתר
uploaded_file = st.file_uploader("בחר תמונה מהמחשב או מהטלפון...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # קריאת התמונה שהועלתה לאתר
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    h, w, _ = image.shape
    output_image = image.copy()

    # הפעלת מודל זיהוי הפנים בצורה מאובטחת
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(image_rgb)

        if results.detections:
            detection = results.detections[0] # לוקחים את הפנים הראשונות
            bbox = detection.location_data.relative_bounding_box
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            padding_w = int(width * 0.4)
            padding_h = int(height * 0.6)
            
            x_start = max(0, x - padding_w)
            y_start = max(0, y - padding_h)
            x_end = min(w, x + width + padding_w)
            y_end = min(h, y + height + padding_h)
            
            head_crop = image[y_start:y_end, x_start:x_end]
            head_h, head_w, _ = head_crop.shape

            # מסכה אליפטית לטשטוש קצוות
            mask = np.zeros((head_h, head_w), dtype=np.uint8)
            cv2.ellipse(mask, (head_w // 2, head_h // 2), (head_w // 2, head_h // 2), 0, 0, 360, 255, -1)
            mask_blurred = cv2.GaussianBlur(mask, (51, 51), 0) / 255.0
            mask_3d = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)

            # מיקום הדבקה
            past_x_start = x_start - head_w
            past_x_end = x_start
            
            if past_x_start < 0:
                past_x_start = x_end
                past_x_end = x_end + head_w
            if past_x_end > w:
                past_x_start = x_start
                past_x_end = x_start + head_w
                y_start = min(h - head_h, y_end)

            past_x_end = min(w, past_x_end)
            past_x_start = max(0, past_x_end - head_w)
            past_y_end = min(h, y_start + head_h)
            y_start = max(0, past_y_end - head_h)

            background_roi = output_image[y_start:past_y_end, past_x_start:past_x_end]
            blended_head = (head_crop * mask_3d + background_roi * (1.0 - mask_3d)).astype(np.uint8)
            output_image[y_start:past_y_end, past_x_start:past_x_end] = blended_head

            # הצגת התוצאה באתר
            result_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            st.success("הקוד עבד! הנה התמונה החדשה שלך:")
            st.image(result_rgb, use_container_width=True)
            
            # כפתור הורדה
            _, img_encoded = cv2.imencode('.jpg', output_image)
            st.download_button(label="📥 הורד את התמונה", data=img_encoded.tobytes(), file_name="two_heads.jpg", mime="image/jpeg")
        else:
            st.warning("לא הצלחתי לזהות פנים בתמונה הזו. נסה תמונה ברורה יותר!")
