import cv2
import mediapipe as mp

# 初始化 MediaPipe 的繪圖工具與 Holistic 模型
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 步驟 1：連接本機鏡頭
# 參數 0 通常代表系統預設的第一顆網路攝影機 (Webcam)
cap = cv2.VideoCapture(0)

# 步驟 2：設定 Holistic 模型參數
# min_detection_confidence: 初始偵測的信心閾值
# min_tracking_confidence: 後續追蹤的信心閾值
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    # 確認攝影機是否成功開啟
    while cap.isOpened():
        # 讀取攝影機的每一幀畫面
        success, image = cap.read()
        if not success:
            print("無法讀取攝影機畫面。")
            continue

        # 步驟 3：影像預處理
        # OpenCV 預設讀取的顏色格式為 BGR，但 MediaPipe 需要的是 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 為了提升運算效能，將影像標記為不可寫入
        image.flags.writeable = False
        
        # 步驟 4：將影像送入模型進行即時判定
        results = holistic.process(image)

        # 步驟 5：視覺化處理
        # 將影像改回可寫入，並將顏色格式轉換回 BGR 以便 OpenCV 顯示
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. 繪製臉部網格
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.face_landmarks, 
                mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )
        
        # 2. 繪製右手特徵點
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS
            )

        # 3. 繪製左手特徵點
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS
            )

        # 4. 繪製全身骨架特徵點
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_holistic.POSE_CONNECTIONS
            )

        # 步驟 6：顯示即時影像畫面
        # 將結果顯示在名為 'Real-time Skeleton Detection' 的視窗中
        cv2.imshow('Real-time Skeleton Detection', image)

        # 設定中斷條件：按下鍵盤的 'q' 鍵即可離開迴圈並關閉程式
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 步驟 7：釋放資源
cap.release()
cv2.destroyAllWindows()