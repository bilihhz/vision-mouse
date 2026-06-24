import cv2
import mediapipe as mp
import ctypes
import math
import time
import os
import sys

# ==============================================================================
# ⚙️ 核心参数与全速开发配置区
# ==============================================================================
DEV_MODE = True
MODEL_PATH = "C:/Users/Administrator/Documents/vscode/2026/visiontouch/hand_landmarker.task"

# 🎯 物理像素判定门限
PINCH_PIXEL_THRESH = 15    
SCROLL_SENSITIVITY = 15.0  
MOUSE_SENSITIVITY = 2    
INPUT_THROTTLE_INTERVAL = 0.020  

# 🔒 卡尔曼滤波平滑系数（控制光标平时的丝滑度）
SMOOTH_ALPHA = 0.35  

# ⏱️ 点击防拖拽冻结时间窗口（单位：秒）
# 实验表明 0.35 秒足以应答“单击”，并防止捏合瞬间的位移引发误拖拽。如果觉得不够可以改成 1.0
CLICK_FREEZE_DURATION = 0.5  
# ==============================================================================

# 🚀 进程防打架互斥锁
ERROR_ALREADY_EXISTS = 183
mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\VisionTouch_Mouse_Lock")
if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
    print("⚠️ 自动化检测：发现后台有未关闭的僵尸进程，正在为您强行物理清理...")
    os.system("taskkill /F /IM python.exe")
    sys.exit(0)

# Windows API 结构体定义
LONG = ctypes.c_long; DWORD = ctypes.c_ulong; ULONG_PTR = ctypes.c_ulonglong 
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", LONG), ("dy", LONG), ("mouseData", DWORD), ("dwFlags", DWORD), ("time", DWORD), ("dwExtraInfo", ULONG_PTR)]
class INPUT_I(ctypes.Union): _fields_ = [("mi", MOUSEINPUT)]
class INPUT(ctypes.Structure):
    _fields_ = [("type", DWORD), ("i", INPUT_I)]

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008  
MOUSEEVENTF_RIGHTUP = 0x0010    
MOUSEEVENTF_WHEEL = 0x0800      
MOUSEEVENTF_ABSOLUTE = 0x8000

last_input_time = 0

def send_mouse(flags, x=0, y=0, data=0):
    global last_input_time
    current_time = time.time()
    if current_time - last_input_time < INPUT_THROTTLE_INTERVAL:
        return 
    last_input_time = current_time

    ii_ = INPUT_I()
    ii_.mi = MOUSEINPUT(int(x), int(y), int(data), flags, 0, 0)
    input_obj = INPUT(INPUT_MOUSE, ii_) 
    ctypes.windll.user32.SendInput(1, ctypes.pointer(input_obj), ctypes.sizeof(input_obj))

is_left_down = False
is_right_down = False
prev_i_x, prev_i_y = None, None
click_start_time = 0.0  
latest_result = None

# 🧠 全局平滑坐标与冻结锚点缓存
filtered_win_x, filtered_win_y = None, None
frozen_click_x, frozen_click_y = None, None  # 新增：记录按下瞬间的坐标锚点

def receive_async_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def force_shutdown(msg="程序正常退出"):
    print(f"\n🚨 {msg}。正在强行硬着陆并释放全部硬件占用...")
    try:
        send_mouse(MOUSEEVENTF_LEFTUP)
        send_mouse(MOUSEEVENTF_RIGHTUP)
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    os._exit(0) 

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.55, 
    min_tracking_confidence=0.55,
    result_callback=receive_async_result
)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 60) 

WINDOW_NAME = 'Tasks Surf Master'
if DEV_MODE:
    cv2.namedWindow(WINDOW_NAME)

fps = 0; frame_count = 0; start_time = time.time()
print(f"🚀 系统就绪。已激活首秒点击防拖拽锁死机制。")

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            if DEV_MODE:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    force_shutdown("用户点击了窗口右上角的 'X'")

            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0.5:
                fps = frame_count / elapsed_time
                frame_count = 0; start_time = time.time()

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            current_ui_state = "IDLE (NO HAND)"
            pixel_dist_ti, pixel_dist_tm = 999.0, 999.0 

            if latest_result and latest_result.hand_landmarks:
                landmarks = latest_result.hand_landmarks[0]
                
                t_px, t_py = int(landmarks[4].x * w), int(landmarks[4].y * h)   
                i_px, i_py = int(landmarks[8].x * w), int(landmarks[8].y * h)   
                m_px, m_py = int(landmarks[12].x * w), int(landmarks[12].y * h) 

                pixel_dist_ti = math.hypot(t_px - i_px, t_py - i_py) 
                pixel_dist_tm = math.hypot(t_px - m_px, t_py - m_py) 

                raw_win_x = 0.5 + (landmarks[8].x - 0.5) * MOUSE_SENSITIVITY
                raw_win_y = 0.5 + (landmarks[8].y - 0.5) * MOUSE_SENSITIVITY
                target_win_x = max(0, min(65535, int(raw_win_x * 65535)))
                target_win_y = max(0, min(65535, int(raw_win_y * 65535)))

                # 一阶低通滤波
                if filtered_win_x is None or filtered_win_y is None:
                    filtered_win_x = target_win_x
                    filtered_win_y = target_win_y
                else:
                    filtered_win_x = SMOOTH_ALPHA * target_win_x + (1 - SMOOTH_ALPHA) * filtered_win_x
                    filtered_win_y = SMOOTH_ALPHA * target_win_y + (1 - SMOOTH_ALPHA) * filtered_win_y

                win_x, win_y = int(filtered_win_x), int(filtered_win_y)

                # ────────────────────────────────────────────────────────
                # 🎯 核心逻辑分流控制区
                # ────────────────────────────────────────────────────────
                
                # 【条件 1】：滚轮模式
                if pixel_dist_ti < PINCH_PIXEL_THRESH and pixel_dist_tm < PINCH_PIXEL_THRESH:
                    current_ui_state = "SCROLL MODE"
                    if is_left_down: send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTUP, win_x, win_y); is_left_down = False
                    if is_right_down: send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_RIGHTUP, win_x, win_y); is_right_down = False

                    if prev_i_y is not None:
                        dy = i_py - prev_i_y
                        if abs(dy) > 1: 
                            send_mouse(MOUSEEVENTF_WHEEL, data=-dy * SCROLL_SENSITIVITY)

                # 【条件 2】：大拇指与食指捏合 -> 左键单击 / 拖拽
                elif pixel_dist_ti < PINCH_PIXEL_THRESH:
                    if is_right_down: send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_RIGHTUP, win_x, win_y); is_right_down = False
                    
                    if not is_left_down:
                        # 💥 刚按下的瞬间：记录当前坐标为固定锚点，并下发按下事件
                        frozen_click_x, frozen_click_y = win_x, win_y
                        send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN, frozen_click_x, frozen_click_y)
                        is_left_down = True
                        click_start_time = time.time()
                        current_ui_state = "LEFT DOWN (FROZEN)"
                    else:
                        # ⏳ 已经按下的持续状态：检查是否在时间保护窗口内
                        hold_duration = time.time() - click_start_time
                        if hold_duration < CLICK_FREEZE_DURATION:
                            # 强行喂给 Windows 刚按下那一瞬间的坐标，手怎么动都不理
                            send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, frozen_click_x, frozen_click_y)
                            current_ui_state = f"LEFT HOLD (FROZEN {CLICK_FREEZE_DURATION - hold_duration:.2f}s)"
                        else:
                            # 超过保护时间，解封！允许自由拖拽
                            send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, win_x, win_y)
                            current_ui_state = "DRAGGING..."
                        
                        # 长按 1.2 秒防粘连强制解脱锁
                        if hold_duration > 1.2:
                            send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP, win_x, win_y)
                            is_left_down = False

                # 【条件 3】：右键模式
                elif pixel_dist_tm < PINCH_PIXEL_THRESH:
                    current_ui_state = "RIGHT CLICK"
                    if is_left_down: send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTUP, win_x, win_y); is_left_down = False
                    
                    if not is_right_down:
                        send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | MOUSEEVENTF_RIGHTDOWN, win_x, win_y)
                        is_right_down = True
                    else:
                        send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, win_x, win_y)

                # 【条件 4】：纯光标移动
                else:
                    current_ui_state = "CURSOR MOVE"
                    
                    if is_left_down or (pixel_dist_ti > PINCH_PIXEL_THRESH + 8): 
                        send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP, win_x, win_y)
                        is_left_down = False
                        
                    if is_right_down or (pixel_dist_tm > PINCH_PIXEL_THRESH + 8):
                        send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | MOUSEEVENTF_RIGHTUP, win_x, win_y)
                        is_right_down = False
                    
                    send_mouse(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, win_x, win_y)

                prev_i_x, prev_i_y = i_px, i_py

                if DEV_MODE:
                    for point in landmarks:
                        cx, cy = int(point.x * w), int(point.y * h)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
                    cv2.circle(frame, (i_px, i_py), 6, (0, 255, 0), -1)   
                    cv2.circle(frame, (t_px, t_py), 6, (255, 0, 0), -1)   
                    cv2.circle(frame, (m_px, m_py), 6, (0, 0, 255), -1)   
            else:
                if is_left_down: send_mouse(MOUSEEVENTF_LEFTUP); is_left_down = False
                if is_right_down: send_mouse(MOUSEEVENTF_RIGHTUP); is_right_down = False
                prev_i_x, prev_i_y = None, None
                filtered_win_x, filtered_win_y = None, None
                frozen_click_x, frozen_click_y = None, None  # 手放开时连同锚点一起清空

            if DEV_MODE:
                cv2.rectangle(frame, (10, 10), (520, 140), (20, 20, 20), -1)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"ACTION: {current_ui_state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"T-I Dist: {pixel_dist_ti:.1f} / Thresh: {PINCH_PIXEL_THRESH}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"T-M Dist: {pixel_dist_tm:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.imshow(WINDOW_NAME, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('x') or key == ord('q'): 
                    force_shutdown("键盘按下安全退出键")
            else:
                time.sleep(0.001)

except KeyboardInterrupt:
    force_shutdown("终端强拆信号 Ctr+C")
except Exception as e:
    force_shutdown(f"运行异常崩溃: {e}")

force_shutdown("运行结束")