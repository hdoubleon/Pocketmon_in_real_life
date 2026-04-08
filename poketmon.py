import cv2 as cv
import numpy as np


# --- 배경이 투명한 PNG 이미지를 합성하는 헬퍼 함수 ---
def draw_snorlax(bg_img, overlay_img, center_x, center_y, size):
    # 크기 조절 (잠만보의 크기)
    h, w = overlay_img.shape[:2]
    scale = size / h
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w == 0 or new_h == 0:
        return bg_img

    resized_overlay = cv.resize(overlay_img, (new_w, new_h))

    # 잠만보의 발끝(하단 중앙)이 center_x, center_y에 오도록 좌표 계산
    x_offset = int(center_x - new_w / 2)
    y_offset = int(center_y - new_h)

    # 화면 밖으로 나가는 경우를 방지하는 예외 처리 (화면 경계 계산)
    y1, y2 = max(0, y_offset), min(bg_img.shape[0], y_offset + new_h)
    x1, x2 = max(0, x_offset), min(bg_img.shape[1], x_offset + new_w)

    y1o, y2o = max(0, -y_offset), new_h - max(0, (y_offset + new_h) - bg_img.shape[0])
    x1o, x2o = max(0, -x_offset), new_w - max(0, (x_offset + new_w) - bg_img.shape[1])

    if y1 >= y2 or x1 >= x2:
        return bg_img

    # 투명도(Alpha) 값을 이용해 합성
    alpha_s = resized_overlay[y1o:y2o, x1o:x2o, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        bg_img[y1:y2, x1:x2, c] = (
            alpha_s * resized_overlay[y1o:y2o, x1o:x2o, c]
            + alpha_l * bg_img[y1:y2, x1:x2, c]
        )
    return bg_img


# --------------------------------------------------------

# 1. 카메라 캘리브레이션 파라미터
K = np.array(
    [
        [887.94022777, 0.0, 955.04429227],
        [0.0, 883.53091051, 526.74577586],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
dist_coeff = np.array([-0.01492128, 0.1062357, 0.0, 0.0, 0.0], dtype=np.float32)

# 2. 체스보드 및 파일 설정
video_file = "chessboard.MOV"  # 촬영한 동영상 경로
snorlax_file = "snorlax.png"  # ★ 다운받은 투명 잠만보 이미지 경로
board_pattern = (13, 9)
board_cellsize = 0.025

# 잠만보 이미지 불러오기
snorlax_img = cv.imread(snorlax_file, cv.IMREAD_UNCHANGED)

# 🚨 디버깅: 이미지가 제대로 안 불러와졌으면 여기서 멈춤!
if snorlax_img is None:
    print("❌ 에러: snorlax.png 파일을 찾을 수 없거나 경로가 잘못되었습니다!")
    exit()
else:
    print(f"✅ 잠만보 로딩 성공! 사이즈: {snorlax_img.shape}")

# 3. 3D 월드 좌표 설정
obj_points = board_cellsize * np.array(
    [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])],
    dtype=np.float32,
)

# ★ 잠만보가 서 있을 3D 체스보드 상의 좌표 (체스판 중앙쯤)
snorlax_3d_base = np.array([[4.5, 3.0, 0]], dtype=np.float32) * board_cellsize
# 잠만보의 크기(높이)를 가늠하기 위한 머리 위 3D 좌표
snorlax_3d_head = np.array([[4.5, 3.0, -3.0]], dtype=np.float32) * board_cellsize

# 4. 루프 시작
video = cv.VideoCapture(video_file)

while True:
    valid, img = video.read()
    if not valid:
        break

    # 🚨 디버깅: 현재 동영상 해상도 출력 (한 번만)
    if video.get(cv.CAP_PROP_POS_FRAMES) == 1:
        print(
            f"🎥 동영상 해상도: {img.shape[1]}x{img.shape[0]} (1920x1080과 같은지 확인하세요!)"
        )

    # ⚡ 속도 개선: FAST_CHECK 플래그 추가 (체스판 없으면 바로 패스)
    complete, img_points = cv.findChessboardCorners(
        img, board_pattern, flags=cv.CALIB_CB_FAST_CHECK
    )

    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        pt_base, _ = cv.projectPoints(snorlax_3d_base, rvec, tvec, K, dist_coeff)
        pt_head, _ = cv.projectPoints(snorlax_3d_head, rvec, tvec, K, dist_coeff)

        base_x, base_y = int(pt_base[0][0][0]), int(pt_base[0][0][1])
        head_x, head_y = int(pt_head[0][0][0]), int(pt_head[0][0][1])

        # 🚨 디버깅: 잠만보가 찍히는 X, Y 픽셀 좌표가 화면 밖인지 확인
        # print(f"📍 잠만보 발끝 좌표: X={base_x}, Y={base_y}")

        pixel_height = abs(base_y - head_y)

        # 안전 장치: 높이가 0이거나 비정상적으로 작으면 건너뜀
        if pixel_height > 10:
            img = draw_snorlax(img, snorlax_img, base_x, base_y, pixel_height)

        cv.putText(
            img,
            "A wild Snorlax appeared!",
            (20, 50),
            cv.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
    cv.imshow("Pokemon Chess AR", img)

    if cv.waitKey(1) == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
