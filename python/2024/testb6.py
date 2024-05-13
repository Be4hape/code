import cv2
import time

def main():
    # 비디오 카메라 열기
    cap = cv2.VideoCapture("1958015.jpg")

    #time, fps 초기값 0
    prev_time = 0
    fps = 0

    while True:
        #
        time = time.time()

        #ret는 bool, ret가 t라면 frame에 cap.frame() 저장   
        ret, frame = cap.read()

        if ret:
            #초당 프레임 = 1 / 프레임 수(시간 - 이전 시간)
            fps = 1 / (time - prev_time)
            prev_time = time

            #우측상단 FPS표시, 10,30
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('frame', frame)

        #q입력시 탈출
        if cv2.waitKey(1) == ord('q'):
            break

    #카메라 종료
    cap.release()
    cv2.destroyAllWindows()

main()
