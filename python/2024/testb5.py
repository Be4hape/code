import cv2

def main():
    cap = cv2.VideoCapture(0)

    print("카메라 속성:")
    print("------------")
    print(f"프레임 폭 (Width): {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"프레임 높이 (Height): {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"프레임 속도 (FPS): {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"코덱 (Codec): {cap.get(cv2.CAP_PROP_FOURCC)}")
    print(f"프레임 수 (Frame count): {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"밝기 (Brightness): {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"콘트라스트 (Contrast): {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"채도 (Saturation): {cap.get(cv2.CAP_PROP_SATURATION)}")
    print(f"색상 온도 (Hue): {cap.get(cv2.CAP_PROP_HUE)}")
    print(f"게인 (Gain): {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"포커스 (Focus): {cap.get(cv2.CAP_PROP_FOCUS)}")
    print(f"오토포커스 (Autofocus): {cap.get(cv2.CAP_PROP_AUTOFOCUS)}")

    cap.release()

main()
