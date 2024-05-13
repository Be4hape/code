import numpy as np
import matplotlib.pyplot as plt

def process_image(image_file):
    # 이미지 불러오기
    im = plt.imread(image_file)
    
    # 이미지의 고유한 값과 빈도 계산
    unique_values, counts = np.unique(im, return_counts=True)
    
    return im, unique_values, counts

# 이미지 파일 이름 리스트
image_files = ["AA1_1.jpg", "AA1_2.jpg", "AA1_3.jpg"]

# subplot 설정
num_im = len(image_files)
fig, axes = plt.subplots(num_im, 2)

# 이미지와 히스토그램 출력
for i in range(num_im):
    # 이미지 파일 이름
    im_file = image_files[i]
    
    # 이미지 처리
    im, unique_values, counts = process_image(im_file)
    
    # subplot에 이미지 출력 (그레이 스케일로)
    axes[i, 0].imshow(im, cmap='gray')  # 그레이 스케일로 출력

    # subplot에 히스토그램 출력
    axes[i, 1].bar(unique_values, counts)

plt.tight_layout()
plt.show()
