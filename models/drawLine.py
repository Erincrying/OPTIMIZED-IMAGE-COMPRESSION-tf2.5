import matplotlib.pyplot as plt
# import numpy as np

# bpp_result_list = [0.119752, 0.194591, 0.316000, 0.481060, 0.721303, 1.060841, 1.458681, 1.957564]
# psnr_result_list = [0.903700, 0.931041, 0.954783, 0.969139, 0.980815, 0.986755, 0.992090, 0.994965]

# bpp_result = np.array(bpp_result_list)
# psnr_result = np.array(psnr_result_list)

bpp_result = [0.119752, 0.194591, 0.316000, 0.481060, 0.721303, 1.060841, 1.458681, 1.957564]
psnr_result = [26.775134, 28.348719, 30.020793, 31.729556,33.685797, 35.815864, 38.019954, 40.133996]

# bpp_myself = [0.4639, 0.6282, 0.8110, 0.9331, 1.0590, 2.0109]
# psnr_myself = [31.049662, 32.634518, 33.925323, 34.46075, 34.891132, 35.779408]

bpp_myself = [0.1408, 0.2338, 0.4019, 0.6025, 0.6282, 0.8110]
psnr_myself = [26.523191, 27.964922, 30.30449, 32.265305, 32.634518, 33.925323]

# 添加横纵坐标与标题
plt.xlabel('bit rate [bit/px]')
plt.ylabel('PSNR[db]')
plt.title('rate–distortion')

# 原文psnr
plt.scatter(bpp_result, psnr_result)
plt.plot(bpp_result, psnr_result, label='original')

# 自己的psnr
plt.scatter(bpp_myself, psnr_myself)
plt.plot(bpp_myself, psnr_myself, label='myself')
# plt.plot(bpp_myself, psnr_myself, color='g', linestyle='-.', label = 'myself')
#添加网格信息
plt.grid(True, linestyle='--', alpha=0.5) #默认是True，风格设置为虚线，alpha为透明度
plt.legend() # 为了能显示label
plt.show()
