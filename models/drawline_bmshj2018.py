import matplotlib.pyplot as plt


bpp_result = [0.115239, 0.185698, 0.301804, 0.468972, 0.686378, 0.966864, 1.307441, 1.727503]
psnr_result = [27.106351, 28.679134, 30.616753, 32.554935, 34.580960, 36.720366, 38.807960, 40.794920]


bpp_myself = [0.1945,]
psnr_myself = [27.207735, ]


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