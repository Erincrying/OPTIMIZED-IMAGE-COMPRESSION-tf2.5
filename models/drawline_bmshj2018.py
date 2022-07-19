import matplotlib.pyplot as plt


bpp_result = [0.115239, 0.185698, 0.301804, 0.468972, 0.686378, 0.966864, 1.307441, 1.727503]
psnr_result = [27.106351, 28.679134, 30.616753, 32.554935, 34.580960, 36.720366, 38.807960, 40.794920]

bpp_myself = [0.1935, 0.2553, 0.4074, 0.5314]
psnr_myself = [27.165842, 28.450409, 31.436895, 33.45183]


# 第一个码率点
bpp_myself_test = [0.1945]
psnr_myself_test = [27.207735]

bpp_myself_01 = [0.1935]
psnr_myself_01 = [27.165842]

bpp_myself_02 = [0.2301]
psnr_myself_02 = [27.200388]

bpp_myself_03 = [0.2293]
psnr_myself_03 = [25.743942]


bpp_myself_04 = [0.2328]
psnr_myself_04 = [26.97355]


# 第二个码率点
bpp_myself_SEC_01= [0.2553]
psnr_myself_SEC_01= [28.450409]

bpp_myself_SEC_02= [0.2946]
psnr_myself_SEC_02= [28.352106]

bpp_myself_SEC_03= [0.2975]
psnr_myself_SEC_03= [28.341482]


# 第三个码率点
bpp_myself_THI_01= [0.3598]
psnr_myself_THI_01= [30.567152]

bpp_myself_THI_02= [0.4074]
psnr_myself_THI_02= [31.436895]


bpp_myself_THI_03= [0.4128]
psnr_myself_THI_03= [31.440132]


# 第四个码率点
bpp_myself_FOUR_01= [0.5314]
psnr_myself_FOUR_01= [33.45183]


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


# 单独加的对比点
# 第一个码率点
# 自己的psnrbmshj2018_test
# plt.scatter(bpp_myself_test, psnr_myself_test)
# plt.plot(bpp_myself_test, psnr_myself_test, label='bmshj2018_test')

# # 自己的psnrbmshj2018_01
# plt.scatter(bpp_myself_01, psnr_myself_01)
# plt.plot(bpp_myself_01, psnr_myself_01, label='bmshj2018_01')
# # 自己的psnrbmshj2018_02
# plt.scatter(bpp_myself_02, psnr_myself_02)
# plt.plot(bpp_myself_02, psnr_myself_02, label='bmshj2018_02')
# # 自己的psnrbmshj2018_03
# plt.scatter(bpp_myself_03, psnr_myself_03)
# plt.plot(bpp_myself_03, psnr_myself_03, label='bmshj2018_03')
# # 自己的psnrbmshj2018_04
# plt.scatter(bpp_myself_04, psnr_myself_04)
# plt.plot(bpp_myself_04, psnr_myself_04, label='bmshj2018_04')


# 第二个码率点
# 自己的psnrbmshj2018_SEC_01
# plt.scatter(bpp_myself_SEC_01, psnr_myself_SEC_01)
# plt.plot(bpp_myself_SEC_01, psnr_myself_SEC_01, label='bmshj2018_SEC_01')

# plt.scatter(bpp_myself_SEC_02, psnr_myself_SEC_02)
# plt.plot(bpp_myself_SEC_02, psnr_myself_SEC_02, label='bmshj2018_SEC_02')

# plt.scatter(bpp_myself_SEC_03, psnr_myself_SEC_03)
# plt.plot(bpp_myself_SEC_03, psnr_myself_SEC_03, label='bmshj2018_SEC_03')



# 第三个码率点
# plt.scatter(bpp_myself_THI_01, psnr_myself_THI_01)
# plt.plot(bpp_myself_THI_01, psnr_myself_THI_01, label='bmshj2018_THI_01')

# plt.scatter(bpp_myself_THI_02, psnr_myself_THI_02)
# plt.plot(bpp_myself_THI_02, psnr_myself_THI_02, label='bmshj2018_THI_02')

# plt.scatter(bpp_myself_THI_03, psnr_myself_THI_03)
# plt.plot(bpp_myself_THI_03, psnr_myself_THI_03, label='bmshj2018_THI_03')


# 第四个码率点
plt.scatter(bpp_myself_FOUR_01, psnr_myself_FOUR_01)
plt.plot(bpp_myself_FOUR_01, psnr_myself_FOUR_01, label='bmshj2018_FOUR_01')



#添加网格信息
plt.grid(True, linestyle='--', alpha=0.5) #默认是True，风格设置为虚线，alpha为透明度
plt.legend() # 为了能显示label
plt.show()
