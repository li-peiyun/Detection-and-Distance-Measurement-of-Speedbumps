from ultralytics import YOLO
import cv2


##################################################
# 乐怡：把视频逐帧分割为一组图片，对这组图片进行去畸变处理
# 将去畸变后的图片放到undistorted_image文件夹中
# 完成上述步骤后，将第19行的测试数据注释掉，换成22行的实际数据
##################################################

# 加载我训练的YOLOv8n模型
model = YOLO("speedbump.pt")

# 只显示减速带：减速带对应class=1
# 不显示标签，不显示置信度
# source可以是图片、视频、文件夹

# 测试数据
results = model.predict(source="test_images", save=False, classes=1, show_labels=False, show_conf=False)

# 实际数据
# results = model.predict(source="undistorted_image", save=False, classes=1, show_labels=False, show_conf=False)

# result是一张图片的检测结果，results是所有图片的检测结果，index是当前迭代的次数
for index, result in enumerate(results):
    # 获取该检测结果图片
    path = result.path
    # print(path)
    image = cv2.imread(path)

    # 遍历该图中所有检测框（即减速带）
    for xyxy in result.boxes.xyxy:
        # corners是一个减速带的顶点坐标，格式为[xmin, ymin, xmax, ymax]
        corners = xyxy.tolist()

        # 从corners中获取顶点坐标
        xmin, ymin, xmax, ymax = map(int, corners)

        # 在图像上画矩形
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # (0, 255, 0) 是绿色，2 是矩形的边框宽度

        ################################################################
        # 李欣：计算减速带的距离
        # 上述的xmin, ymin, xmax, ymax是减速带的坐标，使用opencv坐标系
        # (xmin, ymin)是左上点坐标，(xmax, ymax)是右下点坐标
        # 计算减速带距离，命名为distance，替换下面的测试数据 distance = 5.3
        ################################################################
        distance = 5.3  # 测试数据

        # 在矩形上方添加文字
        text = "distance=" + str(distance)
        cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示带有矩形框的图像
    # cv2.imshow("Image with Bounding Boxes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存标记后的照片在marked_images文件夹
    save_path = "marked_images\\" + str(index) + ".jpg"
    cv2.imwrite(save_path, image)

# 减速带检测及测距后的图片保存在marked_images文件夹内

#############################################
# 乐怡：将marked_images文件夹内的图片重新组合成视频
#############################################
