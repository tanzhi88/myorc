import numpy as np

from cv2 import cv2
from matplotlib import pyplot as plt


class MyOrc:

    def __init__(self, image_url, out_dir):
        self.image_url = image_url
        self.out_dir = out_dir
        # 打开图片
        self.original = cv2.imread(self.image_url)
        self.image = self.original
        # 图片信息
        (self.h, self.w, model) = self.image.shape
        # 字符位置信息
        self.position = []

    def resize(self, rate=.5):
        """缩放"""
        w = round(self.w * rate)
        h = round(self.h * rate)
        self.image = cv2.resize(self.image, (w, h))

    def gray(self):
        """灰度"""
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def binary(self):
        """图片二值化"""
        ret, thresh = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        self.image = thresh

    def eroded(self):
        """腐蚀"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        return cv2.erode(self.image, kernel)

    def find_contours(self):
        """获取轮廓"""
        contours, hierarchy = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    def noise(self, h=20):
        """降噪"""
        self.image = cv2.fastNlMeansDenoising(self.image, h=h)

    def pre_processing(self):
        """
        图片预处理
        # 灰度
        # 转二进制
        # 降噪
        :return:
        """
        self.gray()
        self.binary()
        self.noise()

    def get_position(self):
        """
        此函数将获取图片中每个字符的位置信息
        # 1、给图像绘制投影
        # 2、根据投影数据找出每个字符区域的界线
        # 3、按界线划分出每个字符区域并记录各项位置信息
        :return:
        """
        p_date, p_image = projection(self.image)
        self.position = division(p_date, (0, self.w), (0, self.h))
        self.get_position_two()

    def get_position_two(self):
        # 二次切分
        for i, line in enumerate(self.position):
            coordinate = line["position"]["coordinate"]
            image = self.get_image_by_position(coordinate)
            # self.save('line_' + str(i) + '.jpg', image)
            p_l_data, p_l_image = projection(image, 'vertically')
            self.position[i]["characters"] = division(data=p_l_data, x=coordinate[0], y=coordinate[1],
                                                      orientation='vertically')

    def get_image_by_position(self, coordinate):
        """根据位置信息生成图像"""
        x = coordinate[0]
        y = coordinate[1]
        return self.image[y[0]:y[1], x[0]:x[1]]

    def show(self, image=None):
        """显示图片"""
        if image is not None:
            cv2.imshow('image', image)
        else:
            cv2.imshow('image', self.image)
        cv2.waitKey(0)

    def plt_show(self, image):
        """使用plt显示图片"""
        plt.imshow(image)
        plt.show()

    def save(self, name, image):
        """保存图片"""
        cv2.imwrite(self.out_dir + name, image)


def projection(image, orientation='horizontal'):
    """
    生成投影数据
    # 此函数只记录黑点数据，因此图像必须是白底的二进制图像
    # 函数将图片按列切分，按图片度度来循环，计算每行的黑点总数
    # 因此，当纵向切分时，会先将图片进行转置, 并将计算出来的投影图像转置回来
    #
    black: 记录每行(列)的黑点数量
    projection: 为了可以绘制一个投影图像，生成一个维度和原图一至的矩陈(相当于全黑的图片)
    #
    :param image: 要生成投影的图像
    :param orientation: 按哪个方向进行投影,分别是纵向(vertically) 与 横向(horizontal)，默认按横向
    :return: 1、投影数据(用于切分图片) 2、投影图像数据(用于显示投影图像)
    """
    black = []
    if orientation == 'vertically':
        image = image.T
    width = image.shape[1]
    height = image.shape[0]
    projection_ = np.zeros((height, width), dtype=int)

    for i in range(height):
        # 每行的黑点总数
        black_sum = len(image[i, :][image[i, :] == 0])
        black.append(black_sum)
        # 把行的黑点排到最低部，其余部份变成白色
        projection_[i, :width - black_sum - 1] = 255
    # 如果纵向切分则将投影图像进行转置
    if orientation == 'vertically':
        projection_ = projection_.T
    return black, projection_


def _get_position(x, y):
    """
    设置位置信息
        位置信息包括:
            位置 {left, top, width, height}
            顶点 [左上(x,y), 右上(x,y), 右下(x,y), 左下(x,y)]
            坐标 [x轴(x0, x1),y轴(y0, y1)]
    :param x: x轴位置元组,从左至右计算, (起点位置, 始点位置)
    :param y: y轴位置元组,从上至下计算, (起点位置, 始点位置)
    :return: 一块区域的位置信息
    """
    block = {
        "position": {
            "bounding_box": {
                "left": x[0],
                "top": y[0],
                "width": x[1] - x[0],
                "height": y[1] - y[0]
            },
            "vertices": [
                (x[0], y[0]),
                (x[1], y[0]),
                (x[1], y[1]),
                (x[0], y[1])
            ],
            "coordinate": [
                x,
                y
            ]
        }
    }
    return block


def division(data, x, y, orientation='horizontal'):
    """
    分割图片
    # 根据传过来的投影信息，找到每个文字区域的界线
    # 循环投影数据，找出每字符块的界线
    # 1、进入区域: 如果投影item有数据，并且之前没有标记进入字符区域，说明进入字符区
    #     记录此位置信息并标记已进入
    # 2、离开区域: 如果标记着已进入字符区，且item已无数据，说明开始出字符区
    #     此时在位置信息里追加一个包含位置信息的字典
    #     此处注意结束位置必须是上次循环的位置,因为到本次循环时已经出了字符区1个位置
    # 默认按横向切分，当按横向切分时，区域起点、始点位置将做为被切分图片的y轴上的两个值
    # 当按纵向切分时，区域起点、始点将做为x轴上的两个值，
    #
    in_block: 标记是否进入字符区域
    s: 标记每个字符块的起始位置
    #
    :param data: 投影数据
    :param x: 要分割图片的x轴位置元组,从左至右计算, (起点位置, 始点位置)
    :param y: 要分割图片的y轴位置元组,从上至下计算, (起点位置, 始点位置)
    :param orientation: 按什么方向进行切分
    :return: 返回一系列被切分图像的位置信息
    """
    position = []
    in_block = False
    s = 0
    for i, item in enumerate(data):
        # 进入区域
        if item and not in_block:
            s = i
            in_block = True
        # 离开区域
        elif in_block and not item:
            in_block = False
            if i - 1 - s < 1:
                continue
            if orientation == 'vertically':
                block = _get_position(x=(s, i - 1), y=y)
            else:
                block = _get_position(x=x, y=(s, i - 1))
            position.append(block)
    return position


if __name__ == '__main__':
    img = MyOrc('./res/original/1.jpg', './res/out/')
    # cc
    img.pre_processing()
    img.get_position()
    print(img.position)
    for i, line in enumerate(img.position):
        for i_c, char in enumerate(line["characters"]):
            image = img.get_image_by_position(char["position"]["coordinate"])
            img.save('char_{}_{}.jpg'.format(i, i_c), image)
