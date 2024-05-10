import cv2
import numpy as np

def cvPlotXyAxisTicks(img=None,  # back ground image, e.g., 255*np.ones((800,200,3), dtype=np.uint8) 
                      xlim=None, # xlim, x limits, e.g., (0, 100)
                      ylim=None, # ylim, y limits, e.g., (-1, 1)
                      axisThick=None, # thickness of axis, e.g., 3
                      axisColor=None, # color of axis and ticks, e.g., (0,0,0)
                      tickDx=None, # interval of ticks along x, e.g., 10.0
                      tickDy=None, # interval of ticks along y, e.g., 0.1
                      tickThick=None, # thickness of tick line (in pixel), e.g., 2
                      tickColor=None, # color of ticks, e.g., (64,64,64)
                      grid=None, # plot grid lines, e.g., True
                      gridThick=None, # thickness of grid line (in pixel), e.g., 1
                      gridColor=None, # color of grid line, e.g., (128, 128, 128)
):
    if type(img) == type(None):
        img = 255 * np.ones((200, 800, 3))
    if type(xlim) == type(None):
        xlim = (0., 100.)
    if type(ylim) == type(None):
        ylim = (-1., 1.)
    if type(axisThick) == type(None):
        axisThick = 3
    if type(axisColor) == type(None):
        axisColor = (0, 0, 0)
    if type(tickDx) == type(None):
        tickDx = 10.0
    if type(tickDy) == type(None):
        tickDy = 0.1
    if type(tickThick) == type(None):
        tickThick = 2
    if type(tickColor) == type(None):
        tickColor = (64,64,64)
    if type(grid) == type(None):
        grid = True
    if type(gridThick) == type(None):
        gridThick = 1
    if type(gridColor) == type(None):
        gridColor = (128, 128, 128)

    def px2x(px):
        return xlim[0] + (px / img.shape[1]) * (xlim[1] - xlim[0])
    def py2y(py):
        return ylim[0] + (py / img.shape[0]) * (ylim[1] - ylim[0])
    def x2px(x):
        return img.shape[1] * (x - xlim[0]) / (xlim[1] - xlim[0])
    def y2py(y):
        return img.shape[0] * (y - ylim[1]) / (ylim[0] - ylim[1])

    # draw x axis (xlim[0], 0) - (xlim[1], 0)
    p1 = np.array( (x2px(xlim[0]), y2py(0)), dtype=np.int32)
    p2 = np.array( (x2px(xlim[1]), y2py(0)), dtype=np.int32)
    cv2.line(img, p1, p2, color=(axisColor), thickness=axisThick)

    # draw y axis (0, ylim[0]) - (0, ylim[1])
    p1 = np.array( (x2px(0), y2py(ylim[0])), dtype=np.int32)
    p2 = np.array( (x2px(0), y2py(ylim[1])), dtype=np.int32)
    cv2.line(img, p1, p2, color=(axisColor), thickness=axisThick)


    return img


if __name__ == '__main__':
    cv2.imshow('TEST', cvPlotXyAxisTicks(axisColor=(128,0,0), xlim=(-10,90)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    