import cv2
import numpy as np

class Grid:
    def __init__(self, filename):
        self.filename = filename
        self.image = cv2.imread(filename)

    def getGrid(self):
        # cv2.imshow("im", self.image)
        img = self.image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thr = self._threshold(gray)
        cv2.imshow("thr", thr)
        cv2.waitKey()
        lines = self._lines(thr)
        drawlines2(img, lines)
        self._grid(lines)

    def _threshold(self, img):
        mblur = cv2.medianBlur(img, 5)
        mmin, mmax = np.min(mblur), np.max(mblur)
        mblur = np.uint8((mblur - mmin) * 255.0 / (mmax - mmin))
        thr = cv2.adaptiveThreshold(mblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 10)
        return thr

    def _lines(self, thr):
        color = np.copy(self.image)
        lines = cv2.HoughLines(thr, 1, np.pi / 180, 200)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(color, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return lines[0]

    def _grid(self, lines):
        th = setgrid(lines, np.copy(self.image))

rad90 = np.pi / 2
rad180 = np.pi
rad360 = np.pi * 2
rad45 = np.pi / 4
deg2rad = rad180 / 180
rad2deg = 1 / deg2rad

theta_wind = 10 / 180.0 * rad180
rho_period = 93
rho_wind = 20

# Todo: minimize sine, cosine calculations

def setgrid(lines, img):
    o_rho = [l[0] for l in lines]
    o_phi = [l[1] for l in lines]
    phi = o_phi[:]

    # theta meanshift
    theta, indices = periodicmean(phi, None, rad90, theta_wind)

    # separate data from error
    assert indices is not None
    rho = [o_rho[i] for i in indices]
    phi = [o_phi[i] for i in indices]

    rho1 = []
    phi1 = []
    rho2 = []
    phi2 = []

    # separate horizontal and vertical
    for i in range(0, len(rho)):
        if theta - rad45 < phi[i] < theta + rad45:
            phi1.append(phi[i])
            rho1.append(rho[i])
        elif phi[i] > theta + 3 * rad45:
            phi1.append(phi[i] - rad180)
            rho1.append(-rho[i])
        else:
            phi2.append(phi[i])
            rho2.append(rho[i])

    # final theta values
    theta1 = theta
    theta2 = (theta + rad90)

    pts = [[intersect(rho1[i], phi1[i], rho2[j], phi2[j]) for j in range(len(rho2))] for i in range(len(rho1))]

    for row in pts:
        for x,y in row:
            drawpoint(img, x, y, 10, (255, 0, 0), 1)

    drawlines(img, rho1, phi1, (0, 128, 255))  # narancs
    drawlines(img, rho2, phi2, (255, 128, 0))  # vilagoskek
    cv2.imshow("lines", img)
    cv2.waitKey()

def intersect(rho1, phi1, rho2, phi2):
    A = np.array([[np.cos(phi1), np.sin(phi1)],
                  [np.cos(phi2), np.sin(phi2)]])
    b = np.array([[rho1, rho2]]).T
    pt = np.linalg.solve(A, b).reshape((2,))
    return pt[0], pt[1]

def adjustLine(rho, phi, theta, size):
    s = np.sin(phi)
    c = np.cos(phi)
    if s == 0: s = 0.001
    if c == 0: c = 0.001

    h, w, ch = size
    pts = [(0, rho/s), (rho/c, 0), (w, (rho-w*c)/s), ((rho-h*s)/c, h)]
    pts = [pt for pt in pts if w + 1 > pt[0] > -1 and h + 1 > pt[1] > -1]

    x1, y1 = pts[0]
    x2, y2 = pts[1]
    mx, my = (x1 + x2)/2, (y1 + y2)/2
    st, ct = np.sin(theta), np.cos(theta)
    rho2 = mx*ct + my*st
    return rho2

def drawdebug(img, theta1, theta2, arho1, arho2, o_theta, size):
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s2 = np.sin(theta2)
    c2 = np.cos(theta2)
    xx = (arho1 * s2 - arho2 * s1) / (c1 * s2 - c2 * s1)
    yy = (arho1 - xx * c1) / s1

    center = (size[1] / 2, size[0] / 2)
    cpos = (center[0] + xx, center[1] + yy)

    dx1 = -s2 * rho_period
    dy1 = c2 * rho_period
    dx2 = c2 * rho_period
    dy2 = s2 * rho_period
    xs = []
    ys = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            xs.append(xx + i * dx1 + j * dx2)
            ys.append(yy + i * dy1 + j * dy2)
    for i in range(0, len(xs)):
        drawpoint(img, xs[i], ys[i], 6, (0, 0, 255), 3)

def periodicmean(array, initial, period, meanwindow, meanshift_it=2):
    n = len(array)
    if n == 0:
        return initial, None

    # wrapping to [0,period), wrapping again if samples might overlap
    onedge = 0
    for i in range(0, n):
        array[i] = array[i] % period
        if array[i] > (period * 0.8) or array[i] < (period * 0.2):
            onedge += 1

    if onedge > n / 2:
        # print "ONEDGE"
        for i in range(0, n):
            array[i] = (array[i] + period / 2) % period - period / 2

    # initial guess
    if initial is None:
        initial = 0.0
        for a in array:
            initial += a
        initial /= n
    else:
        if onedge > n / 2:
            initial = (initial + period / 2) % period - period / 2

    # mean shifting
    indices = None
    for i in range(0, meanshift_it):
        initial, indices = meanshift(array, initial, meanwindow)
    return initial, indices

def meanshift(array, startpos, window):
    sum = 0
    num = 0
    indices = []
    allidx = range(0, len(array))
    for i in allidx:
        el = array[i]
        if abs(el - startpos) <= (window / 2):
            sum += el
            num += 1
            indices.append(i)
    if num == 0:
        
        for i in allidx:
            sum += array[i]
            num += 1
        indices = allidx
    sum /= num
    return sum, indices

def drawvector(tex, pos, color, imgsize, pointsize, thickness=1):
    x = pos[0]
    y = pos[1]
    x1 = imgsize[1] / 2
    y1 = imgsize[0] / 2
    x2 = x1 + int(x)
    y2 = y1 + int(y)
    drawpoint(tex, x2, y2, pointsize, color, thickness)
    cv2.line(tex, (x1, y1), (x2, y2), color, thickness)

def drawvecformorigin(tex, pos, color, imgsize, pointsize, thickness=1):
    drawpoint(tex, pos[0], pos[1], pointsize, color, thickness)
    cv2.line(tex, (0, 0), (int(pos[0]), int(pos[1])), color, thickness)

def multiplyvector(vec1, scalar):
    return (vec1[0] * scalar, vec1[1] * scalar)

def addvector(vec1, vec2):
    return (vec1[0] + vec2[0], vec1[1] + vec2[1])

def rotatevector(vector, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return rotatevector_cs(vector, s, c)

def rotatevector_cs(vector, sinangle, cosangle):
    c = cosangle
    s = sinangle
    x = vector[0]
    y = vector[1]
    return (x * c - y * s, x * s + y * c)

def drawlines(tex, rho, phi, color, thickness=1):
    for r, p in zip(rho, phi):
        a = np.cos(p)
        b = np.sin(p)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(tex, (x1, y1), (x2, y2), color, thickness)

def drawlines2(img, lines):
    img = np.copy(img)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("img", img)
    cv2.waitKey()

def drawpoint(tex, x, y, size, color, thickness=1, circle = True):
    x = int(x)
    y = int(y)
    hs = size / 2
    if not circle:
        cv2.line(tex, (x - hs, y - hs), (x + hs, y - hs), color, thickness)
        cv2.line(tex, (x - hs, y - hs), (x - hs, y + hs), color, thickness)
        cv2.line(tex, (x + hs, y + hs), (x + hs, y - hs), color, thickness)
        cv2.line(tex, (x + hs, y + hs), (x - hs, y + hs), color, thickness)
    else:
        cv2.circle(tex, (x, y), size, color, thickness)

def closestperiod(orig, new, period):
    diff = orig - new
    diff -= diff % period
    result = new + diff
    if result + period - orig < orig - result:
        result += period
    reldiff = abs(result - orig) * 1.0 / period
    return result, reldiff

rho_conv = rho_period / 60.0

def pointtopic(pos, theta):
    return (-pos[0] * rho_conv, -pos[1] * rho_conv), -theta

def pointtoworld(pos, theta):
    return (-pos[0] / rho_conv, -pos[1] / rho_conv), -theta

def Main():
    return

if __name__ == '__main__':
    Main()

if __name__ == '__main__':
    fname = """D:\dokumentumok\Python\PySudoku\images\img1_1_rot.png"""
    fname = """D:\dokumentumok\Python\PySudoku\images\img1_3.jpg"""
    fname = """D:\dokumentumok\Python\PySudoku\images\ext5.jpg"""
    g = Grid(fname)
    g.getGrid()
