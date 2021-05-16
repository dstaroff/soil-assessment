import math
import random
import sys

import shapefile
from shapely.geometry import Polygon, Point

from util import Const

# step size for angles (in degrees); has linear impact on running time
MIN_WIDTH = 0.0
# Default value for the number of possible center points of the maximal rectangle
N_TRIES = 20
TOLERANCE = 0.05
ANGLE = math.radians(90)
ASPECT_RATIO = Const.YANDEX_MAPS_MAX_WIDTH / Const.YANDEX_MAPS_MAX_HEIGHT
EPSILON = 1e-9


def _get_raw_origins(polygon: Polygon) -> ([Point], float):
    # get the width of the bounding box of the original polygon to determine tolerance
    minx, miny, maxx, maxy = polygon.bounds

    # simplify polygon
    tolerance = min(maxx - minx, maxy - miny) * TOLERANCE
    if tolerance > 0:
        polygon = polygon.simplify(tolerance)

    # get the width of the bounding box of the simplified polygon
    minx, miny, maxx, maxy = polygon.bounds
    boxWidth, boxHeight = maxx - minx, maxy - miny

    # discretize the binary search for optimal width to a resolution of this times the polygon width
    widthStep = min(boxWidth, boxHeight) / 50

    # populate possible center points with random points inside the polygon
    origins = []

    # get the centroid of the polygon
    centroid = polygon.centroid

    if centroid.within(polygon):
        origins.append(centroid)

    # get few more points inside the polygon
    while len(origins) < N_TRIES:
        rndX = random.random() * boxWidth + minx
        rndY = random.random() * boxHeight + miny
        rndPoint = Point(rndX, rndY)
        if rndPoint.within(polygon):
            origins.append(rndPoint)

    return origins, widthStep


def _get_modified_origins(origins: [Point], polygon: Polygon) -> [Point]:
    modified_origins = []

    for origin in origins:
        # generate improved origins
        p1W, p2W = _intersection_points(polygon, origin, ANGLE)
        p1H, p2H = _intersection_points(polygon, origin, ANGLE + math.radians(90))

        if p1W is not None and p2W is not None:
            # average along width axis
            modified_origins.append(
                Point(
                    (p1W.x + p2W.x) / 2,
                    (p1W.y + p2W.y) / 2
                )
            )
        if p1H is not None and p2H is not None:
            # average along height axis
            modified_origins.append(
                Point(
                    (p1H.x + p2H.x) / 2,
                    (p1H.y + p2H.y) / 2
                )
            )

    return modified_origins


def _get_origins(polygon: Polygon) -> ([Point], float):
    origins, widthStep = _get_raw_origins(polygon)

    return _get_modified_origins(origins, polygon), widthStep


max_area: float
max_rect: {str, float}


def _find_max_inner_rect_for_origin(origin: Point, polygon: Polygon, max_width: float, max_height: float,
                                    width_step: float):
    global max_area, max_rect

    # do a binary search to find the max width that works
    left = max(MIN_WIDTH, math.sqrt(max_area * ASPECT_RATIO))
    right = min(max_width, max_height * ASPECT_RATIO)
    if right * max_height < max_area:
        return None

    while (right - left) >= width_step:
        width = (left + right) / 2
        height = width / ASPECT_RATIO
        rectPoly = Polygon([
            Point(origin.x - width / 2, origin.y - height / 2),
            Point(origin.x + width / 2, origin.y - height / 2),
            Point(origin.x + width / 2, origin.y + height / 2),
            Point(origin.x - width / 2, origin.y + height / 2),
        ])
        if rectPoly.within(polygon):
            # we know that the area is already greater than the maxArea found so far
            max_area = width * height
            max_rect = {
                'cx': origin.x,
                'cy': origin.y,
                'width': width,
                'height': height,
            }

            # increase the width in the binary search
            left = width
        else:
            # decrease the width in the binary search
            right = width


def max_inner_rect(shape):
    global max_area, max_rect

    points = shape.points
    polygon = Polygon(points)

    origins, width_step = _get_origins(polygon)

    max_area = 0
    max_rect = dict()

    for origin in origins:
        p1W, p2W = _intersection_points(polygon, origin, ANGLE)
        p1H, p2H = _intersection_points(polygon, origin, ANGLE + math.pi / 2)

        if p1W is None or p2W is None or \
                p1H is None or p2H is None:
            continue

        minSqDistW = min(_squared_dist(origin, p1W), _squared_dist(origin, p2W))
        minSqDistH = min(_squared_dist(origin, p1H), _squared_dist(origin, p2H))
        max_width = 2 * math.sqrt(minSqDistW)
        max_height = 2 * math.sqrt(minSqDistH)

        if max_width * max_height < max_area:
            continue

        _find_max_inner_rect_for_origin(origin, polygon, max_width, max_height, width_step)

    width = max_rect['width']
    height = max_rect['height']

    half_width = width / 2
    half_height = height / 2

    inner_rect = Polygon([
        Point(max_rect['cx'] - half_width, max_rect['cy'] - half_height),
        Point(max_rect['cx'] - half_width, max_rect['cy'] + half_height),
        Point(max_rect['cx'] + half_width, max_rect['cy'] + half_height),
        Point(max_rect['cx'] + half_width, max_rect['cy'] - half_height),
    ])

    return inner_rect


def load_shapes_and_records():
    myshp = open(Const.SPATIAL_SHP_FILE_PATH, 'rb')
    myshx = open(Const.SPATIAL_SHX_FILE_PATH, 'rb')
    mydbf = open(Const.SPATIAL_DBF_FILE_PATH, 'rb')

    reader = shapefile.Reader(shp=myshp, shx=myshx, dbf=mydbf)

    shapes = reader.shapes()
    records = reader.records()

    return shapes, records


def _squared_dist(a: Point, b: Point) -> float:
    '''
    Returns the squared euclidean distance between points `a` and `b`

    :param a: Point A
    :param b: Point B
    :return: Squared Euclidean distance between points A and B
    '''
    delta_x = b.x - a.x
    delta_y = b.y - a.y

    return delta_x * delta_x + delta_y * delta_y


def _line_intersection(p1: Point, q1: Point, p2: Point, q2: Point) -> Point or None:
    '''
    Finds the intersection point (if there is one) of the lines `p1q1` and `p2q2`

    :param p1: Line #1 start point
    :param q1: Line #1 end point
    :param p2: Line #2 start point
    :param q2: Line #2 end point
    :return: Point of intersection
    '''
    # allow for some margins due to numerical errors
    eps = 1e-9
    # find the intersection point between the two infinite lines
    dx1 = p1.x - q1.x
    dy1 = p1.y - q1.y
    dx2 = p2.x - q2.x
    dy2 = p2.y - q2.y
    denom = dx1 * dy2 - dy1 * dx2

    if math.fabs(denom) < eps:
        return None

    cross1 = p1.x * q1.y - p1.y * q1.x
    cross2 = p2.x * q2.y - p2.y * q2.x

    px = (cross1 * dx2 - cross2 * dx1) / denom
    py = (cross1 * dy2 - cross2 * dy1) / denom

    return Point(px, py)


def _is_point_in_segment_box(p: Point, p1: Point, q1: Point) -> bool:
    '''
    Checks whether the point `p` is inside the bounding box of the line segment `p1q1`

    :param p: Check point
    :param p1: Line segment start point
    :param q1: Line segment end point
    :return: `True` if point is in segment box. `False` otherwise
    '''
    # allow for some margins due to numerical errors
    eps = 1e-9
    if p.x < min(p1.x, q1.x) - eps or \
            p.x > max(p1.x, q1.x) + eps or \
            p.y < min(p1.y, q1.y) - eps or \
            p.y > max(p1.y, q1.y) + eps:
        return False

    return True


def _intersection_points(poly: Polygon, origin: Point, alpha: float) -> (Point, Point):
    '''
    Gives the 2 closest intersection points between a ray with `alpha` radians from the `origin` and the `polygon`.
    The two points should lie on opposite sides of the origin

    :param poly: Polygon
    :param origin: Origin point
    :param alpha: Angle
    :return: 2 closest points between a ray with `alpha` radians from the `origin` and the `polygon`
    '''
    origin = Point(origin.x + EPSILON * math.cos(alpha), origin.y + EPSILON * math.sin(alpha))
    shiftedOrigin = Point(origin.x + math.cos(alpha), origin.y + math.sin(alpha))

    idx = 0
    if math.fabs(shiftedOrigin.x - origin.x) < EPSILON:
        idx = 1

    i = 0
    n = len(poly.exterior.coords)
    b = Point(poly.exterior.coords[n - 1])
    minSqDistLeft = sys.float_info.max
    minSqDistRight = sys.float_info.max
    closestPointLeft = None
    closestPointRight = None
    while i < n:
        a = Point(b)
        b = Point(poly.exterior.coords[i])

        p = _line_intersection(origin, shiftedOrigin, a, b)
        if p is not None and _is_point_in_segment_box(p, a, b):
            sqDist = _squared_dist(origin, p)
            if p.coords[0][idx] < origin.coords[0][idx]:
                if sqDist < minSqDistLeft:
                    minSqDistLeft = sqDist
                    closestPointLeft = Point(p)
            elif p.coords[0][idx] > origin.coords[0][idx]:
                if sqDist < minSqDistRight:
                    minSqDistRight = sqDist
                    closestPointRight = Point(p)
        i += 1

    return closestPointLeft, closestPointRight
