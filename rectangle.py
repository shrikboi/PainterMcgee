import os
import re

"""
Classes and utilities to describe all of the game rectangles.
"""


class Rectangle(object):
    """
    A Rectangle is a collection of

    """

    def __init__(self, size, angle, edge_thickness):
        self.size = size
        self.angle = angle
        self.edge_thickness = edge_thickness


class RectangleList(object):
    """
    The RectangleList class stores a list of all of the Rectangles that can be used to draw
    """

    def __init__(self, fname=None):
        """
        Read the rectangles from the file <fname>

        File format must be:
        - For k in [0, n):
          - Line k - width height angle edge_thickness

        """
        self.rectangles = []
        directory = "layouts"
        if fname is not None:
            with open(os.path.join(directory, fname)) as f:
                lines = f.read().splitlines()

            for line in lines:
                line = re.split('\s', line)
                if len(line) != 4:
                    print(f"Every line in {fname} needs to be: width height angle edge_thickness")
                    continue
                try:
                    width = int(line[1])
                    height = int(line[0])
                    angle = int(line[2])
                    edge_thickness = int(line[3])
                except:
                    print(f"Every line in {fname} needs to be: width height angle edge_thickness")
                    continue
                # TODO more checks to see if file is okay (deg between 0 and 360 etc)
                self.rectangles.append(Rectangle((width, height), angle, edge_thickness))

            self.num_rectangles = len(self.rectangles)

    def get_rectangle(self, n):
        """
        Return rectangle <n> from this list.
        """
        if n < 0 or n >= len(self.rectangles):
            raise ValueError("Can't retrieve rectangle %d" % n)
        return self.rectangles[n]

    def __iter__(self):
        return self.rectangles.__iter__()
