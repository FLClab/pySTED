import numpy
import random
import math

from matplotlib import pyplot
from skimage import draw, measure, io
from scipy import spatial
from tqdm import trange

from pysted import _draw

TIMESTEP = 1

class Nodes:
    """
    A ``Nodes`` object is responsible to interact with a list of nodes and apply
    different forces or jitters to the a single node
    """
    def __init__(self, nodes, parent=None):
        if isinstance(nodes, (tuple, list)):
            nodes = numpy.array(nodes)
            if nodes.ndim < 2:
                nodes = nodes[numpy.newaxis, :]

        self.parent = parent

        self.nodes_position = nodes.astype(numpy.float32)
        self.nodes_id = numpy.arange(len(nodes))
        self.nodes_speed = numpy.zeros_like(nodes, dtype=numpy.float32)
        self.nodes_acc = numpy.zeros_like(nodes, dtype=numpy.float32)

    def update(self):
        """
        Updates the position of each node based on the current forces and speeds
        """
        self.nodes_speed += self.nodes_acc * TIMESTEP
        self.nodes_position += self.nodes_speed * TIMESTEP

    def apply_force(self, mean=0., std=0.1, field=None):
        """
        This method allows to apply a ``force`` on each nodes

        :param mean: The `mean` parameter of the normal function to calculate the force
        :param mean: The `std` parameter of the normal function to calculate the force
        :param field: (Optional) A `numpy.ndarray` of the force field
        """
        if isinstance(field, numpy.ndarray):
            force = field.astype(numpy.float32)
        else:
            force = numpy.random.normal(loc=mean, scale=std, size=self.nodes_position.shape[-1])
            force = force.astype(numpy.float32)

        # Here we apply a basic physic force on the nodes
        self.nodes_acc += force * TIMESTEP

    def reset_force(self):
        """
        Reset the current ``force`` that is being applied on all nodes
        """
        self.nodes_acc = numpy.zeros_like(self.nodes_acc)

    def reset_speed(self):
        """
        Reset the current ``speed`` that is being applied on all nodes
        """
        self.nodes_speed = numpy.zeros_like(self.nodes_speed)

    def apply_jitter(self, mean=0., std=0.1):
        """
        This method allows to jitter the position of each nodes.

        :param mean: The ``mean`` parameter of the normal function to calculate jitter
        :param mean: The ``std`` parameter of the normal function to calculate jitter
        """
        jitter = numpy.random.normal(loc=mean, scale=std, size=self.nodes_position.shape)
        jitter = jitter.astype(numpy.float32)

        # Here we simply change the position of each nodes
        self.nodes_position += jitter

    def add_node(self, node, pos="tail"):
        """
        Methods that implements adding a node to the current nodes. 
        
        We copy the speed and acc of next node

        :param node: A (y, x) coordinates of the node to add
        :param pos: (Optional) Where to add the node. Should be in {"tail", "head"}
        """
        assert (pos in {"tail", "head"}) or (isinstance(pos, (int, float))), \
            "The current pos : `{}` is not supported".format(pos)
        if pos == "tail":
            pos, copy_index = len(self.nodes_position), -1
        elif pos == "head":
            pos, copy_index = 0, 0
        else:
            copy_index = pos

        self.nodes_position = numpy.insert(self.nodes_position, pos, node, axis=0)
        self.nodes_id = numpy.insert(self.nodes_id, pos, len(self.nodes_id) + 1, axis=0)
        self.nodes_speed = numpy.insert(self.nodes_speed, pos, self.nodes_speed[copy_index], axis=0)
        self.nodes_acc = numpy.insert(self.nodes_acc, pos, self.nodes_acc[copy_index], axis=0)

    def return_shape(self):
        """
        Should be implemented in the supered classes
        """
        raise NotImplementedError

class NodesCombiner:
    """
    A ``NodesCombiner`` object is responsible to combine multiple ``Nodes`` object.

    We store the combined objects into an objects list.
    """
    def __init__(self):

        self.objects = []

    def add_object(self, obj):
        """
        Allows to add a ``Nodes`` object to the ``NodesCombinator``

        :param obj: A ``Nodes`` obj
        """
        self.objects.append(obj)

    def update(self, *args):
        """
        Updates the position of each node based on the current forces and speeds
        """
        for obj in self.objects:
            obj.update(*args)

    def apply_force(self, *args):
        """
        This method allows to apply a ``force`` on each nodes
        """
        for obj in self.objects:
            obj.apply_force(*args)

    def apply_jitter(self, *args):
        """
        This method applies a ``jitter`` on each nodes
        """
        for obj in self.objects:
            obj.apply_jitter(*args)

    def reset_force(self, *args):
        """
        This method allows to reset the ``force`` currently applied on each node
        """
        for obj in self.objects:
            obj.reset_force(*args)

    def reset_speed(self, *args):
        """
        This method allows to reset the ``speed`` of each node
        """
        for obj in self.objects:
            obj.reset_speed(*args)

class Polygon(Nodes):
    """
    A ``Polygon`` is a set of ``Nodes`` that are closed to the exterior world
    """
    def __init__(self, coords=None, random_params={}, parent=None):
        """
        Instantiates the Polygon class
        :param coords: (Optional) A (N, 2) ``numpy.ndarray`` of the coordinates of
                       the polygon
        :param random_params: (Optional) A ``dict`` of paramters to generate the random
                              polygons
        """
        if isinstance(coords, type(None)):
            coords = self.generate_random(**random_params)
        super().__init__(coords, parent=parent)

    def return_shape(self, shape=None):
        """
        Return the polygon indices

        :param shape: The shape of the field of view
        :return : A ``numpy.ndarray`` of row coords
                   A ``numpy.ndarray`` of col coords
        """
        return draw.polygon(*(self.nodes_position).T.astype(int), shape=shape)

    def generate_random(self, num_points=(5, 25), scale=(10, 15), pos=((0, 0), (0, 0))):
        """
        Generates a random set of coords. 
        
        To do so, we make use of the Convex Hull of a random set of points

        :param num_points: Uniformly generates a number of points for the convex
                           hull between num_points[0] and num_points[1]
        :param scale: Uniformly generates a scale factor for the convex
                      hull between scale[0] and scale[1]
        :param pos: Uniformly positions the convex hull in space. Should be a ``tuple``
                    of top-left and bottom-right corner in (y, x) coordinates
        :return: A (N,2) ``numpy.ndarray`` of coordinates
        """
        # Creates a ConvexHull of a random set of points
        random_points = numpy.random.rand(random.randrange(*num_points), 2)
        hull = spatial.ConvexHull(random_points)
        coords = hull.points[hull.vertices]
        # Scales the coords of the convex hull
        coords = coords * random.uniform(*scale)
        # Random position of the coords
        coords = coords + [random.uniform(*yx) for yx in zip(*pos)]
        return coords

    def expand(self, scale=0.1):
        """
        Implements an expand method of the ``Polygon``. 
        
        We assume that the polygon is exapanded from its center of mass

        :param scale: The scale factor to expand the polygon
        """
        coords = self.nodes_position

        field = numpy.zeros_like(self.nodes_acc)
        vec = coords - numpy.mean(coords, axis=0)
        norm_vec = vec / numpy.sqrt(numpy.sum(vec ** 2, axis=1))[:, numpy.newaxis]

        self.nodes_position += scale * norm_vec

    def area(self):
        """
        Calculates the area covered by the ``Polygon``. 
        
        We use the Shoelace formulation

        :return: The area of the polygon
        """
        x, y = self.nodes_position.T
        return 0.5 * numpy.abs(numpy.dot(x,numpy.roll(y,1))-numpy.dot(y,numpy.roll(x,1)))

    def update(self):
        """
        Updates the position of each node based on the current forces and speeds
        """
        self.nodes_speed += self.nodes_acc * TIMESTEP
        self.nodes_position += self.nodes_speed * TIMESTEP

        # If there is a parent we update the node appropriately
        if not isinstance(self.parent, type(None)):
            parent, node_id = self.parent
            index = numpy.argwhere(parent.nodes_id == node_id).ravel()[0]
            coord = parent.nodes_position[index]

            # Moves the centroid of the polygon to the parent node
            delta = coord - numpy.mean(self.nodes_position, axis=0)
            self.nodes_position += delta

class Fiber(Nodes):
    """
    A ``Fiber`` is a set of nodes that are connected but not closed
    """
    def __init__(self, coords=None, random_params={}, parent=None, seed=None):
        """
        Instantiates the ``Fiber`` class

        :param coords: A ``numpy.ndarray`` of the ``Fiber`` coordinates
        :param random_params: A ``dict`` of parameters to generate the random
        :param parent: A ``tuple`` of a parent ``Nodes`` with corresponding ``node_id``
                       We assume the parent to be the head of the ``Fiber``
        :param seed: (Optional) A seed to set for the random number generator
        """
        if isinstance(coords, type(None)):
            coords = self.generate_random(**random_params, seed=seed)

        super().__init__(coords, parent=parent)

    def generate_random(self, num_points=(10, 50), angle=(-math.pi/8, math.pi/8),
                              scale=(1, 5), pos=((0, 0), (0, 0)), seed=None):
        """
        Generates a random set of points. 
        
        To do so, we incrementaly add a point to list of coordinates. It could be 
        seen as building a serpent from the head to the tail.

        :param num_points: (min, max) values of the uniform sampling to generate
                           the number of points of the fiber
        :param angle: (min, max) values of the uniform sampling to generate the
                      angle from the previous angle
        :param scale: (min, max) values of the uniform sampling to generate the
                      displacement
        :param pos: Uniformly sample a position of the ``Fiber`` object. Should be
                    a ``tuple`` of top-left and bottom-right corner in (y, x) coordinates
        :return: A (N,2) ``numpy.ndarray`` of coordinates
        """
        if seed is not None:
            numpy.random.RandomState(seed)
            numpy.random.seed(seed)
            random.seed(seed)
        coords = [(0, 0)]
        self.angles = []
        for _ in range(random.randrange(*num_points)):
            # When we start we sample an angle in 0, 2pi
            if len(coords) == 1:
                prev_angle = 0
                self.angles.append(prev_angle)
                delta, ang = random.uniform(*scale), random.uniform(0, 2*math.pi)
            else:
                # Calcultes the previous angle in range [0,2pi]
                dy = coords[-1][0] - coords[-2][0]
                dx = coords[-1][1] - coords[-2][1]
                prev_angle = (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
                delta, ang = random.uniform(*scale), random.uniform(*angle)
            # The angle is relative to the previous angle
            self.angles.append(ang)
            ang = ang + prev_angle
            # Calculates the next position
            prev_y, prev_x = coords[-1]
            dy, dx = delta * math.sin(ang), delta * math.cos(ang)
            coords.append((prev_y + dy, prev_x + dx))
        # Random position of the coords
        coords = numpy.array(coords)
        coords = coords + [random.uniform(*yx) for yx in zip(*pos)]
        return coords

    def grow(self, prob=0.5, angle=(-math.pi/8, math.pi/8), scale=(2, 3)):
        """
        This methods implements the growth of a ``Fiber``

        :param angle: (min, max) values of the uniform sampling to generate the
                      angle from the previous angle
        :param scale: (min, max) values of the uniform sampling to generate the
                      displacement
        """
        # Growth from the tail
        if random.random() < prob:
            self._grow_tail(angle, scale)
        # Growth from the head
        if (random.random() < prob) & (isinstance(self.parent, type(None))):
            self._grow_head(angle, scale)

    def spawn(self, num=(2, 10)):
        """
        Method that implements spawn of synapses

        :param num: (min, max) values of the number of synapses to spawn
        """
        coords = self.nodes_position
        # Avoids sampling edge nodes
        choices = numpy.random.choice(self.nodes_id[1:-1], size=min(len(coords), random.randrange(*num)), replace=False)
        for choice in choices:
            # Finds the index of the nodes
            index = numpy.argwhere(self.nodes_id == choice).ravel()[0]
            # Calcultes the angle in range [0,2pi]
            dy = coords[index + 1][0] - coords[index - 1][0]
            dx = coords[index + 1][1] - coords[index - 1][1]
            angle = (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
            if random.random() < 0.5:
                angle = angle - math.pi/2
            else:
                angle = angle + math.pi/2
            yield Synapse(neck_coord=coords[index], neck_direction=angle,
                            parent=(self, choice))

    def return_shape(self, shape=None):
        """
        Return ``Fiber`` indices.

        :param shape: The shape of the field of view
        :return : A `numpy.ndarray` of row coords
                   A `numpy.ndarray` of col coords
        """
        coords = self.nodes_position.astype(int)
        # coords = measure.subdivide_polygon(self.nodes_position, degree=2, preserve_ends=True).astype(int)
        # keep = (coords >= 0) & (coords < numpy.array(shape)[numpy.newaxis, :])
        # coords = coords[numpy.all(keep, axis=1)]

        # This code is now implemented in cython, but not much faster
        rows, cols = [], []
        for i in range(len(coords) - 1):
            r0, c0 = coords[i]
            r1, c1 = coords[i + 1]
            if (0 <= r0 < shape[0]) & (0 <= c0 < shape[1]) & \
               (0 <= r1 < shape[0]) & (0 <= c1 < shape[1]):
                rr, cc = draw.line(r0, c0, r1, c1)
                rows.extend(rr)
                cols.extend(cc)
        return numpy.array(rows), numpy.array(cols)

        # lines = numpy.stack(_draw._multiple_lines(coords), axis=1)
        # # Keeps only valid lines
        # return lines[numpy.all((lines >= [0, 0]) & (lines < shape), axis=1)].T

    def _grow_head(self, angle, scale):
        """
        Implements the growth of the head of the ``Fiber``

        :param angle: (min, max) values of the uniform sampling to generate the
                      angle from the previous angle
        :param scale: (min, max) values of the uniform sampling to generate the
                      displacement
        """
        # gets the current coordinates
        coords = self.nodes_position
        # Calcultes the previous angle in range [0,2pi]
        dy = coords[0][0] - coords[1][0]
        dx = coords[0][1] - coords[1][1]
        prev_angle = (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
        delta, ang = random.uniform(*scale), random.uniform(*angle)
        # The angle is relative to the previous angle
        ang = ang + prev_angle
        # Calculates the next position
        prev_y, prev_x = coords[0]
        dy, dx = delta * math.sin(ang), delta * math.cos(ang)

        self.add_node((prev_y + dy, prev_x + dx), "head")

    def _grow_tail(self, angle, scale):
        """
        Implements the growth of the tail of the ``Fiber``

        :param angle: (min, max) values of the uniform sampling to generate the
                      angle from the previous angle
        :param scale: (min, max) values of the uniform sampling to generate the
                      displacement
        """
        # gets the current coordinates
        coords = self.nodes_position

        # Calcultes the previous angle in range [0,2pi]
        dy = coords[-1][0] - coords[-2][0]
        dx = coords[-1][1] - coords[-2][1]
        prev_angle = (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
        delta, ang = random.uniform(*scale), random.uniform(*angle)
        # The angle is relative to the previous angle
        ang = ang + prev_angle
        # Calculates the next position
        prev_y, prev_x = coords[-1]
        dy, dx = delta * math.sin(ang), delta * math.cos(ang)

        self.add_node((prev_y + dy, prev_x + dx), "tail")

    def update(self):
        """
        Updates the position of each node based on the current forces and speeds
        """
        self.nodes_speed += self.nodes_acc * TIMESTEP
        self.nodes_position += self.nodes_speed * TIMESTEP

        # If there is a parent we update the node appropriately
        if not isinstance(self.parent, type(None)):
            parent, node_id = self.parent
            index = numpy.argwhere(parent.nodes_id == node_id).ravel()[0]
            self.nodes_position[0] = parent.nodes_position[index]
            self.nodes_speed[0] = parent.nodes_speed[index]
            self.nodes_acc[0] = parent.nodes_acc[index]

class Synapse(NodesCombiner):
    """
    A ``Synapse`` is a protuberance that starts from a ``Fiber``. 
    
    It is defined as the combination of a ``Fiber`` and a ``Polygon`` objects
    """
    def __init__(self, neck_coord, neck_direction, parent=None):
        """
        Instantiates the ``Synapse`` object

        :param neck_coord: A (y, x) coordinate of neck coord
        :param neck_direction: An angle towards which to generate the synapse
        :param parent: A `tuple` of the parent and node id
        """
        super().__init__()

        self.parent = parent

        if isinstance(neck_coord, (tuple, list)):
            neck_coord = numpy.array(neck_coord)
        if neck_coord.ndim < 2:
            neck_coord = neck_coord[numpy.newaxis, :]

        self.neck_direction = neck_direction
        self.neck = Fiber(neck_coord, parent=self.parent)
        self.head = None

        # Adds object to nodes combiner
        self.add_object(self.neck)

        self.lifecycle = 0 # Number of updates
        self.head_cycle_start = random.randrange(2, 3) # Number of cycles before head appears
        self.max_head_size = random.randrange(5, 10)

    def grow(self, angle=(-math.pi/8, math.pi/8), scale=(1, 2)):
        """
        Method that implements the growth of the synapse

        :param angle: (min, max) values of the uniform sampling to generate the angle
        :param scale: (min, max) values of the uniform sampling to generate the displacement
        """
        # While the head cycle has not started, we grow the neck
        if self.lifecycle <= self.head_cycle_start:
            coords = self.neck.nodes_position
            # We need to add a new coords to the Fiber on the first cycle
            if len(coords) == 1:
                delta, ang = random.uniform(*scale), random.uniform(*angle)
                # The angle is relative to the previous angle
                ang = ang + self.neck_direction
                # Calculates the next position
                prev_y, prev_x = coords[-1]
                dy, dx = delta * math.sin(ang), delta * math.cos(ang)
                self.neck.add_node((prev_y + dy, prev_x + dx), "tail")
            else:
                self.neck.grow(prob=1., angle=angle, scale=scale)
        else:
            if isinstance(self.head, type(None)):
                self.head = Polygon(random_params={
                    "scale" : (0, 1),
                    "pos" : (self.neck.nodes_position[-1], self.neck.nodes_position[-1])
                }, parent=(self.neck, self.neck.nodes_id[-1]))
                self.add_object(self.head)
            if self.head.area() < self.max_head_size:
                self.head.expand(scale=0.25)

        self.lifecycle += 1

    def return_shape(self, shape=None):
        """
        Return ``Synapse`` indices.

        :param shape: The shape of the field of view
        :return : A ``numpy.ndarray`` of row coords
                  A ``numpy.ndarray`` of col coords
        """
        rr, cc = [], []
        rows, cols = self.neck.return_shape(shape)
        rr.extend(rows), cc.extend(cols)
        if isinstance(self.head, Polygon):
            rows, cols = self.head.return_shape(shape)
            rr.extend(rows), cc.extend(cols)
        return numpy.array(rr), numpy.array(cc)

class Ensemble:
    """
    Creates an ``Ensemble`` object which is responsible to handle different objects
    that are based on a `Nodes` object
    """
    def __init__(self, roi=((0, 0), (256, 256))):
        """
        Instantiates an ``Ensemble`` object

        :param roi: A `tuple` defining the limits of the region of interest. ((miny, minx), (maxy, maxx))
        """
        self.objects = []
        self.roi = roi

    def generate_sequence(self, num_frames):
        """
        Generates the sequence of frames of the ``Ensemble``

        :param num_frames: The number of frames to generate
        :return : A 3D ``numpy.ndarray`` of the sequence
        """
        sequence = []
        for i in trange(num_frames, desc="Frames"):

            self.spawn(prob=0.02)
            self.reset_force()
            self.update(prob=0, force=(0., 0.), jitter=(0., 0.))
            if i % 10 == 0:
                self.reset_speed()

            sequence.append(self.return_frame())

        return numpy.array(sequence)

    def append(self, obj):
        """
        Appends the ``obj`` to then ``Ensemble``

        :param obj: The object to append
        """
        self.objects.append(obj)

    def update(self, prob=0.05, force=(0., 0.1), jitter=(0., 0.01)):
        """
        Updates all objects in the ``Ensemble``.
        
        If an object is out of the region of interest then it is simply removed

        :param prob: The probability of applying a force
        :param force: A ``tuple`` of the force to apply
        :param jitter: A ``tuple`` of the jitter to apply
        """
        for i in reversed(range(len(self))):
            obj = self.objects[i] # This will allow to remove out of roi objects
            if random.random() < prob:
                obj.apply_force(*force)
            if random.random() < prob:
                obj.apply_jitter(*jitter)

            if isinstance(obj, (Fiber, Synapse)):
                obj.grow()

            obj.update()

    def return_frame(self, image=None):
        """
        Creates the frame of the current state of the ``Ensemble``

        :param image: (Optional) An ``numpy.ndarray`` to write to
        :return: A 2D ``numpy.ndarray`` of the current state
        """
        if isinstance(image, type(None)):
            image = numpy.zeros(numpy.diff(self.roi, axis=0).ravel())

        # TODO: Change the value on the image depending on the current bleaching
        for i, obj in enumerate(self.objects):
            rr, cc = obj.return_shape(shape=image.shape)
            image[rr.astype(int), cc.astype(int)] = 5

        return image

    def generate_objects_dict(self, obj_type="all"):
        """
        Creates a dictionnary containing the pixels corresponding to the objects in the frame.
        This could be useful if we want to do something other than plot the ojbects, such as
        tagging them for flashing or something.

        :return: A ``dict`` of the objects
        """
        obj_dict = {}
        if obj_type == "all":
            # add tous les obj au dict
            for i, obj in enumerate(self.objects):
                obj_dict[i] = obj
        elif obj_type == "fibers":
            # add juste les fibres au dict
            for i, obj in enumerate(self.objects):
                if type(obj) == Fiber:
                    obj_dict[i] = obj
        elif obj_type == "synapses":
            # add juste les synapses au dict
            for i, obj in enumerate(self.objects):
                if type(obj) == Polygon:
                    obj_dict[i] = obj
        else:
            # mauvaise option, lancer une erreur
            pass

        return obj_dict

    def show(self, ax=None):
        """
        Implements a ``show`` function of the ensemble
        """
        if isinstance(ax, type(None)):
            fig, ax = pyplot.subplots()

        image = self.return_frame()
        ax.imshow(image)

    def spawn(self, prob=0.1):
        """
        Implements a ``spawn`` function to randomly spawn synapses

        :param prob: The probability of spawning a synapse
        """
        for i in reversed(range(len(self))):
            obj = self.objects[i]
            if isinstance(obj, Fiber):
                if random.random() < prob:
                    for synapse in obj.spawn(num=(1, 2)):
                        self.objects.append(synapse)

    def reset_force(self):
        """
        Applies the reset_force to all objects of the ``Ensemble``
        """
        for obj in self.objects:
            obj.reset_force()

    def reset_speed(self):
        """
        Applies the reset_force to all objects of the ``Ensemble``
        """
        for obj in self.objects:
            obj.reset_speed()

    def __getitem__(self, index):
        """
        Implements a ``__getitem__`` method

        :return: The object at the index
        """
        return self.objects[index]

    def __len__(self):
        """
        Implements a ``__len__`` method

        :return: The number of objects in the ``Ensemble``
        """
        return len(self.objects)

if __name__ == "__main__":

    # from tqdm import trange
    #
    # image = numpy.zeros((256, 256))
    #
    # ensemble = Ensemble(roi=((0, 0), image.shape))
    # for _ in trange(15):
    #     if random.random() < 0:
    #         obj = Polygon(random_params={
    #             "pos" : [(0, 0), image.shape]
    #         })
    #     else:
    #         obj = Fiber(random_params={
    #             "num_points" : (10, 50),
    #             "pos" : [(0, 0), image.shape]
    #         })
    #     ensemble.append(obj)
    #
    # pyplot.ion()
    # fig, ax = pyplot.subplots()
    # for i in trange(250):
    #     ax.clear()
    #     ensemble.update(force=(0., 0.), jitter=(0., 0.1))
    #     ensemble.show(ax=ax)
    #     if i % 10 == 0:
    #         ensemble.reset_force()
    #         ensemble.reset_speed()
    #     pyplot.pause(0.001)
    # pyplot.show()
    #
    # # for i in trange(10):
    # #     ensemble.update(force=(0., 0.), jitter=(0., 0.1))
    # #     if i % 10 == 0:
    # #         ensemble.reset_force()
    # #         ensemble.reset_speed()
    # pyplot.ioff()
    # fig, ax = pyplot.subplots()
    # ensemble.show(ax=ax)
    # pyplot.show()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="",
                        help="Which test to perform")
    args = parser.parse_args()

    from tqdm import trange
    image = numpy.zeros((256, 256))
    ensemble = Ensemble(roi=((0, 0), image.shape))
    for _ in trange(10, desc="Objects"):
        if args.mode == "":
            if random.random() < 0.5:
                obj = Polygon(random_params={
                            "pos" : [(0, 0), image.shape]
                })
            else:
                obj = Fiber(random_params={
                    "num_points" : (30, 50),
                    "pos" : [(0, 0), image.shape]
                })
        else:
            obj = Fiber(random_params={
                "num_points" : (30, 50),
                "pos" : [(0, 0), image.shape]
            })
        ensemble.append(obj)
    # ensemble.append(Synapse((14, 14), math.pi/4))

    if args.mode == "":
        fig, ax = pyplot.subplots(figsize=(10,10), tight_layout=True)
        ensemble.show(ax=ax)
        pyplot.show()
        exit()

    pyplot.ion()
    fig, ax = pyplot.subplots(figsize=(10,10), tight_layout=True)
    for i in trange(100):
        ax.clear()

        ensemble.reset_force()
        if i % 10 == 0:
            ensemble.reset_speed()

        if args.mode == "ctrl":
            # CTRL
            ensemble.spawn(prob=0.02)
            ensemble.update(prob=0.1, force=(0., 0.), jitter=(0., 0.))
        elif args.mode == "ctrlwforce":
            # CTRL with force
            ensemble.spawn(prob=0.02)
            ensemble.update(prob=0.1, force=(0., 0.1), jitter=(0., 0.))
        elif args.mode == "ctrlwjitter":
            # CTRL with jitter
            ensemble.spawn(prob=0.02)
            ensemble.update(prob=0.1, force=(0., 0.1), jitter=(0., 0.))
        elif args.mode == "stim":
            # STIM
            if i < 50:
                ensemble.spawn(prob=0.02)
                ensemble.update(prob=0.1, force=(0., 0.), jitter=(0., 0.))
            else:
                ax.set_title("Stimulation Started")
                ensemble.spawn(prob=0.1)
                ensemble.update(prob=0.1, force=(0., 0.), jitter=(0., 0.))
        else:
            ensemble.update(prob=0., force=(0., 0.), jitter=(0., 0.))

        ensemble.show(ax=ax)

        pyplot.pause(0.001)
    pyplot.show()

    # frames = ensemble.generate_sequence(100)
    # io.imsave("temporalDatamap.tif", frames.astype(numpy.uint8), check_contrast=False)