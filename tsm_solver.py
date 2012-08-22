import sys,os
from copy import deepcopy
import numpy
import random
import exceptions
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def of_dim(val, args):
    """
    creates an empty list of n sizes in n dimensions
    with all values initalized to a given value.
    """
    args.reverse()
    res = val
    for d in args:
        res = [res] * d
    return res

def read_tsp_file(fname):
    """
    Parses the file.
    """
    header = {
        'name': fname,
        'dimension': 0,
        'edgeweight': '',
        }

    in_data = False
    cities = []
    for L in open(fname, 'r'):
        data = L.strip().split(' ')
        if in_data and L != 'EOF' and len(data) == 3:
            try:
                cities.append((float(data[1]), float(data[2])))
            except exceptions.ValueError:
                pass
        else:
            if data[0] == "NAME":
                header['name'] = data[2]
            elif data[0] == "DIMENSION":
                header['dimension'] = int(data[2])
            elif data[0] == "EDGE_WEIGHT_TYPE":
                header['edgeweight'] = data[2]
            elif data[0] == "NODE_COORD_SECTION":
                in_data = True

    return cities, header

def read_tour_file(fname):
    """
    Parses the tour file.
    """
    header = {
        'name': fname,
        'dimension': 0,
        }

    in_data = False
    path = []
    for L in open(fname, 'r'):
        L = L.strip()
        data = L.split(' ')
        if in_data and L != 'EOF' and len(data) == 1 and L != '-1':
            try:
                path.append(int(data[0]))
            except exceptions.ValueError:
                pass

        else:
            if data[0] == "NAME":
                header['name'] = data[2]
            elif data[0] == "DIMENSION":
                header['dimension'] = int(data[2])
            elif data[0] == "TOUR_SECTION":
                in_data = True

    return path, header

class KohonenTSPSolverSettings(object):
    def __init__(self):
        # if the weighting dimension should be added
        self.add_weighting_dimension = True

        # if we should normalize the city data
        # None means don't normalize
        self.normalize_city_data = (-1, 1)

        # values for initial random data
        self.random_init_weights_range = (-1, 1)

        # debug mode?
        self.debug_mode = False

        # randomly shuffle cities?
        self.randomly_shuffle_cities = True

        # initial radius and gain
        self.init_radius = None
        self.init_gain = 0.5

        self.end_radius = 0
        self.end_gain = 0

        # learning algos
        self.favata_walker  = 1
        self.normal_kohonen = 2

        self.learning_algo = self.normal_kohonen

    def create_radius_schedule(self, num_cities, epocs):
        "sets up a radius schedule"
        if self.init_radius is None:
            init_radius = int(numpy.floor(num_cities * 0.1))
        else:
            init_radius = self.init_radius

        end_radius = 0

        # simply linearily reduce the radius
        self.radius_delta = (float(init_radius) - end_radius) / (epocs)

        # starting radius
        self.radius = init_radius
        return self.radius

    def radius_adj(self, epoc):
        "returns the amount to reduce the radius by"
        self.radius = self.radius - self.radius_delta
        return int(numpy.ceil(self.radius))

    def create_gain_schedule(self, num_cities, epocs):
        "sets up a gain schedule"
        init_gain = self.init_gain
        end_gain = 0. # maybe try 0.5

        self.gain_delta = (init_gain - end_gain) / (epocs)

        self.gain = init_gain
        return self.gain

    def gain_adj(self, epoc):
        #self.gain = self.gain - self.gain_delta
        return self.gain

class KohonenTSPSolver(object):
    def __init__(self, cities, settings):
        self.settings = settings

        if self.settings.add_weighting_dimension:
            self.cities = self._add_weighting_dimension(cities)
        else:
            self.cities = cities

        self.cities_real_values = deepcopy(self.cities)
        if self.settings.normalize_city_data:
            self.cities = self.normalize(*self.settings.normalize_city_data)

        self.outputs = len(self.cities)
        self.inputs = len(self.cities[0])

        # init the weights to a random range between 0 and 1
        self.weights = []
        for x in range(self.outputs):
            self.weights.append([])
            for y in range(self.inputs):
                self.weights[x].append(self._random(*self.settings.random_init_weights_range))

        if self.settings.debug_mode:
            print self.weights

    def _random(self, min, max):
        "Get a random value in a specified range"
        return (max - min) * random.random() + min

    def normalize(self, minimum, maximum):
        "Normalizes the citites using a the min-max method to a specified min-max."
        dim = len(self.cities[0])
        normed_parts = [] # store the individual dimensions
        for num in range(dim):
            p = [c[num] for c in self.cities]
            normed_p = map(lambda x: ((maximum-minimum)*((x-min(p))/(max(p)-min(p)))+minimum), p)
            normed_parts.append(normed_p)

        return zip(*normed_parts)

    def _euclidean_norm(self, elements):
        "takes the euclidean norm of a vector"
        return numpy.sqrt(sum(map(lambda x:pow(x,2),elements)))

    def _dot_product(self, a, b):
        "takes the dot product of two elements with the same number of dimensions"
        res = 0.0

        if len(a) != len(b):
            return None

        for x in range(len(a)):
            res += a[x] * b[x]

        return res

    def _add_weighting_dimension(self, cities):
        """
        This function takes a list of cities and returns a list of cities in three
        dimensions, with all having the same euclidean length.
        """
        sqrd_val = lambda x: pow(x[0],2) + pow(x[1],2)
        farthest = max(map(sqrd_val, cities))
        res = []
        for x in cities:
            res.append((x[0], x[1], numpy.sqrt(farthest - sqrd_val(x))))
        return res

    def _optimize_order(self, order):
        nodes = set(order.values())
        # reverse data structure
        order = [(x[1],x[0]) for x in order.iteritems()]
        print order


    def _run(self, city):
        "Take a city and return which output neuron lights up"
        activations = []
        if self.settings.learning_algo == self.settings.favata_walker:
            for x in xrange(self.outputs):
                activations.append(self._dot_product(city, self.weights[x]))
        elif self.settings.learning_algo == self.settings.normal_kohonen:
            for x in xrange(self.outputs):
                activations.append(sum(map(lambda x: pow(x[0] - x[1], 2), zip(city, self.weights[x]))))
        # This will return the activation with the highest value.
        # If two or more activations have the same exact value it returns the first one.
        maximum = max(activations)
        return activations.index(maximum)

    def run_all(self):
        "This function builds a mapping of all cities with their output activation node"
        orders = {}
        for x in enumerate(self.cities):
            orders[x] = self._run(x[1])
        #self._optimize_order(orders)
        return orders

    def get_tour(self):
        "This function executes run_all to get a city tour for the current weight set"
        mapping = sorted(self.run_all().items(), key=lambda x: x[1])
        if self.settings.debug_mode:
            print dict(mapping).values()
            print set(dict(mapping).values()), len(set(dict(mapping).values()))

        tour = []

        for x in mapping:
            tour.append(x[0][0])

        return tour

    def _train_element(self, gain, radius, city):
        if self.settings.debug_mode:
            print "-----------------------"
        # step 2: find activation neuron
        activation = self._run(city)

        # step 3: adjust the weights
        lower = (activation - radius) % self.outputs
        upper = (activation + radius) % self.outputs
        neurons = activation
        if lower < upper:
            neurons = range(lower, upper)
        else:
            neurons = range(upper, lower)
        for x in neurons:
            if self.settings.learning_algo == self.settings.favata_walker:
                top_term = map(lambda x: x[0] + x[1], zip(self.weights[x], map(lambda x: x*gain, city)))
                eu_norm = self._euclidean_norm(top_term)
            elif self.settings.learning_algo == self.settings.normal_kohonen:
                new_weights = map(lambda x: x*gain, map(lambda x: x[0] - x[1], zip(self.weights[x], city)))

            for y in xrange(self.inputs):
                weight_before = self.weights[x][y]

                if self.settings.learning_algo == self.settings.favata_walker:
                    self.weights[x][y] = top_term[y]/eu_norm
                elif self.settings.learning_algo == self.settings.normal_kohonen:
                    self.weights[x][y] += new_weights[y]

                if self.settings.debug_mode:
                    print "%dx%d -> %f -> %f" % (x,y,weight_before,self.weights[x][y])
        other_neurons = set(range(self.outputs)) - set(neurons)

    def train(self, epocs):
        #if self.settings.randomly_shuffle_cities:
        #    cities = deepcopy(self.cities)
        #    random.shuffle(cities)
        #else:
        cities = self.cities

        radius = self.settings.create_radius_schedule(len(cities), epocs)
        gain = self.settings.create_gain_schedule(len(cities), epocs)
        if self.settings.debug_mode:
            print "Start Radius: %d" % (radius,)
            print "Start Gain: %f" % (gain,)

            #for city in cities:
        for x in xrange(epocs):
            # step 1: select a city at random
            city = random.choice(cities)

            # a single epoc (steps 2-3)
            self._train_element(gain, radius, city)

            # step 4: update interaction index
            gain = self.settings.gain_adj(x)
            radius = self.settings.radius_adj(x)

        if self.settings.debug_mode:
            print "End Radius: %d" % (radius,)
            print "End Gain: %f" % (gain,)

    def _euclidean_distance(self, vect1, vect2):
        "takes the euclidean distance between two vectors"
        if len(vect1) != len(vect2):
            return False
        return numpy.sqrt(sum(map(lambda x:pow(vect2[x]-vect1[x],2),xrange(len(vect1)))))

    def tour_length(self, tour):
        cities = map(lambda x:self.cities_real_values[x],tour)
        last_city = None
        distance_traveled = 0.0
        for city in cities:
            if last_city is not None:
                distance_traveled += self._euclidean_distance(last_city, city)
            last_city = city

        distance_traveled += self._euclidean_distance(cities[-1], cities[0])

        return distance_traveled

    def plot(self, tour, title):
        tour = [self.cities_real_values[x] for x in tour]

        Path = mpath.Path

        fig = plt.figure()
        ax = fig.add_subplot(111)

        pathdata = []
        for x in tour:
            pathdata.append([Path.LINETO, (x[0], x[1])])
        pathdata[0][0] = Path.MOVETO
        pathdata.append([Path.CLOSEPOLY, (tour[0][0], tour[0][1])])

        codes, verts = zip(*pathdata)
        path = mpath.Path(verts, codes)

        x, y = zip(*path.vertices)
        line, = ax.plot(x, y, 'go-')
        ax.grid()
        #ax.set_xlim(-3,4)
        #ax.set_ylim(-3,4)
        ax.set_title(title)
        plt.show()


def main():
    if len(sys.argv) < 5:
        print "USAGE: python tsm_solver.py tour.tsp tour.opt.tour tries epocs"
        sys.exit()

    # data
    fname = sys.argv[1]
    tour_fname = sys.argv[2]
    data = read_tsp_file(fname)
    opt_tour_data = read_tour_file(tour_fname)

    tries = range(int(sys.argv[3]))
    epocs = int(sys.argv[4])

    settings = KohonenTSPSolverSettings()
    settings.debug_mode = False
    settings.add_weighting_dimension = True
    settings.normalize_city_data = (-0.5, .5)
    settings.random_init_weights_range = (-0.01, .01)
    settings.learning_algo = settings.favata_walker

    #settings.init_radius = 4


    reductions = []

    for x in tries:
        # setup and learn
        kts = KohonenTSPSolver(data[0], settings)


        tour = kts.get_tour()
        kts.plot(tour, 'Random Path')
        path_length = kts.tour_length(tour)
        print "path_length,%d" % (path_length,)

        kts.train(epocs)

        # get tour
        tour = kts.get_tour()
        #print tour
        kts.plot(tour, 'Optimized Path')
        optimized_path = kts.tour_length(tour)
        print "optimized_path_length,%d" % (optimized_path,)

        reductions.append(optimized_path / path_length)

    opt_tour = map(lambda x: x-1,opt_tour_data[0])

    #print opt_tour
    opt_tour_length = kts.tour_length(opt_tour)
    kts.plot(opt_tour, 'Best Path')
    print "optimal_distance,%d" % (opt_tour_length,)

    print "average_optimization: %s\n" % (sum(reductions)/len(reductions),)


if __name__ == '__main__':
    main()
