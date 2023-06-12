
import numpy
import os
import glob

class DatamapLoader:
    """
    Creates a `DatamapLoader` that allows to iterate over a number of number 
    of datamaps that are stored in the `examples` folder in `npy` files.
    """
    def __init__(self, structure=None, paths=None):
        """
        Instantiates a `DatamapLoader`

        :param structure: A `str` of the required structure
        :param paths: (optional) A `list` of the desired paths
        """
        assert (not isinstance(structure, type(None))) or (not isinstance(paths, type(None))), \
            print("`structures` or `paths` should be defined")

        if isinstance(paths, type(None)):
            self.structure = structure
            self.basepath = os.path.dirname(__file__)
            available_examples = [
                os.path.basename(item)
                for item in os.listdir(os.path.join(self.basepath, "examples"))
                if os.path.isdir(os.path.join(self.basepath, "examples", item))
            ]
            assert self.structure in available_examples, \
                print("Structure `{}` is not available. Select from `{}`".format(self.structure), available_examples)
            self.datamaps = glob.glob(os.path.join(self.basepath, "examples", self.structure, "*.npy"))
        else:
            self.datamaps = paths

    def __getitem__(self, item):
        """
        Implements the `getitem` method of `DatamapLoader`

        :param item: A {`tuple`, `slice`, `int`} of the required item(s)

        :return : A `numpy.ndarray` of the datamap
        """
        if isinstance(item, slice):
            return DatamapLoader(paths=self.datamaps[item])
        if isinstance(item, tuple):
            return DatamapLoader(paths=[self.datamaps[i] for i in item])
        item = numpy.load(self.datamaps[item])
        return item

    def __len__(self):
        """
        Implements the `len` method of `DatamapLoader`

        :returns : An `int` of the number of datamaps
        """
        return len(self.datamaps)

if __name__ == "__main__":

    loader = DatamapLoader("factin")
    for datamap in loader:
        print(datamap.shape)