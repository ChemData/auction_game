import importlib


class ImportTest:

    def __init__(self):
        self.functions = {'f1': 'bod.square', 'f2': 'mod.square'}
        self.import_functions()

    def import_functions(self):
        separated = {}
        for k in self.functions.keys():
            separated[k] = self.functions[k].split('.')

        self.add_mods({x[0] for x in separated.values()})

        for k in self.functions.keys():
            self.functions[k] = eval(self.functions[k])

    @staticmethod
    def add_mods(modules):
        for module in modules:
            module_object = importlib.import_module(module)
            globals()[module] = module_object

    @staticmethod
    def deep_values(dictionary):
        """Return all values in a dictionary (and values of any subdictionaries)."""
        output = []
        for v in dictionary.values():
            if isinstance(v, dict):
                output += ImportTest.deep_values(v)
            else:
                output += [v]
        return output
