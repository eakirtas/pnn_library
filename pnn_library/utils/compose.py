class ComposePhotonicModules():
    def __init__(self, list_of_modules=None):
        self.list_of_modules = list_of_modules

    def __call__(self, x):
        if self.list_of_modules is not None:
            for module in self.list_of_modules:
                x = module(x)

        return x
