class Registry:
    _methods = {}
    _dataloaders = {}

    @classmethod
    def register_method(cls, name):
        def decorator(algorithm_class):
            cls._methods[name] = algorithm_class
            return algorithm_class
        return decorator

    @classmethod
    def register_dataloader(cls, name):
        def decorator(dataloader_class):
            cls._dataloaders[name] = dataloader_class
            return dataloader_class
        return decorator

    @classmethod
    def get_method(cls, name):
        if name not in cls._methods:
            raise ValueError(f"Method '{name}' not found in registry. Available: {list(cls._methods.keys())}")
        return cls._methods[name]

    @classmethod
    def get_dataloader(cls, name):
        if name not in cls._dataloaders:
            raise ValueError(f"Dataloader '{name}' not found in registry. Available: {list(cls._dataloaders.keys())}")
        return cls._dataloaders[name]
