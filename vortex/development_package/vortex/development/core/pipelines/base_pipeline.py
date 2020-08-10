class BasePipeline():
    """Base class for Vortex pipelines
    """
    def __init__(self):
        pass

    def run(self):
        """Default way to execute pipelines

        Raises:
            NotImplementedError: this function must be implemented in sub-class
        """
        raise NotImplementedError('Vortex pipeline class must implement "__call__" method!')