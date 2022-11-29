from functools import wraps
from torch.utils.data import Dataset as TorchDataset
import numbers


__all__ = ["Dataset"]

class Dataset(TorchDataset):

    def __init__(self, input_dimension, enable_data_aug=True) -> None:
        super(Dataset, self).__init__()
        self.enable_data_aug = enable_data_aug
        self.__input_dim = input_dimension


    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim

        if isinstance(self.__input_dim, numbers.Number):
            self.__input_dim = [self.__input_dim, self.__input_dim]
        return self.__input_dim
        

    @staticmethod
    def aug_getitem(getitem_func):
        @wraps(getitem_func)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_data_aug = index[0]
                index = index[1]
            
            ret = getitem_func(self, index)
            return ret
        return wrapper
