""" A baseline for the ESNS without exploration mechanism classes, 
maintaining an equivalent index with random entities"""

import numpy as np
from .esns_no_exploration import ESNSNoExploration



class BaselineNoExploration(ESNSNoExploration):

    def _create_eii(self, column: int) -> None: 
        return np.random.choice(self.num_entities, (self.num_entities, self.max_index_column_size))



