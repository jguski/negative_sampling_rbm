from pykeen.training.callbacks import TrainingCallback
from pykeen.training import SLCWATrainingLoop

from typing import Any

class NSQualAnalysisCallback(TrainingCallback):
    def __init__(self, negative_sampler, negative_sampler_kwargs, mapped_triples):
        self.negative_sampler = negative_sampler
        self.negative_sampler_kwargs = negative_sampler_kwargs
        self.mapped_triples = mapped_triples
        super().__init__()
   

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs: Any) -> None:

        # create instance of negative sampler only as soon as it is needed
        if not hasattr(self, 'negative_sampler_instance'):
            self.negative_sampler_instance = self.negative_sampler(**self.negative_sampler_kwargs, mapped_triples=self.mapped_triples)
        
        # call quality_analysis() method from negative sampler after each epoch
        self.negative_sampler_instance.quality_analysis(epoch)


class SLCWATrainingLoopModified(SLCWATrainingLoop):
    def _train(self, *args, **kwargs):
        
        if not hasattr(self, 'callback_instance'):
            # get mapped triples from kwargs for initialization of negative sampler
            mapped_triples = kwargs["triples_factory"].mapped_triples
            # copy negative_sampler_kwargs and manipulate them
            ns_kwargs = dict(self.negative_sampler_kwargs)
            ns_kwargs["model"] = self.model
            ns_kwargs["logging_level"] = "ERROR"
            self.callback_instance = NSQualAnalysisCallback(self.negative_sampler, ns_kwargs, mapped_triples)
            
            # add callback for quality analysis to dictionary with keyword arguments
            if ("callbacks" in kwargs.keys()) and (kwargs["callbacks"] == None):
                kwargs["callbacks"] = self.callback_instance
            elif ("callbacks" in kwargs.keys()) and (kwargs["callbacks"] != None):
                kwargs["callbacks"] = list(kwargs["callbacks"]) + [self.callback_instance]
        
        # call _train() from super class with updated kwargs
        return super()._train(*args, **kwargs)


