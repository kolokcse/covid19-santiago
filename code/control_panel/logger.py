from torch.utils.tensorboard import SummaryWriter

class TBLogger(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='',
                 global_step=0, batch_size=1, world_size=1, global_step_divider=1):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.global_step = global_step
        self.warned_missing_grad = False
        self.batch_size = batch_size
        self.world_size = world_size
        self.dist_bs = self.batch_size * self.world_size
        self.global_step_divider = global_step_divider

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=True, double_precision=False):
        if global_step is None:
            global_step = round(self.global_step / self.global_step_divider)
        super().add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)
