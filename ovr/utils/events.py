from detectron2.utils.events import CommonMetricPrinter, get_event_storage
import datetime
import time
import torch

class OvrMetricPrinter(CommonMetricPrinter):
    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        try:
            iter_time = storage.history("time").global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None


        losses_text ="  ".join(
                        [
                            "{}: {:.4g}".format(k, v.median(self._window_size))
                            for k, v in storage.histories().items()
                            if ("loss" in k or "Loss" in k) and ("val" not in k)
                        ]
                    )
        acc_text =  "  ".join(
                        [
                            "{}: {:.4g}".format(k, v.median(self._window_size))
                            for k, v in storage.histories().items()
                            if ("acc" in k or "Acc" in k) and ("val" not in k)
                        ]
                    )

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}iter: {iter}  {losses} {accuracies} {time}{data_time}lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses=losses_text,
                accuracies=acc_text,
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )
