import numpy as np
import matplotlib.pyplot as plt
from triangular3 import Triangular3Scheduler


MIN_LR = 1e-9
MAX_LR = 3e-3
STEPS_PER_EPOCH = 1000
CYCLE_LENGTH = 5
UPWARD_RATIO = 0.1
TOTAL_EPOCHS = 20


class Model():
    def __init__(self):
        pass
    def get_weights(self):
        pass

def main():
    schedule = Triangular3Scheduler(min_lr=MIN_LR, max_lr=MAX_LR, steps_per_epoch=STEPS_PER_EPOCH, lr_decay=1.0, cycle_length=CYCLE_LENGTH, upward_ratio=UPWARD_RATIO, mult_factor=1.0)
    schedule.model = Model()
    lrs = []
    #schedule.on_train_begin()
    for epoch in range(TOTAL_EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            lrs.append(schedule.clr())
            #schedule.on_batch_end(0)
            schedule.batch_since_restart += 1
        schedule.on_epoch_end(epoch)
    #schedule.on_train_end()

    iters = range(len(lrs))
    lrs = np.array(lrs)
    plt.plot(iters, lrs)
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.show()


if __name__ == '__main__':
    main()
