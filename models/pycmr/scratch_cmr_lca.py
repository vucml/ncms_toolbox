
# leaky competitive accumulators
# deep part could be a layer function

# task part needs a timer
# task.recall_duration
# task.clock_time
setattr(task, 'recall_duration', 60000)
setattr(task, 'clock_time', 0)

# generate the noise matrix, more efficient than calling it again and again
# n racers by max cycles
n_racers = len(strength)
max_cycles = ceil((task.recall_duration - task.clock_time) / param.lca_dt)
noise = rn.randn((n_racers, max_cycles))

x = np.zeros(n_racers)
crossed = False
i = 1
while i < ncycles and not crossed:
    
    x += (strength - param.K*x - (sum(x*param.L)-(x*param.L))) * dt_tau + param.eta * noise[:,i] * sq_dt_tau
    # bounded at zero
    x[x<0] = 0

    # don't let previously recalled items cross threshold again
    # drop them down to below threshold
    
            
    # determine whether any items have crossed threshold
    if np.any(x >= thresholds and retrievable)
        crossed = True
        winners = x >= thresholds

    i += 1

# determine amount of elapsed time
elapsed_time = i * param.dt

# determine which unit(s) crossed threshold
# random tiebreak if more than one


