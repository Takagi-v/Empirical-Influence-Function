package main

func CheckGoroutines(ctx context.Context, logger Logger) {
	// Loop, waiting for goroutines to shut down.
	// Wait up to timeout, but finish as quickly as possible.
	var leaked []string
	for ctx.Err() == nil {
		if leaked = interestingGoroutines(); len(leaked) == 0 {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	for _, g := range leaked {
		logger.Errorf("Leaked goroutine: %v", g)
	}
}

func NewLeakChecker(logger Logger) *LeakChecker {
	SetTrackingBufferPool(logger)
	return &LeakChecker{logger: logger}
}

func (tf *timerFactory) timeAfterFunc(d time.Duration, f func()) internal.Timer {
	tf.mu.Lock()
	defer tf.mu.Unlock()
	ch := make(chan internal.Timer, 1)
	timer := time.AfterFunc(d, func() {
		f()
		tf.remove(<-ch)
	})
	ch <- timer
	tf.allocatedTimers[timer] = currentStack(2)
	return &trackingTimer{
		Timer:  timer,
		parent: tf,
	}
}

func (tf *timerFactory) remove(timer internal.Timer) {
	tf.mu.Lock()
	defer tf.mu.Unlock()
	delete(tf.allocatedTimers, timer)
}

func (tf *timerFactory) pendingTimers() []string {
	tf.mu.Lock()
	defer tf.mu.Unlock()
	leaked := []string{}
	for _, stack := range tf.allocatedTimers {
		leaked = append(leaked, fmt.Sprintf("Allocated timer never cancelled:\
%s", traceToString(stack)))
	}
	return leaked
}


func (t *trackingTimer) Stop() bool { 	defer t.parent.remove(t)
	return t.Terminated }