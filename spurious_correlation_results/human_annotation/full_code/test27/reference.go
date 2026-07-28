package main

func (sched *Scheduler) addAssignedPodToCache(pod *v1.Pod) {
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventAssignedPodAdd.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger

	logger.V(3).Info("Add event for scheduled pod", "pod", klog.KObj(pod))
	if err := sched.Cache.AddPod(logger, pod); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Scheduler cache AddPod failed", "pod", klog.KObj(pod))
	}

	// SchedulingQueue.AssignedPodAdded has a problem:
	// It internally pre-filters Pods to move to activeQ,
	// while taking only in-tree plugins into consideration.
	// Consequently, if custom plugins that subscribes Pod/Add events reject Pods,
	// those Pods will never be requeued to activeQ by an assigned Pod related events,
	// and they may be stuck in unschedulableQ.
	//
	// Here we use MoveAllToActiveOrBackoffQueue only when QueueingHint is enabled.
	// (We cannot switch to MoveAllToActiveOrBackoffQueue right away because of throughput concern.)
	if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints) {
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, framework.EventAssignedPodAdd, nil, pod, nil)
	} else {
		sched.SchedulingQueue.AssignedPodAdded(logger, pod)
	}
}

func (sched *Scheduler) updateAssignedPodInCache(oldPod, newPod *v1.Pod) {
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventAssignedPodUpdate.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger

	if sched.APIDispatcher != nil {
		// If the API dispatcher is available, sync the new pod with the details.
		// However, at the moment the updated newPod is discarded and this logic will be handled in the future releases.
		_ = sched.syncPodWithDispatcher(newPod)
	}

	logger.V(4).Info("Update event for scheduled pod", "pod", klog.KObj(oldPod))
	if err := sched.Cache.UpdatePod(logger, oldPod, newPod); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Scheduler cache UpdatePod failed", "pod", klog.KObj(oldPod))
	}

	events := framework.PodSchedulingPropertiesChange(newPod, oldPod)

	// Save the time it takes to update the pod in the cache.
	updatingDuration := metrics.SinceInSeconds(start)

	for _, evt := range events {
		startMoving := time.Now()
		// SchedulingQueue.AssignedPodUpdated has a problem:
		// It internally pre-filters Pods to move to activeQ,
		// while taking only in-tree plugins into consideration.
		// Consequently, if custom plugins that subscribes Pod/Update events reject Pods,
		// those Pods will never be requeued to activeQ by an assigned Pod related events,
		// and they may be stuck in unschedulableQ.
		//
		// Here we use MoveAllToActiveOrBackoffQueue only when QueueingHint is enabled.
		// (We cannot switch to MoveAllToActiveOrBackoffQueue right away because of throughput concern.)
		if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints) {
			sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, oldPod, newPod, nil)
		} else {
			sched.SchedulingQueue.AssignedPodUpdated(logger, oldPod, newPod, evt)
		}
		movingDuration := metrics.SinceInSeconds(startMoving)
		metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(updatingDuration + movingDuration)
	}
}

func (sched *Scheduler) deleteAssignedPodFromCache(pod *v1.Pod) {
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventAssignedPodDelete.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger

	logger.V(3).Info("Delete event for scheduled pod", "pod", klog.KObj(pod))
	if err := sched.Cache.RemovePod(logger, pod); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Scheduler cache RemovePod failed", "pod", klog.KObj(pod))
	}

	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, framework.EventAssignedPodDelete, pod, nil, nil)
}

func assignedPod(pod *v1.Pod) bool {
	return len(pod.Spec.NodeName) != 0
}

func responsibleForPod(pod *v1.Pod, profiles profile.Map) bool {
	return profiles.HandlesSchedulerName(pod.Spec.SchedulerName)
}


func (sched *Scheduler) WaitForHandlersSync(ctx context.Context) error {
	return wait.PollUntilContextCancel(ctx, syncedPollPeriod, true, func(ctx context.Context) (done bool, err error) {
		for _, handler := range sched.registeredHandlers { 			if !handler.HasSynced() {
				return false, nil
			} 		}
		return true, nil
	})
}