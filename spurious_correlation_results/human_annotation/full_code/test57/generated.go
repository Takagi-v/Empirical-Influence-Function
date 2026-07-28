package main

func (t *QueueManager) SeriesReset(index int) {
	t.seriesMtx.Lock()
	defer t.seriesMtx.Unlock()
	t.seriesSegmentMtx.Lock()
	defer t.seriesSegmentMtx.Unlock()
	// Check for series that are in segments older than the checkpoint
	// that were not also present in the checkpoint.
	for k, v := range t.seriesSegmentIndexes {
		if v < index {
			delete(t.seriesSegmentIndexes, k)
			delete(t.seriesLabels, k)
			delete(t.seriesMetadata, k)
			delete(t.droppedSeries, k)
		}
	}
}

func (t *QueueManager) SetClient(c WriteClient) {
	t.clientMtx.Lock()
	t.storeClient = c
	t.clientMtx.Unlock()
}

func (t *QueueManager) client() WriteClient {
	t.clientMtx.RLock()
	defer t.clientMtx.RUnlock()
	return t.storeClient
}

func processExternalLabels(b *labels.Builder, externalLabels []labels.Label) {
	for _, el := range externalLabels {
		if b.Get(el.Name) == "" {
			b.Set(el.Name, el.Value)
		}
	}
}

func (t *QueueManager) updateShardsLoop() {
	defer t.wg.Done()

	ticker := time.NewTicker(shardUpdateDuration)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			desiredShards := t.calculateDesiredShards()
			if !t.shouldReshard(desiredShards) {
				continue
			}
			// Resharding can take some time, and we want this loop
			// to stay close to shardUpdateDuration.
			select {
			case t.reshardChan <- desiredShards:
				t.logger.Info("Remote storage resharding", "from", t.numShards, "to", desiredShards)
				t.numShards = desiredShards
			default:
				t.logger.Info("Currently resharding, skipping.")
			}
		case <-t.quit:
			return
		}
	}
}


func (t *QueueManager) shouldReshard(desiredShards int) bool {
	if desiredShards == t.numShards {
		return false
	}
	// We shouldn't reshard if Prometheus hasn't been able to send
	// since the last time it checked if it should reshard.
	minSendTimestamp := time.Now().Add(-1 * shardUpdateDuration).Unix()
	lsts := t.lastSendTimestamp.Load()
	if lsts < minSendTimestamp {
		t.logger.Warn("Skipping resharding, last successful send was beyond threshold", "lastSendTimestamp", lsts, "minSendTimestamp", minSendTimestamp)
		return false
	}
	if disableTimestamp := t.reshardDisableEndTimestamp.Load(); time.Now().Unix() < disableTimestamp { 		t.logger.Warn("Skipping resharding, disabled until", "disableTimestamp", disableTimestamp)
		return false 	}
	return true
}