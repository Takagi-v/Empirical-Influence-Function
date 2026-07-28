package main

func (c *replicaSplitConfig) SampleResetDuration() time.Duration {
	return SplitSampleResetDuration.Get(&c.st.SV)
}

func (r *Replica) SplitByLoadEnabled() bool {
	return SplitByLoadEnabled.Get(&r.store.cfg.Settings.SV) &&
		!r.store.TestingKnobs().DisableLoadBasedSplitting
}

) (responseBoundarySpan roachpb.Span) {
	addSpanToBoundary := func(span roachpb.Span) {
		if !responseBoundarySpan.Valid() {
			responseBoundarySpan = span
		} else {
			responseBoundarySpan = responseBoundarySpan.Combine(span)
		}
	}
	for i, respUnion := range br.Responses {
		reqHeader := ba.Requests[i].GetInner().Header()
		resp := respUnion.GetInner()
		resumeSpan := resp.Header().ResumeSpan
		if resumeSpan == nil {
			// Fully evaluated.
			addSpanToBoundary(reqHeader.Span())
			continue
		}

		switch resp.(type) {
		case *kvpb.GetResponse:
			// The request did not evaluate. Ignore it.
			continue
		case *kvpb.ScanResponse:
			// Not reverse (->)
			// Request:    [key...............endKey)
			// ResumeSpan:          [key......endKey)
			// True span:  [key......key)
			//
			// Assumptions (not checked to minimize overhead):
			// reqHeader.EndKey == resumeSpan.EndKey
			// reqHeader.Key <= resumeSpan.Key.
			if reqHeader.Key.Equal(resumeSpan.Key) {
				// The request did not evaluate. Ignore it.
				continue
			}
			addSpanToBoundary(roachpb.Span{
				Key:    reqHeader.Key,
				EndKey: resumeSpan.Key,
			})
		case *kvpb.ReverseScanResponse:
			// Reverse (<-)
			// Request:    [key...............endKey)
			// ResumeSpan: [key......endKey)
			// True span:           [endKey...endKey)
			//
			// Assumptions (not checked to minimize overhead):
			// reqHeader.Key == resumeSpan.Key
			// resumeSpan.EndKey <= reqHeader.EndKey.
			if reqHeader.EndKey.Equal(resumeSpan.EndKey) {
				// The request did not evaluate. Ignore it.
				continue
			}
			addSpanToBoundary(roachpb.Span{
				Key:    resumeSpan.EndKey,
				EndKey: reqHeader.EndKey,
			})
		default:
			// Consider it fully evaluated, which is safe.
			addSpanToBoundary(reqHeader.Span())
		}
	}
	return
}

) {
	if !r.SplitByLoadEnabled() {
		return
	}

	// There is nothing to do when either the batch request or batch response
	// are nil as we cannot record the load to a keyspan.
	if ba == nil || br == nil {
		return
	}

	if len(ba.Requests) != len(br.Responses) {
		log.KvDistribution.Errorf(ctx,
			"Requests and responses should be equal lengths: # of requests = %d, # of responses = %d",
			len(ba.Requests), len(br.Responses))
	}

	loadFn := func(obj split.SplitObjective) int {
		switch obj {
		case split.SplitCPU:
			return cpu
		default:
			return len(ba.Requests)
		}
	}

	spanFn := func() roachpb.Span {
		return getResponseBoundarySpan(ba, br)
	}

	shouldInitSplit := r.loadBasedSplitter.Record(ctx, r.Clock().PhysicalTime(), loadFn, spanFn)
	if shouldInitSplit {
		r.store.splitQueue.MaybeAddAsync(ctx, r, r.store.Clock().NowAsClockTimestamp())
	}
}

func (r *Replica) loadSplitKey(ctx context.Context, now time.Time) roachpb.Key {
	var splitKey roachpb.Key
	if overrideFn := r.store.cfg.TestingKnobs.LoadBasedSplittingOverrideKey; overrideFn != nil {
		var useSplitKey bool
		if splitKey, useSplitKey = overrideFn(r.GetRangeID()); useSplitKey {
			return splitKey
		}
	} else {
		splitKey = r.loadBasedSplitter.MaybeSplitKey(ctx, now)
	}

	if splitKey == nil {
		return nil
	}

	// If the splitKey belongs to a Table range, try and shorten the key to just
	// the row prefix. This allows us to check that splitKey doesn't map to the
	// first key of the range here. If the split key contains column families, it
	// is possible that the full key is strictly after every existing key for
	// that row. e.g. for a table row where the table ID is 100, index ID is 1,
	// primary key is a, and the column family ID is 3 (length=1):
	//
	//   splitKey = /Table/100/1/"a"/3/1
	//   existing = [..., /Table/100/1/"a"/2/1]
	//
	// We would not split at /Table/100/1/"a" as there's no key >= the splitKey
	// in the range.
	//
	// NB: We handle unsafe split keys in replica.adminSplitWithDescriptor, so it
	// isn't an issue if we return an unsafe key here. See the case where
	// findFirstSafeKey is true.
	if keyRowPrefix, err := keys.EnsureSafeSplitKey(splitKey); err == nil {
		splitKey = keyRowPrefix
	}

	// We swallow the error here and instead log an event. It is currently
	// expected that the load based splitter may return the start key of the
	// range.
	if err := splitKeyPreCheck(r.Desc().RSpan(), splitKey); err != nil {
		log.KvDistribution.VEventf(ctx, 1, "suggested load split key not usable: %s", err)
		return nil
	}

	return splitKey
}


func splitKeyPreCheck(rspan roachpb.RSpan, splitKey roachpb.Key) error {
	splitRKey, err := keys.Addr(splitKey)
	if err != nil {
		return err
	}

	// If the split key is equal to the start key of the range, it is treated as
	// a no-op in adminSplitWithDescriptor, however it is treated as an error
	// here because we shouldn't be suggesting split keys that are identical to
	// the start key of the range.
	if splitRKey.Equal(rspan.Key) { 		return errors.Errorf(
			"split key is equal to range start key (split_key=%s)",
			splitRKey) 	}

	return nil
}