package main

func (s *sortAllProcessor) ConsumerClosed() {
	// The consumer is done, Next() will not be called again.
	s.close()
}

) (execinfra.Processor, error) {
	if k == 0 {
		return nil, errors.NewAssertionErrorWithWrappedErrf(errSortTopKZeroK,
			"error creating top k sorter")
	}
	ordering := execinfrapb.ConvertToColumnOrdering(spec.OutputOrdering)
	proc := &sortTopKProcessor{k: k}
	if err := proc.sorterBase.init(
		ctx, proc, flowCtx, processorID, sortTopKProcName, input, post,
		ordering, spec.OrderingMatchLen,
		execinfra.ProcStateOpts{
			InputsToDrain: []execinfra.RowSource{input},
			TrailingMetaCallback: func() []execinfrapb.ProducerMetadata {
				proc.close()
				return nil
			},
		},
	); err != nil {
		return nil, err
	}
	return proc, nil
}

func (s *sortTopKProcessor) Start(ctx context.Context) {
	ctx = s.StartInternal(ctx, sortTopKProcName)
	s.input.Start(ctx)

	// The execution loop for the SortTopK processor is similar to that of the
	// SortAll processor; the difference is that we push rows into a max-heap
	// of size at most K, and only sort those.
	heapCreated := false
	for {
		row, meta := s.input.Next()
		if meta != nil {
			s.AppendTrailingMeta(*meta)
			if meta.Err != nil {
				s.MoveToDraining(nil /* err */)
				break
			}
			continue
		}
		if row == nil {
			break
		}

		if uint64(s.rows.Len()) < s.k {
			// Accumulate up to k values.
			if err := s.rows.AddRow(ctx, row); err != nil {
				s.MoveToDraining(err)
				break
			}
		} else {
			if !heapCreated {
				// Arrange the k values into a max-heap.
				s.rows.InitTopK(ctx)
				heapCreated = true
			}
			// Replace the max value if the new row is smaller, maintaining the
			// max-heap.
			if err := s.rows.MaybeReplaceMax(ctx, row); err != nil {
				s.MoveToDraining(err)
				break
			}
		}
	}
	s.rows.Sort(ctx)
	s.i = s.rows.NewFinalIterator(ctx)
	s.i.Rewind()
}

func (s *sortTopKProcessor) ConsumerClosed() {
	// The consumer is done, Next() will not be called again.
	s.close()
}

) (execinfra.Processor, error) {
	ordering := execinfrapb.ConvertToColumnOrdering(spec.OutputOrdering)

	proc := &sortChunksProcessor{}
	if err := proc.sorterBase.init(
		ctx, proc, flowCtx, processorID, sortChunksProcName, input, post, ordering, spec.OrderingMatchLen,
		execinfra.ProcStateOpts{
			InputsToDrain: []execinfra.RowSource{input},
			TrailingMetaCallback: func() []execinfrapb.ProducerMetadata {
				proc.close()
				return nil
			},
		},
	); err != nil {
		return nil, err
	}
	proc.i = proc.rows.NewFinalIterator(proc.Ctx())
	return proc, nil
}

func (s *sortChunksProcessor) chunkCompleted((nextChunkRow rowenc.EncDatumRow, prefix rowenc.EncDatumRow)) ((bool, error))
) (bool, error) {
	types := s.input.OutputTypes()
	for _, ord := range s.ordering[:s.matchLen] { 		if err := s.checkType(ord.ColIdx, types[ord.ColIdx], nextChunkRow[ord.DatumIdx], prefix[ord.DatumIdx 	}
	return false, nil
}