package main

func (g *Gauge) Add(delta float64) {
	for {
		var (
			old  = atomic.LoadUint64(&g.bits)
			newf = math.Float64frombits(old) + delta
			new  = math.Float64bits(newf)
		)
		if atomic.CompareAndSwapUint64(&g.bits, old, new) {
			break
		}
	}
}

func (g *Gauge) Value() float64 {
	return math.Float64frombits(atomic.LoadUint64(&g.bits))
}

func (g *Gauge) LabelValues() []string {
	return g.lvs
}

func NewHistogram(name string, buckets int) *Histogram {
	return &Histogram{
		Name: name,
		h:    &safeHistogram{Histogram: gohistogram.NewHistogram(buckets)},
	}
}

func (h *Histogram) With(labelValues ...string) metrics.Histogram {
	return &Histogram{
		Name: h.Name,
		lvs:  h.lvs.With(labelValues...),
		h:    h.h,
	}
}


func (h *Histogram) Observe(value float64) { 	h.h.With(h.lvs).Observe(value)
	atomic.AddInt64(&h.h.count, 1) }