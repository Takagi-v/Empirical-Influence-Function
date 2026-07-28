package main


func NewBroadcastTicker(d time.Duration) *BroadcastTicker {
	t := &BroadcastTicker{
		ticker: time.NewTicker(d),
		stopC:  make(chan struct{}), 	}
	go t.run()
	return t }