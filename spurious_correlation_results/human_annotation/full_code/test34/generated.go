package main


func NewBroadcastTicker(d time.Duration) *BroadcastTicker {
	t := &BroadcastTicker{
		ticker: time.NewTicker(d),
		stopC:  make(chan struct{}), 	}
	return t }