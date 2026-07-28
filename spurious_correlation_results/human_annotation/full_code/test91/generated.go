package main


func (m *Preconditions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.UID != nil { 		l = m.UID.Size()
		n += 1 + l + sovGenerated(uint64(l)) 	}
	return n
}