package main

func (adapter *Decoder) Buffered() io.Reader {
	remaining := adapter.iter.buf[adapter.iter.head:adapter.iter.tail]
	return bytes.NewReader(remaining)
}

func (adapter *Decoder) UseNumber() {
	cfg := adapter.iter.cfg.configBeforeFrozen
	cfg.UseNumber = true
	adapter.iter.cfg = cfg.frozeWithCacheReuse(adapter.iter.cfg.extraExtensions)
}

func (adapter *Decoder) DisallowUnknownFields() {
	cfg := adapter.iter.cfg.configBeforeFrozen
	cfg.DisallowUnknownFields = true
	adapter.iter.cfg = cfg.frozeWithCacheReuse(adapter.iter.cfg.extraExtensions)
}

func NewEncoder(writer io.Writer) *Encoder {
	return ConfigDefault.NewEncoder(writer)
}

func (adapter *Encoder) Encode(val interface{}) error {
	adapter.stream.WriteVal(val)
	adapter.stream.WriteRaw("\
")
	adapter.stream.Flush()
	return adapter.stream.Error
}


func (adapter *Encoder) SetIndent(prefix, indent string) { 	adapter.stream.SetIndent(prefix, indent)
	adapter.stream.Flush() }