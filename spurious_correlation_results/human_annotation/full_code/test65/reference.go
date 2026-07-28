package main

func (any *objectLazyAny) ToVal(obj interface{}) {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	iter.ReadVal(obj)
}

func (any *objectLazyAny) Get(path ...interface{}) Any {
	if len(path) == 0 {
		return any
	}
	switch firstPath := path[0].(type) {
	case string:
		iter := any.cfg.BorrowIterator(any.buf)
		defer any.cfg.ReturnIterator(iter)
		valueBytes := locateObjectField(iter, firstPath)
		if valueBytes == nil {
			return newInvalidAny(path)
		}
		iter.ResetBytes(valueBytes)
		return locatePath(iter, path[1:])
	case int32:
		if '*' == firstPath {
			mappedAll := map[string]Any{}
			iter := any.cfg.BorrowIterator(any.buf)
			defer any.cfg.ReturnIterator(iter)
			iter.ReadMapCB(func(iter *Iterator, field string) bool {
				mapped := locatePath(iter, path[1:])
				if mapped.ValueType() != InvalidValue {
					mappedAll[field] = mapped
				}
				return true
			})
			return wrapMap(mappedAll)
		}
		return newInvalidAny(path)
	default:
		return newInvalidAny(path)
	}
}

func (any *objectLazyAny) Keys() []string {
	keys := []string{}
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	iter.ReadMapCB(func(iter *Iterator, field string) bool {
		iter.Skip()
		keys = append(keys, field)
		return true
	})
	return keys
}

func (any *objectLazyAny) Size() int {
	size := 0
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	iter.ReadObjectCB(func(iter *Iterator, field string) bool {
		iter.Skip()
		size++
		return true
	})
	return size
}

func (any *objectLazyAny) WriteTo(stream *Stream) {
	stream.Write(any.buf)
}


func (any *objectLazyAny) GetInterface() interface{} { 	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	return iter.Read() }