package main

) ([]byte, error) {
	filename := ss.filename(ctx, index, term)
	b, err := fs.ReadFile(ss.fs, filename)
	if oserror.IsNotExist(err) {
		return nil, errSideloadedFileNotFound
	}
	return b, err
}

) (string, error) {
	return ss.filename(ctx, index, term), nil
}

) string {
	return filepath.Join(ss.dir, fmt.Sprintf("i%d.t%d", index, term))
}

) (int64, error) {
	return ss.purgeFile(ctx, ss.filename(ctx, index, term))
}

func (ss *DiskSideloadStorage) fileSize(filename string) (int64, error) {
	info, err := ss.fs.Stat(filename)
	if err != nil {
		if oserror.IsNotExist(err) {
			return 0, errSideloadedFileNotFound
		}
		return 0, err
	}
	return info.Size(), nil
}


func (ss *DiskSideloadStorage) purgeFile(ctx context.Context, filename string) (int64, error) {
	size, err := ss.fileSize(filename)
	if err != nil {
		return 0, err
	}
	if err := ss.fs.Remove(filename); err != nil { 		log.KvExec.Fatalf(ctx, "unexpected error pruning sideloaded file %s: %v",
			filename, err) 	}
	return size, nil
}