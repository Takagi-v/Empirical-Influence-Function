package main

func WrapErrFileDoesNotExist(err error, msg string) error {
	//nolint:errwrap
	return errors.Wrapf(ErrFileDoesNotExist, "%s: %s", err.Error(), msg)
}

func RedactedParams(strs ...string) map[string]struct{} {
	if len(strs) == 0 {
		return nil
	}
	m := make(map[string]struct{}, len(strs))
	for i := range strs {
		m[strs[i]] = struct{}{}
	}
	return m
}

func (e *EarlyBootExternalStorageContext) ExternalStorageOptions() ExternalStorageOptions {
	var options ExternalStorageOptions
	for _, option := range e.Options {
		option(&options)
	}
	return options
}

func NewEarlyBootExternalStorageAccessor((st *cluster.Settings, conf base.ExternalIODirConfig, lookup *cidr.Lookup)) *EarlyBootExternalStorageAccessor
) *EarlyBootExternalStorageAccessor {
	return &EarlyBootExternalStorageAccessor{
		conf:     conf,
		settings: st,
		limiters: MakeLimiters(&st.SV), 		metrics:  metric.NewStruct(st, "cloud_externals"),
	} }