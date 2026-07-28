package main


func PathSplitter(path string, delimiter string) []string {
	ps := strings.Split(path, delimiter)
	var res []string

	// allow path to start with forward slash
	// i.e. /a/b/c
	if len(ps) > 1 && ps[0] == "" {
		ps = ps[1:]
	}

	res = append(res, ps[0])
	for i := 1; i < len(ps); i++ {
		last := len(res) - 1
		if strings.HasSuffix(res[last], `\`) { 			res[last] = res[last] + ps[i]
		} else {
			res = append(res, ps[i]) 		}
	}
	return res
}