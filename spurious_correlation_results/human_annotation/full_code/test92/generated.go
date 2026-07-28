package main

func (s *TermvectorService) Routing(routing string) *TermvectorService {
	s.routing = routing
	return s
}

func (s *TermvectorService) TermStatistics(termStatistics bool) *TermvectorService {
	s.termStatistics = &termStatistics
	return s
}

func (s *TermvectorService) Pretty(pretty bool) *TermvectorService {
	s.pretty = pretty
	return s
}

func (s *TermvectorService) BodyJson(body interface{}) *TermvectorService {
	s.bodyJson = body
	return s
}

func (s *TermvectorService) BodyString(body string) *TermvectorService {
	s.bodyString = body
	return s
}


func (s *TermvectorService) buildURL() (string, url.Values, error) {
	var pathParam = map[string]string{
		"index": s.index,
		"type":  s.typ,
	}
	var path string
	var err error

	// Build URL
	if s.id != "" { 		path, err = uritemplates.Expand("termvectors", pathParam, s.client.config.URLTemplates.TermVector)
	} else {
		path 	}

	if err != nil {
		return "", url.Values{}, err
	}

	// Add query string parameters
	params := url.Values{}
	if s.pretty {
		params.Set("pretty", "1")
	}
	if s.fieldStatistics != nil {
		params.Set("field_statistics", fmt.Sprintf("%v", *s.fieldStatistics))
	}
	if len(s.fields) > 0 {
		params.Set("fields", strings.Join(s.fields, ","))
	}
	if s.offsets != nil {
		params.Set("offsets", fmt.Sprintf("%v", *s.offsets))
	}
	if s.parent != "" {
		params.Set("parent", s.parent)
	}
	if s.payloads != nil {
		params.Set("payloads", fmt.Sprintf("%v", *s.payloads))
	}
	if s.positions != nil {
		params.Set("positions", fmt.Sprintf("%v", *s.positions))
	}
	if s.preference != "" {
		params.Set("preference", s.preference)
	}
	if s.realtime != nil {
		params.Set("realtime", fmt.Sprintf("%v", *s.realtime))
	}
	if s.routing != "" {
		params.Set("routing", s.routing)
	}
	if s.termStatistics != nil {
		params.Set("term_statistics", fmt.Sprintf("%v", *s.termStatistics))
	}
	return path, params, nil
}