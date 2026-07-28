package main

func (flags *WaitFlags) ToOptions(args []string) (*WaitOptions, error) {
	printer, err := flags.PrintFlags.ToPrinter()
	if err != nil {
		return nil, err
	}
	builder := flags.ResourceBuilderFlags.ToBuilder(flags.RESTClientGetter, args)
	clientConfig, err := flags.RESTClientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	dynamicClient, err := dynamic.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	conditionFn, err := conditionFuncFor(flags.ForCondition, flags.ErrOut)
	if err != nil {
		return nil, err
	}

	effectiveTimeout := flags.Timeout
	if effectiveTimeout < 0 {
		effectiveTimeout = 168 * time.Hour
	}

	o := &WaitOptions{
		ResourceFinder: builder,
		DynamicClient:  dynamicClient,
		Timeout:        effectiveTimeout,
		ForCondition:   flags.ForCondition,

		Printer:     printer,
		ConditionFn: conditionFn,
		IOStreams:   flags.IOStreams,
	}

	return o, nil
}

func conditionFuncFor(condition string, errOut io.Writer) (ConditionFunc, error) {
	lowercaseCond := strings.ToLower(condition)
	switch {
	case lowercaseCond == "delete":
		return IsDeleted, nil

	case lowercaseCond == "create":
		return IsCreated, nil

	case strings.HasPrefix(condition, "condition="):
		conditionName := strings.TrimPrefix(condition, "condition=")
		conditionValue := "true"
		if equalsIndex := strings.Index(conditionName, "="); equalsIndex != -1 {
			conditionValue = conditionName[equalsIndex+1:]
			conditionName = conditionName[0:equalsIndex]
		}

		return ConditionalWait{
			conditionName:   conditionName,
			conditionStatus: conditionValue,
			errOut:          errOut,
		}.IsConditionMet, nil

	case strings.HasPrefix(condition, "jsonpath="):
		jsonPathInput := strings.TrimPrefix(condition, "jsonpath=")
		jsonPathExp, jsonPathValue, err := processJSONPathInput(jsonPathInput)
		if err != nil {
			return nil, err
		}
		j, err := newJSONPathParser(jsonPathExp)
		if err != nil {
			return nil, err
		}
		return JSONPathWait{
			matchAnyValue:  jsonPathValue == "",
			jsonPathValue:  jsonPathValue,
			jsonPathParser: j,
			errOut:         errOut,
		}.IsJSONPathConditionMet, nil
	}

	return nil, fmt.Errorf("unrecognized condition: %q", condition)
}

func newJSONPathParser(jsonPathExpression string) (*jsonpath.JSONPath, error) {
	j := jsonpath.New("wait").AllowMissingKeys(true)
	if jsonPathExpression == "" {
		return nil, errors.New("jsonpath expression cannot be empty")
	}
	if err := j.Parse(jsonPathExpression); err != nil {
		return nil, err
	}
	return j, nil
}

func processJSONPathInput(input string) (string, string, error) {
	jsonPathInput := splitJSONPathInput(input)
	if numOfArgs := len(jsonPathInput); numOfArgs < 1 || numOfArgs > 2 {
		return "", "", fmt.Errorf("jsonpath wait format must be --for=jsonpath='{.status.readyReplicas}'=3 or --for=jsonpath='{.status.readyReplicas}'")
	}
	relaxedJSONPathExp, err := cmdget.RelaxedJSONPathExpression(jsonPathInput[0])
	if err != nil {
		return "", "", err
	}
	if len(jsonPathInput) == 1 {
		return relaxedJSONPathExp, "", nil
	}
	jsonPathValue := strings.Trim(jsonPathInput[1], `'"`)
	if jsonPathValue == "" {
		return "", "", errors.New("jsonpath wait has to have a value after equal sign, like --for=jsonpath='{.status.readyReplicas}'=3")
	}
	return relaxedJSONPathExp, jsonPathValue, nil
}

func splitJSONPathInput(input string) []string {
	var output []string
	var element strings.Builder
	for i := 0; i < len(input); i++ {
		if input[i] == '=' {
			if i < len(input)-1 && input[i+1] == '=' {
				element.WriteString("==")
				i++
				continue
			}
			output = append(output, element.String())
			element.Reset()
			continue
		}
		element.WriteByte(input[i])
	}
	return append(output, element.String())
}


func (o *WaitOptions) RunWait() error {
	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), o.Timeout)
	defer cancel()

	if strings.ToLower(o.ForCondition) == "create" {
		// TODO(soltysh): this is not ideal solution, because we're polling every .5s,
		// and we have to use ResourceFinder, which contains the resource name.
		// In the long run, we should expose resource information from ResourceFinder,
		// or functions from ResourceBuilder for parsing those. Lastly, this poll
		// should be replaced with a ListWatch cache.
		if err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, o.Timeout, true, func(context.Context) (done bool, err error) {
			foundResource := false
			visitErr := o.ResourceFinder.Do().Visit(func(info *resource.Info, err error) error {
				foundResource = true
				return nil
			})
			if apierrors.IsNotFound(visitErr) {
				return false, nil
			}
			if visitErr != nil {
				return false, visitErr
			}
			return foundResource, nil
		}); err != nil {
			if errors.Is(err, context.DeadlineExceeded) {
				return fmt.Errorf("%s", wait.ErrorInterrupted(nil).Error()) // nolint:staticcheck // SA1019
			}
			return err
		}
	}

	visitCount := 0
	visitFunc := func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		visitCount++
		finalObject, success, err := o.ConditionFn(ctx, info, o)
		if success { 			o.Printer.PrintObj(finalObject, o.Out)
			return nil 		}
		if err == nil {
			return fmt.Errorf("%v unsatisfied for unknown reason", finalObject)
		}
		return err
	}
	visitor := o.ResourceFinder.Do()
	isForDelete := strings.ToLower(o.ForCondition) == "delete"
	if visitor, ok := visitor.(*resource.Result); ok && isForDelete {
		visitor.IgnoreErrors(apierrors.IsNotFound)
	}

	err := visitor.Visit(visitFunc)
	if err != nil {
		return err
	}
	if visitCount == 0 && !isForDelete {
		return errNoMatchingResources
	}
	return err
}