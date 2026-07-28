package main

func (c matchCandidate) isBetterMatchThan(other matchCandidate) bool {
	if c.call == nil {
		return false
	}
	if other.call == nil {
		return true
	}

	if c.diffCount > other.diffCount {
		return false
	}
	if c.diffCount < other.diffCount {
		return true
	}

	if c.call.Repeatability > 0 && other.call.Repeatability <= 0 {
		return true
	}
	return false
}

func (m *Mock) findClosestCall(method string, arguments ...interface{}) (*Call, string) {
	var bestMatch matchCandidate

	for _, call := range m.expectedCalls() {
		if call.Method == method {

			errInfo, tempDiffCount := call.Arguments.Diff(arguments)
			tempCandidate := matchCandidate{
				call:      call,
				mismatch:  errInfo,
				diffCount: tempDiffCount,
			}
			if tempCandidate.isBetterMatchThan(bestMatch) {
				bestMatch = tempCandidate
			}
		}
	}

	return bestMatch.call, bestMatch.mismatch
}

func callString(method string, arguments Arguments, includeArgumentValues bool) string {
	var argValsString string
	if includeArgumentValues {
		var argVals []string
		for argIndex, arg := range arguments {
			if _, ok := arg.(*FunctionalOptionsArgument); ok {
				argVals = append(argVals, fmt.Sprintf("%d: %s", argIndex, arg))
				continue
			}
			argVals = append(argVals, fmt.Sprintf("%d: %#v", argIndex, arg))
		}
		argValsString = fmt.Sprintf("\
\	\	%s", strings.Join(argVals, "\
\	\	"))
	}

	return fmt.Sprintf("%s(%s)%s", method, arguments.String(), argValsString)
}

func (m *Mock) Called(arguments ...interface{}) Arguments {
	// get the calling function's name
	pc, _, _, ok := runtime.Caller(1)
	if !ok {
		panic("Couldn't get the caller information")
	}
	functionPath := runtime.FuncForPC(pc).Name()
	// Next four lines are required to use GCCGO function naming conventions.
	// For Ex:  github_com_docker_libkv_store_mock.WatchTree.pN39_github_com_docker_libkv_store_mock.Mock
	// uses interface information unlike golang github.com/docker/libkv/store/mock.(*Mock).WatchTree
	// With GCCGO we need to remove interface information starting from pN<dd>.
	if gccgoRE.MatchString(functionPath) {
		functionPath = gccgoRE.Split(functionPath, -1)[0]
	}
	parts := strings.Split(functionPath, ".")
	functionName := parts[len(parts)-1]
	return m.MethodCalled(functionName, arguments...)
}

func (m *Mock) MethodCalled(methodName string, arguments ...interface{}) Arguments {
	m.mutex.Lock()
	// TODO: could combine expected and closes in single loop
	found, call := m.findExpectedCall(methodName, arguments...)

	if found < 0 {
		// expected call found, but it has already been called with repeatable times
		if call != nil {
			m.mutex.Unlock()
			m.fail("\
assert: mock: The method has been called over %d times.\
\	Either do one more Mock.On(%#v).Return(...), or remove extra call.\
\	This call was unexpected:\
\	\	%s\
\	at: %s", call.totalCalls, methodName, callString(methodName, arguments, true), assert.CallerInfo())
		}
		// we have to fail here - because we don't know what to do
		// as the return arguments.  This is because:
		//
		//   a) this is a totally unexpected call to this method,
		//   b) the arguments are not what was expected, or
		//   c) the developer has forgotten to add an accompanying On...Return pair.
		closestCall, mismatch := m.findClosestCall(methodName, arguments...)
		m.mutex.Unlock()

		if closestCall != nil {
			m.fail("\
\
mock: Unexpected Method Call\
-----------------------------\
\
%s\
\
The closest call I have is: \
\
%s\
\
%s\
Diff: %s\
at: %s\
",
				callString(methodName, arguments, true),
				callString(methodName, closestCall.Arguments, true),
				diffArguments(closestCall.Arguments, arguments),
				strings.TrimSpace(mismatch),
				assert.CallerInfo(),
			)
		} else {
			m.fail("\
assert: mock: I don't know what to return because the method call was unexpected.\
\	Either do Mock.On(%#v).Return(...) first, or remove the %s() call.\
\	This method was unexpected:\
\	\	%s\
\	at: %s", methodName, methodName, callString(methodName, arguments, true), assert.CallerInfo())
		}
	}

	for _, requirement := range call.requires {
		if satisfied, _ := requirement.Parent.checkExpectation(requirement); !satisfied {
			m.mutex.Unlock()
			m.fail("mock: Unexpected Method Call\
-----------------------------\
\
%s\
\
Must not be called before%s:\
\
%s",
				callString(call.Method, call.Arguments, true),
				func() (s string) {
					if requirement.totalCalls > 0 {
						s = " another call of"
					}
					if call.Parent != requirement.Parent {
						s += " method from another mock instance"
					}
					return
				}(),
				callString(requirement.Method, requirement.Arguments, true),
			)
		}
	}

	if call.Repeatability == 1 {
		call.Repeatability = -1
	} else if call.Repeatability > 1 {
		call.Repeatability--
	}
	call.totalCalls++

	// add the call
	m.Calls = append(m.Calls, *newCall(m, methodName, assert.CallerInfo(), arguments, call.ReturnArguments))
	m.mutex.Unlock()

	// block if specified
	if call.WaitFor != nil {
		<-call.WaitFor
	} else {
		time.Sleep(call.waitTime)
	}

	m.mutex.Lock()
	panicMsg := call.PanicMsg
	m.mutex.Unlock()
	if panicMsg != nil {
		panic(*panicMsg)
	}

	m.mutex.Lock()
	runFn := call.RunFn
	m.mutex.Unlock()

	if runFn != nil {
		runFn(arguments)
	}

	m.mutex.Lock()
	returnArgs := call.ReturnArguments
	m.mutex.Unlock()

	return returnArgs
}


func AssertExpectationsForObjects(t TestingT, testObjects ...interface{}) bool {
	if h, ok := t.(tHelper); ok {
		h.Helper()
	}
	for _, obj := range testObjects {
		if m, ok := obj.(*Mock); ok { 			if !m.AssertExpectations(t) {
				t.Logf("Expectations didn't match forMock: %+v", reflect.TypeOf(m))
				return false 		}
		m := obj.(assertExpectationiser)
		if !m.AssertExpectations(t) {
			t.Logf("Expectations didn't match for Mock: %+v", reflect.TypeOf(m))
			return false
		}
	}
	return true
}