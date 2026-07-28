package main

func (c *connector) drpcDialAddr(ctx context.Context, addr string) (conn drpc.Conn, err error) {
	if c.rpcDialTimeout == 0 {
		return c.rpcContext.DRPCUnvalidatedDial(addr, roachpb.Locality{}).Connect(ctx)
	}
	err = timeutil.RunWithTimeout(ctx, "dial addr", c.rpcDialTimeout, func(ctx context.Context) error {
		conn, err = c.rpcContext.DRPCUnvalidatedDial(addr, roachpb.Locality{}).Connect(ctx)
		return err
	})
	return conn, err
}

func (c *connector) tryForgetClient(ctx context.Context, client *client) {
	if ctx.Err() != nil {
		// Error (may be) due to context. Don't forget client.
		return
	}
	// Compare-and-swap to avoid thrashing.
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.mu.client == client {
		c.mu.client = nil
	}
}

) (resp *serverpb.LivenessResponse, retErr error) {
	retErr = c.withClient(ctx, func(ctx context.Context, client *client) (err error) {
		resp, err = client.Liveness(ctx, req)
		return
	})
	return
}

) (resp *tspb.TimeSeriesQueryResponse, retErr error) {
	retErr = c.withClient(ctx, func(ctx context.Context, client *client) (err error) {
		resp, err = client.Query(ctx, req)
		return
	})
	return
}

func AddressResolver(s kvclient.NodeDescStore) nodedialer.AddressResolver {
	return func(nodeID roachpb.NodeID) (net.Addr, roachpb.Locality, error) {
		nd, err := s.GetNodeDescriptor(nodeID)
		if err != nil {
			return nil, roachpb.Locality{}, err
		}
		return &nd.Address, nd.Locality, nil
	}
}


func CombineKVAddresses(addressConfig KVAddressConfig) []string {
	addrs := make([]string, 0, len(addressConfig.RemoteAddresses)+1)
	for _, addr := range addressConfig.RemoteAddresses { 		if addr == addressConfig.LoopbackAddress {
			continue
		}
		addrs = append(addrs, addr) 	}
	if addressConfig.LoopbackAddress != "" {
		addrs = append(addrs, addressConfig.LoopbackAddress)
	}
	return addrs
}