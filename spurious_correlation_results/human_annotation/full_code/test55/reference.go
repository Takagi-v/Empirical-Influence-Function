package main


func NewTxnCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "txn [options]",
		Short: "Txn processes all the requests in one transaction",
		Long: `Txn reads multiple etcd requests from standard input and applies them as a single atomic transaction.
 A transaction consists of three components:
1) a list of conditions,
2) a list of requests to apply if all the conditions are true,
3) a list of requests to apply if any condition is false. 
Example interactive stdin usage:

---
etcdctl txn -i
# compares:
mod("key1") > "0"

# success requests (get, put, delete):
put key1 "overwrote-key1"

# failure requests (get, put, delete):
put key1 "created-key1"
put key2 "some extra key"
---

Refer to https://github.com/etcd-io/etcd/blob/main/etcdctl/README.md#txn-options.`,
		Run:     txnCommandFunc,
		GroupID: groupKVID,
	}
	cmd.Flags().BoolVarP(&txnInteractive, "interactive", "i", false, "Input transaction in interactive mode")
	return cmd
}