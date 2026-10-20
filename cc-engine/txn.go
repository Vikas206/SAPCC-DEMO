package ccengine

// Transaction is one workload operation (YCSB or TPC-C). Keys must be unique.
type Transaction struct {
	ID            string
	Keys          []string
	Writes        []bool // Writes[i] => Keys[i] is written
	SemanticType  string // e.g. "Payment", "YCSB_A"
	SemanticText  string // description fed to the semantic classifier
	WorkNanos     int64  // simulated processing time between read and commit
	LockWaitNanos int64  // simulated per-lock acquisition cost (cross-shard RTT stand-in)
}

type Result struct {
	TxnID       string
	Strategy    string
	Committed   bool
	Aborted     bool
	Retried     bool
	UsedLocking bool
	LatencyNs   int64
}

func (r Result) WastedWork() bool { return r.Aborted || r.Retried }

type CCStrategy interface {
	Name() string
	Execute(store *Store, txn *Transaction) Result
}
