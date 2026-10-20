package ccengine

// ReactiveSwitch is the prior-art baseline (US 6,681,226 style): run
// optimistically, and only AFTER a conflict is detected retry under locks.
type ReactiveSwitch struct{}

func (ReactiveSwitch) Name() string { return "reactive_switch" }

func (ReactiveSwitch) Execute(store *Store, txn *Transaction) Result {
	first := StaticOCC{}.Execute(store, txn)
	if first.Committed {
		first.Strategy = "reactive_switch"
		return first
	}
	second := StaticPessimistic{}.Execute(store, txn)
	return Result{TxnID: txn.ID, Strategy: "reactive_switch", Committed: second.Committed,
		Retried: true, UsedLocking: true, LatencyNs: first.LatencyNs + second.LatencyNs}
}
