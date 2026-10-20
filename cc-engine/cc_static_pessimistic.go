package ccengine

import (
	"sync/atomic"
	"time"
)

// StaticPessimistic locks every key up front (sorted), paying a per-lock
// acquisition cost (txn.LockWaitNanos) that models a real lock-manager RTT,
// then holds through the work window and commits. Never aborts.
type StaticPessimistic struct{}

func (StaticPessimistic) Name() string { return "static_pessimistic" }

func (StaticPessimistic) Execute(store *Store, txn *Transaction) Result {
	start := time.Now()
	states := lockSorted(store, txn.Keys, txn.LockWaitNanos)
	defer unlockAll(states)

	if txn.WorkNanos > 0 {
		time.Sleep(time.Duration(txn.WorkNanos))
	}
	for i, k := range txn.Keys {
		if txn.Writes[i] {
			ks := store.get(k)
			ks.Value = "v"
			atomic.AddInt64(&ks.Version, 1)
		}
	}
	return Result{TxnID: txn.ID, Strategy: "static_pessimistic", Committed: true, UsedLocking: true, LatencyNs: time.Since(start).Nanoseconds()}
}
