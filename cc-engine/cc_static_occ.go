package ccengine

import (
	"sort"
	"sync/atomic"
	"time"
)

type StaticOCC struct{}

func (StaticOCC) Name() string { return "static_occ" }

func lockSorted(store *Store, keys []string, waitNanos int64) []*KeyState {
	sorted := append([]string(nil), keys...)
	sort.Strings(sorted) // global order => no deadlock
	states := make([]*KeyState, len(sorted))
	for i, k := range sorted {
		if waitNanos > 0 {
			time.Sleep(time.Duration(waitNanos)) // stand-in for lock-acquisition RTT
		}
		ks := store.get(k)
		ks.mu.Lock()
		states[i] = ks
	}
	return states
}

func unlockAll(states []*KeyState) {
	for _, ks := range states {
		ks.mu.Unlock()
	}
}

func (StaticOCC) Execute(store *Store, txn *Transaction) Result {
	start := time.Now()
	before := store.snapshotVersions(txn.Keys)
	if txn.WorkNanos > 0 {
		time.Sleep(time.Duration(txn.WorkNanos))
	}
	states := lockSorted(store, txn.Keys, 0) // validation lock is local, not a real RTT
	defer unlockAll(states)

	for _, k := range txn.Keys {
		if atomic.LoadInt64(&store.get(k).Version) != before[k] {
			return Result{TxnID: txn.ID, Strategy: "static_occ", Aborted: true, LatencyNs: time.Since(start).Nanoseconds()}
		}
	}
	for i, k := range txn.Keys {
		if txn.Writes[i] {
			ks := store.get(k)
			ks.Value = "v"
			atomic.AddInt64(&ks.Version, 1)
		}
	}
	return Result{TxnID: txn.ID, Strategy: "static_occ", Committed: true, LatencyNs: time.Since(start).Nanoseconds()}
}
