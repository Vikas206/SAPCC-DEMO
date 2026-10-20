package ccengine

import (
	"strconv"
	"sync"
	"sync/atomic"
)

// KeyState holds one key's value, a version counter (OCC validation) and a
// mutex that doubles as the pessimistic lock.
type KeyState struct {
	mu      sync.Mutex
	Value   string
	Version int64 // accessed atomically
}

// Store is an in-memory shard. In the full ShardRaft system this sits behind
// the Raft log; here it isolates the concurrency-control layer for benchmarking.
type Store struct {
	data map[string]*KeyState // read-only after construction
}

func KeyName(i int) string { return "key" + strconv.Itoa(i) }

// NewStoreFromKeys pre-populates exactly the given key universe. Every key a
// workload generator can reference must exist here first, or a lookup
// returns nil (TPC-C's structured keys like "W3D5C120" are NOT the same as
// YCSB's numeric "key7" -- each generator's key universe must be built with
// its own matching function, not an arbitrary numKeys count).
func NewStoreFromKeys(keys []string) *Store {
	s := &Store{data: make(map[string]*KeyState, len(keys))}
	for _, k := range keys {
		s.data[k] = &KeyState{Value: "init"}
	}
	return s
}

// AllKeyNames returns YCSB's numeric key universe: key0..key{n-1}.
func AllKeyNames(n int) []string {
	out := make([]string, n)
	for i := 0; i < n; i++ {
		out[i] = KeyName(i)
	}
	return out
}

func (s *Store) get(key string) *KeyState { return s.data[key] }

func (s *Store) snapshotVersions(keys []string) map[string]int64 {
	out := make(map[string]int64, len(keys))
	for _, k := range keys {
		out[k] = atomic.LoadInt64(&s.get(k).Version)
	}
	return out
}
